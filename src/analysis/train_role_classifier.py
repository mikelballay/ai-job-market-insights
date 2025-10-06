from __future__ import annotations

import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from xgboost import XGBClassifier


# -------------------------
# Función principal
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out-model", dest="out_model", required=True)
    parser.add_argument("--out-report", dest="out_report", required=True)
    parser.add_argument("--out-cm", dest="out_cm", required=True)
    parser.add_argument("--clf", choices=["logreg", "rf", "xgb"], default="logreg")
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=0)
    args = parser.parse_args()

    # -------------------------
    # 1. Cargar datos
    # -------------------------
    df = pd.read_parquet(args.in_path)

    if args.cap:
        df = df.sample(n=args.cap, random_state=42)

    role_counts = df["role_label"].value_counts()
    keep_roles = role_counts[role_counts >= args.min_count].index
    df = df[df["role_label"].isin(keep_roles)]

    # -------------------------
    # 2. Preparar features
    # -------------------------
    X_text = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    X_skills = df["skills"].tolist()
    y = df["role_label"].tolist()

    # -------------------------
    # 3. Preprocesamiento
    # -------------------------
    mlb = MultiLabelBinarizer()
    _ = mlb.fit_transform(X_skills)

    # TF-IDF básico para texto (pipeline más sencillo en esta versión)

    # -------------------------
    # 4. Clasificador
    # -------------------------
    if args.clf == "logreg":
        clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1)
    elif args.clf == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1)
    elif args.clf == "xgb":
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError("Classifier no reconocido")

    # LabelEncoder para y
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # -------------------------
    # 5. Entrenar pipeline
    # -------------------------
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", clf),
        ]
    )

    pipe.fit(X_text, y_enc)

    # -------------------------
    # 6. Reporte
    # -------------------------
    y_pred = pipe.predict(X_text)
    report = classification_report(y_enc, y_pred, target_names=le.classes_, digits=3)
    print(report)

    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_enc, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(args.out_cm)
    plt.close()

    # -------------------------
    # 7. Guardar modelo
    # -------------------------
    joblib.dump({"pipeline": pipe, "label_encoder": le, "mlb": mlb}, args.out_model)
    print(f"✅ Modelo guardado en {args.out_model}")


if __name__ == "__main__":
    main()
