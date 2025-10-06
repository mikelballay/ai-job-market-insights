from __future__ import annotations

import argparse
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from xgboost import XGBClassifier

# ------------------------
# Funciones auxiliares (sin lambdas -> picklable)
# ------------------------


def pick_text_cols(X: pd.DataFrame) -> pd.DataFrame:
    return X[["title", "description"]]


def join_title_desc_df(X: pd.DataFrame) -> np.ndarray:
    t = X["title"].fillna("").astype(str)
    d = X["description"].fillna("").astype(str)
    return (t + " " + d).values


def pick_skills_col(X: pd.DataFrame) -> pd.Series:
    return X["skills"]


def normalize_skills(x):
    """Normaliza la celda 'skills' a una lista de strings."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [str(s) for s in x.tolist() if str(s).strip()]
    # NaN flotante
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(s) for s in x if str(s).strip()]
    if isinstance(x, (set, tuple)):
        return [str(s) for s in x if str(s).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def normalize_skills_series(s: pd.Series) -> pd.Series:
    """Aplica normalize_skills elemento a elemento (función top-level -> picklable)."""
    return s.apply(normalize_skills)


def skills_to_dict(lsts: List[List[str] | None]) -> List[Dict[str, int]]:
    """Convierte lista de skills en dict binario para DictVectorizer."""
    out: List[Dict[str, int]] = []
    iterable = lsts.tolist() if isinstance(lsts, pd.Series) else lsts
    for lst in iterable:
        if lst is None or (isinstance(lst, float) and pd.isna(lst)):
            out.append({})
        else:
            if isinstance(lst, np.ndarray):
                lst = lst.tolist()
            elif not isinstance(lst, (list, tuple, set)):
                lst = [lst]
            out.append({str(s): 1 for s in lst if str(s).strip()})
    return out


def present_classes(y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> List[str]:
    present_idx = np.unique(np.concatenate([y_true, y_pred], axis=0))
    return [str(c) for c in le.inverse_transform(present_idx)]


# ------------------------
# Model zoo
# ------------------------


def make_classifier(name: str):
    name = (name or "auto").lower()
    if name in ("lr", "logreg", "logistic"):
        return LogisticRegression(max_iter=2000)
    if name in ("rf", "randomforest", "forest"):
        return RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)
    if name in ("xgb", "xgboost"):
        return XGBClassifier(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=0,
        )
    # auto por defecto
    return XGBClassifier(
        n_estimators=350,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=0,
    )


# ------------------------
# Pipeline
# ------------------------


def make_pipeline(clf) -> Pipeline:
    text_union = FeatureUnion(
        [
            (
                "tfidf_word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40000,
                ),
            ),
            (
                "tfidf_char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                ),
            ),
        ]
    )

    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        # Texto: selecciona columnas, une y vectoriza
                        (
                            "text",
                            Pipeline(
                                [
                                    (
                                        "pick_text_cols",
                                        FunctionTransformer(pick_text_cols, validate=False),
                                    ),
                                    (
                                        "join",
                                        FunctionTransformer(join_title_desc_df, validate=False),
                                    ),
                                    ("vecs", text_union),
                                ]
                            ),
                        ),
                        # Skills: selecciona, normaliza, convierte a dict y vectoriza
                        (
                            "skills",
                            Pipeline(
                                [
                                    (
                                        "pick_skills",
                                        FunctionTransformer(pick_skills_col, validate=False),
                                    ),
                                    (
                                        "normalize",
                                        FunctionTransformer(
                                            normalize_skills_series, validate=False
                                        ),
                                    ),
                                    (
                                        "to_dict",
                                        FunctionTransformer(skills_to_dict, validate=False),
                                    ),
                                    ("dv", DictVectorizer(sparse=True)),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", clf),
        ]
    )
    return pipe


# ------------------------
# Main
# ------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train+Eval role classifier (robusto para clases raras)."
    )
    parser.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Parquet con columnas: title, description, skills, role_label",
    )
    parser.add_argument("--out-model", dest="out_model", required=True)
    parser.add_argument("--out-report-train", dest="out_report_train", required=True)
    parser.add_argument("--out-report-test", dest="out_report_test", required=True)
    parser.add_argument("--out-cm-test", dest="out_cm_test", required=True)
    parser.add_argument("--clf", dest="clf", default="auto", help="auto|lr|rf|xgb")
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--cv", dest="cv", type=int, default=5)
    parser.add_argument("--cap", dest="cap", type=int, default=0, help="Opcional: limitar filas")
    parser.add_argument(
        "--min-class-count",
        dest="min_class_count",
        type=int,
        default=2,
        help="Mínimo por clase; inferiores se mapean a 'other'",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.inp)
    if args.cap and args.cap > 0:
        df = df.sample(n=min(args.cap, len(df)), random_state=42).reset_index(drop=True)

    required = {"title", "description", "skills", "role_label"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise SystemExit(f"Faltan columnas en el dataset: {missing}")

    # Normaliza y reagrupa clases raras a 'other'
    df["role_label"] = df["role_label"].astype(str)
    counts = df["role_label"].value_counts()
    rare = counts[counts < args.min_class_count].index.tolist()
    if rare:
        df["role_label"] = df["role_label"].where(~df["role_label"].isin(rare), other="other")

    # Si aún quedan clases con <2, se eliminan
    counts2 = df["role_label"].value_counts()
    if (counts2 < 2).any():
        drop_classes = counts2[counts2 < 2].index.tolist()
        before = len(df)
        df = df[~df["role_label"].isin(drop_classes)].reset_index(drop=True)
        print(
            f"[WARN] Eliminadas clases con <2 muestras: {drop_classes} (de {before} -> {len(df)})"
        )

    # X (DataFrame) e y
    X = df[["title", "description", "skills"]].copy()
    X["skills"] = normalize_skills_series(X["skills"])

    y = df["role_label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split (estratificado si se puede)
    min_count = pd.Series(y).value_counts().min()
    strat = y_enc if min_count >= 2 else None
    if strat is None:
        print("[WARN] Split sin estratificar (alguna clase <2).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=42, stratify=strat
    )

    clf = make_classifier(args.clf)
    pipe = make_pipeline(clf)

    # CV segura: n_splits ≤ min_count_en_train
    min_count_train = pd.Series(le.inverse_transform(y_train)).value_counts().min()
    n_splits = max(2, min(args.cv, int(min_count_train))) if min_count_train >= 2 else 0
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        print(f"[CV {n_splits}-fold] F1-macro mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    else:
        print("[WARN] No se realiza CV (min_count_train < 2).")

    # Fit + eval
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    # Reports (solo clases presentes en cada split)
    present_train = present_classes(y_train, y_pred_train, le)
    present_test = present_classes(y_test, y_pred_test, le)
    labels_train = le.transform(present_train)
    labels_test = le.transform(present_test)

    report_train = classification_report(
        y_train, y_pred_train, labels=labels_train, target_names=present_train, digits=3
    )
    report_test = classification_report(
        y_test, y_pred_test, labels=labels_test, target_names=present_test, digits=3
    )

    os.makedirs(os.path.dirname(args.out_report_train), exist_ok=True)
    with open(args.out_report_train, "w", encoding="utf-8") as f:
        f.write(report_train)
    with open(args.out_report_test, "w", encoding="utf-8") as f:
        f.write(report_test)

    # Matriz de confusión (solo con clases presentes en test)
    cm = confusion_matrix(y_test, y_pred_test, labels=labels_test)
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 5), dpi=130)
        ax = fig.add_subplot(111)
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix (Test)")
        ax.set_xticks(range(len(present_test)))
        ax.set_xticklabels(present_test, rotation=45, ha="right")
        ax.set_yticks(range(len(present_test)))
        ax.set_yticklabels(present_test)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out_cm_test), exist_ok=True)
        fig.savefig(args.out_cm_test, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] No se pudo guardar la matriz de confusión: {e}")

    # Guardar modelo + encoder (ya sin lambdas en el pipeline)
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump({"pipeline": pipe, "label_encoder": le}, args.out_model)
    print("✅ Done.")
    print("Train report:\n", report_train)
    print("Test report:\n", report_test)


if __name__ == "__main__":
    main()
