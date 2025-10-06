from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Importar helpers usados en el pipeline entrenado y mapearlos a __main__ si el
# modelo fue serializado desde un script ejecutado como __main__.
try:
    from src.analysis.train_eval import (
        pick_text_cols,
        join_title_desc_df,
        pick_skills_col,
        normalize_skills_series,
        skills_to_dict,
    )
    _main_mod = sys.modules.get("__main__")
    if _main_mod is not None:
        setattr(_main_mod, "pick_text_cols", pick_text_cols)
        setattr(_main_mod, "join_title_desc_df", join_title_desc_df)
        setattr(_main_mod, "pick_skills_col", pick_skills_col)
        setattr(_main_mod, "normalize_skills_series", normalize_skills_series)
        setattr(_main_mod, "skills_to_dict", skills_to_dict)
except Exception:
    # Si no existen (modelo entrenado con otra versión), lo ignoramos; joblib
    # intentará cargar igualmente y si falta fallará con error claro.
    pass

st.set_page_config(
    page_title="AI Job Market Insights — Dashboard",
    page_icon="📊",
    layout="wide",
)

# =========================
# Sidebar / Config
# =========================
st.sidebar.header("⚙️ Configuración")
data_path = Path(
    st.sidebar.text_input(
        "Ruta del parquet procesado",
        value="data/processed/jobs_features_skills.parquet",
    )
)
infer_labels_if_missing = st.sidebar.checkbox(
    "Inferir 'role_label' desde el título si falta", value=True
)
model_path = Path(st.sidebar.text_input("Ruta del modelo (.pkl)", value="models/role_clf.pkl"))

st.sidebar.caption(
    "Columnas recomendadas en el parquet: 'title', "
    "'description' o 'cleaned_text', 'skills', 'role_label' (opcional), 'posted_date' (opcional)."
)


# =========================
# Helpers
# =========================
def _normalize_skills_cell(x):
    """Devuelve siempre list[str] sin provocar truth-value errors."""
    if isinstance(x, (list, tuple)):
        return [str(s) for s in x]
    if isinstance(x, np.ndarray):
        return [str(s) for s in x.tolist()]
    if pd.isna(x):
        return []
    return [str(x)]


ROLE_PATTERNS = [
    ("data_scientist", re.compile(r"(?i)\bdata\s*scientist\b")),
    ("ml_engineer", re.compile(r"(?i)\b(ml|machine\s*learning)\s*engineer\b")),
    ("data_engineer", re.compile(r"(?i)\bdata\s*engineer\b")),
    ("risk_analyst", re.compile(r"(?i)\b(risk|credit)\s*(data\s*)?analyst\b")),
    ("quant_researcher", re.compile(r"(?i)\bquant(itative)?\s*(researcher|analyst)?\b")),
    ("mlops_engineer", re.compile(r"(?i)\bmlops\s*engineer\b")),
    ("cv_engineer", re.compile(r"(?i)\b(computer\s*vision|cv)\s*engineer\b")),
    # Español
    ("data_scientist", re.compile(r"(?i)\b(cient[ií]fic[oa])\s+de\s+datos\b")),
    (
        "ml_engineer",
        re.compile(
            r"(?i)\bingenier[oa]\s+de\s+(ml|aprendizaje\s+autom[aá]tico|machine\s*learning)\b"
        ),
    ),
    ("data_engineer", re.compile(r"(?i)\bingenier[oa]\s+de\s+datos\b")),
    ("risk_analyst", re.compile(r"(?i)\banalist[ae]\s+de\s+r(ie)?sgo(s)?\b")),
    (
        "quant_researcher",
        re.compile(r"(?i)\b(quant|cuantitativ[oa])\s+(investigador|analist[ae])\b"),
    ),
    ("mlops_engineer", re.compile(r"(?i)\bingenier[oa]\s+mlops\b")),
    (
        "cv_engineer",
        re.compile(r"(?i)\b(visi[oó]n\s+(por\s+)?computador(a)?|visi[oó]n\s+artificial)\b"),
    ),
]


def infer_role_label(title: str) -> str:
    if not isinstance(title, str):
        return "other"
    for label, pat in ROLE_PATTERNS:
        if pat.search(title):
            return label
    return "other"


@st.cache_data(show_spinner=True)
def load_data(p: Path, infer_if_missing: bool) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")
    df = pd.read_parquet(p)

    # Normalizar skills a lista
    if "skills" in df.columns:
        df["skills"] = df["skills"].apply(_normalize_skills_cell)

    # Fechas
    if "posted_date" in df.columns:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")

    # role_label (si falta y el usuario marcó inferir)
    if "role_label" not in df.columns and infer_if_missing:
        if "title" not in df.columns:
            st.warning("No hay 'role_label' ni 'title' para inferir etiquetas.")
        else:
            df["role_label"] = [infer_role_label(t) for t in df["title"]]

    return df


def explode_skills(df: pd.DataFrame) -> pd.DataFrame:
    if "skills" not in df.columns:
        return pd.DataFrame(columns=["skill"])
    out = df.explode("skills").rename(columns={"skills": "skill"})
    return out.dropna(subset=["skill"])


@st.cache_resource(show_spinner=False)
def load_model(p: Path):
    if not p.exists():
        return None, None
    bundle = joblib.load(p)
    return bundle.get("pipeline"), bundle.get("label_encoder")


def predict(pipe, le, title: str, description: str):
    # El pipeline entrenado espera DataFrame con columnas específicas
    X = pd.DataFrame(
        [{"title": title or "", "description": description or "", "skills": []}]
    )
    if X[["title", "description"]].astype(str).agg("".join, axis=1).str.strip().iloc[0] == "":
        return None, None
    yhat = pipe.predict(X)[0]
    label = le.inverse_transform([yhat])[0] if hasattr(le, "inverse_transform") else str(yhat)
    proba_df = None
    if hasattr(pipe, "predict_proba"):
        preds = pipe.predict_proba(X)[0]
        proba_df = (
            pd.DataFrame({"role": getattr(le, "classes_", []), "proba": preds})
            .sort_values("proba", ascending=False)
            .reset_index(drop=True)
        )
    return label, proba_df


# =========================
# Carga de datos
# =========================
st.title("📊 AI Job Market Insights — Dashboard (Fase 3)")

try:
    df = load_data(data_path, infer_labels_if_missing)
    st.success(f"Datos cargados: **{data_path}** — filas: {len(df):,}")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

has_role = "role_label" in df.columns
has_skills = "skills" in df.columns
has_date = "posted_date" in df.columns

# =========================
# Tabs
# =========================
tab_overview, tab_trends, tab_skills, tab_predict = st.tabs(
    ["📌 Distribución de roles", "📈 Tendencias", "🧩 Skills", "🎯 Predicción (modelo)"]
)

# -------------------------
# Tab 1: Distribución de roles
# -------------------------
with tab_overview:
    st.subheader("Distribución de roles")
    if has_role:
        role_counts = df["role_label"].value_counts().sort_values(ascending=False)
        st.bar_chart(role_counts)
        # Series -> DataFrame sin duplicar nombres: usa reset_index(name=...)
        role_counts_df = role_counts.reset_index(name="count").rename(columns={"index": "role"})
        # Asegurar unicidad de columnas por si el backend añade metadatos:
        role_counts_df.columns = pd.Index(role_counts_df.columns).map(str)
        role_counts_df = role_counts_df.loc[:, ~role_counts_df.columns.duplicated()]
        st.dataframe(role_counts_df, width="stretch")

        with st.expander("🔎 Muestras por rol"):
            roles_sel = st.multiselect(
                "Filtrar por rol",
                options=role_counts.index.tolist(),
                default=role_counts.index.tolist()[:3],
            )
            n_show = st.slider("Muestras a mostrar", 5, 50, 10)
            if roles_sel:
                cols = [
                    c
                    for c in ["title", "role_label", "company", "location", "url"]
                    if c in df.columns
                ]
                sample_df = df[df["role_label"].isin(roles_sel)][cols].head(n_show)
                st.dataframe(sample_df, width="stretch")
    else:
        st.warning(
            "No hay columna 'role_label' en los datos. Activa la casilla de la barra lateral para inferirla si quieres."
        )

# -------------------------
# Tab 2: Tendencias
# -------------------------
with tab_trends:
    st.subheader("Tendencias temporales")
    if has_date:
        df_date = df.dropna(subset=["posted_date"]).copy()
        if len(df_date) == 0:
            st.info("No hay fechas válidas tras el parseo.")
        else:
            df_date["month"] = df_date["posted_date"].dt.to_period("M").dt.to_timestamp()
            total_monthly = df_date.groupby("month").size()
            st.markdown("**Ofertas por mes (Total)**")
            st.line_chart(total_monthly)

            if has_role:
                roles_to_plot = st.multiselect(
                    "Roles a graficar",
                    options=sorted(df_date["role_label"].dropna().unique()),
                    default=sorted(df_date["role_label"].dropna().unique())[:3],
                )
                if roles_to_plot:
                    by_role = (
                        df_date[df_date["role_label"].isin(roles_to_plot)]
                        .groupby(["month", "role_label"])
                        .size()
                        .unstack(fill_value=0)
                        .sort_index()
                    )
                    st.line_chart(by_role)
    else:
        st.info("No hay columna 'posted_date' en los datos.")

# -------------------------
# Tab 3: Skills
# -------------------------
with tab_skills:
    st.subheader("Top skills")
    if has_skills:
        exploded = explode_skills(df)

        # Global
        top_k = st.number_input("Top K", min_value=5, max_value=50, value=20, step=1)
        top_global = exploded["skill"].value_counts().head(int(top_k))
        st.markdown("**Top skills (global)**")
        st.bar_chart(top_global)
        top_global_df = top_global.reset_index(name="count").rename(columns={"index": "skill"})
        top_global_df.columns = pd.Index(top_global_df.columns).map(str)
        top_global_df = top_global_df.loc[:, ~top_global_df.columns.duplicated()]
        st.dataframe(top_global_df, width="stretch")

        st.markdown("---")
        # Por rol
        if has_role:
            role_sel = st.selectbox("Rol", options=sorted(df["role_label"].dropna().unique()))
            top_role = (
                explode_skills(df[df["role_label"] == role_sel])["skill"]
                .value_counts()
                .head(int(top_k))
            )
            st.markdown(f"**Top skills — {role_sel}**")
            st.bar_chart(top_role)
            top_role_df = top_role.reset_index(name="count").rename(columns={"index": "skill"})
            top_role_df.columns = pd.Index(top_role_df.columns).map(str)
            top_role_df = top_role_df.loc[:, ~top_role_df.columns.duplicated()]
            st.dataframe(top_role_df, width="stretch")
        else:
            st.caption("Añade 'role_label' para ver top skills por rol.")
    else:
        st.warning("No hay columna 'skills' en los datos.")

# -------------------------
# Tab 4: Predicción
# -------------------------
with tab_predict:
    st.subheader("Predicción con modelo entrenado")
    pipe, le = load_model(model_path)
    if pipe is None or le is None:
        st.info(
            "No se encontró un modelo en la ruta indicada. Entrena y guarda como 'models/role_clf.pkl'."
        )
    else:
        col1, _ = st.columns([2, 1])
        with col1:
            title_txt = st.text_input("Título", value="Data Scientist")
        desc_txt = st.text_area(
            "Descripción",
            value="We are looking for a Data Scientist with Python, SQL and TensorFlow.",
            height=200,
        )
        if st.button("🔍 Predecir rol"):
            label, proba_df = predict(pipe, le, title_txt, desc_txt)
            if label:
                st.success(f"Rol estimado: **{label}**")
            if proba_df is not None and len(proba_df):
                st.dataframe(proba_df, width="stretch")
                st.bar_chart(proba_df.set_index("role"))
