import re

import pandas as pd

df = pd.read_parquet("data/processed/jobs_clean.parquet")

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


def infer(title):
    if not isinstance(title, str):
        return "other"
    for label, pat in ROLE_PATTERNS:
        if pat.search(title):
            return label
    return "other"


df["role_label"] = df["title"].map(infer)
print(df["role_label"].value_counts())
