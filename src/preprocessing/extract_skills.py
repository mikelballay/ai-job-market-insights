from __future__ import annotations

import argparse
import re

import pandas as pd

# Diccionario de skills (ampliado y normalizado)
SKILL_PATTERNS = {
    "python": r"\bpython\b",
    "pandas": r"\bpandas\b",
    "numpy": r"\bnumpy\b",
    "sql": r"\bsql\b",
    "aws": r"\baws\b",
    "gcp": r"\bgcp\b",
    "azure": r"\bazure\b",
    "spark": r"\bspark\b",
    "hadoop": r"\bhadoop\b",
    "tensorflow": r"\btensorflow\b",
    "pytorch": r"\bpytorch\b",
    "scikit-learn": r"\bscikit[- ]?learn\b",
    "mlflow": r"\bmlflow\b",
    "airflow": r"\bairflow\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b",
    "tableau": r"\btableau\b",
    "powerbi": r"\bpower\s*bi\b",
}


def extract_skills(text: str) -> list[str]:
    """Extrae skills presentes en el texto usando regex."""
    if not isinstance(text, str):
        return []
    found = []
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(skill)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    df["skills"] = df["description"].apply(extract_skills)
    df.to_parquet(args.out_path, index=False)
    print(f"✅ Features con skills guardadas en {args.out_path}")


if __name__ == "__main__":
    main()
