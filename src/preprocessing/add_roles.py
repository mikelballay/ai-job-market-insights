from __future__ import annotations

import argparse
import re

import pandas as pd

# Patrones EN/ES para mapear a roles estándar
ROLE_PATTERNS = [
    ("data_scientist", re.compile(r"(?i)\b(data\s*scientist|cient[ií]fico\s*de\s*datos)\b")),
    (
        "ml_engineer",
        re.compile(
            r"(?i)\b(ml\s*engineer|machine\s*learning\s*engineer|ingenier[oa]\s*de\s*machine\s*learning)\b"
        ),
    ),
    (
        "mlops_engineer",
        re.compile(
            r"(?i)\b(mlops|ml\s*ops|ml\s*operations|mlops\s*engineer|ml\s*ops\s*engineer)\b"
        ),
    ),
    ("data_engineer", re.compile(r"(?i)\b(data\s*engineer|ingenier[oa]\s*de\s*datos)\b")),
    (
        "cv_engineer",
        re.compile(
            r"(?i)\b(computer\s*vision|visi[oó]n\s*por\s*computador[ae]?|visi[oó]n\s*artificial)\b"
        ),
    ),
    (
        "quant_researcher",
        re.compile(
            r"(?i)\b(quant|quantitative\s+(research(er)?|analyst)|investigador\s*cuantitativ[oa])\b"
        ),
    ),
    ("risk_analyst", re.compile(r"(?i)\b(risk\s*analyst|analista\s*de\s*riesgo[s]?)\b")),
]


def map_text_to_role(title: str, description: str) -> str:
    t = (title or "").strip()
    d = (description or "").strip()
    text = f"{t} {d}"
    for label, pat in ROLE_PATTERNS:
        if pat.search(text):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser(
        description="Añade columna role_label a un Parquet (mantiene columnas existentes)."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Parquet de entrada")
    ap.add_argument("--out", dest="out", required=True, help="Parquet de salida con role_label")
    ap.add_argument(
        "--overwrite", action="store_true", help="Sobrescribir el archivo de salida si ya existe"
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)

    # Si ya existe role_label, lo respetamos (pero rellenamos vacíos si los hubiera)
    if "role_label" not in df.columns:
        df["role_label"] = [
            map_text_to_role(str(row.get("title", "")), str(row.get("description", "")))
            for _, row in df.iterrows()
        ]
    else:
        # Completa NaN con el mapeo
        mask = df["role_label"].isna() | (df["role_label"].astype(str).str.strip() == "")
        if mask.any():
            df.loc[mask, "role_label"] = [
                map_text_to_role(str(row.get("title", "")), str(row.get("description", "")))
                for _, row in df.loc[mask].iterrows()
            ]

    if (not args.overwrite) and args.inp == args.out:
        raise SystemExit("Usa --overwrite si quieres escribir sobre el mismo archivo de entrada.")

    df.to_parquet(args.out, index=False)
    print(f"✅ role_label añadido. Escribí {len(df)} filas en: {args.out}")


if __name__ == "__main__":
    main()
