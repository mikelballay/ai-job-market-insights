from __future__ import annotations

import argparse
import re

import pandas as pd


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r", "\n")
    t = re.sub(r"\s+", " ", t)
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}", " ", t)
    t = t.replace("`", " ")
    t = re.sub(r"[^a-z\s\+\#]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def try_spacy_lemmas(texts: list[str]) -> list[str]:
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            return texts
        out = []
        for doc in nlp.pipe(texts, disable=["ner", "parser"]):
            out.append(" ".join([t.lemma_ for t in doc if not t.is_stop and t.lemma_.strip()]))
        return out
    except Exception:
        return texts


def main():
    ap = argparse.ArgumentParser(description="Clean and optionally lemmatize job descriptions.")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--lemmatize", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    if "description" not in df.columns:
        raise SystemExit("Input parquet must contain 'description' column")

    df["cleaned_text"] = [basic_clean(x) for x in df["description"].tolist()]
    if args.lemmatize:
        df["cleaned_text"] = try_spacy_lemmas(df["cleaned_text"].tolist())

    df.to_parquet(args.out_path, index=False)
    print(f"Wrote cleaned dataset to {args.out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
