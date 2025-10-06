from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List

from ..common.io import normalize_records, read_jsonl, to_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Build normalized dataset (Parquet) from raw JSONL files."
    )
    parser.add_argument(
        "--in", dest="input_dir", required=True, help="Directory with raw *.jsonl files"
    )
    parser.add_argument("--out", dest="out_path", required=True, help="Output Parquet path")
    args = parser.parse_args()

    files = [p for p in glob.glob(os.path.join(args.input_dir, "*.jsonl")) if os.path.isfile(p)]
    if not files:
        raise SystemExit(f"No JSONL files found in {args.input_dir}")

    records: List[Dict[str, Any]] = []
    for fp in files:
        records.extend(read_jsonl(fp))

    df = normalize_records(records)
    df = df.dropna(subset=["title", "company"]).reset_index(drop=True)
    if "location" in df.columns:
        df["location"] = df["location"].astype(str)

    to_parquet(df, args.out_path)
    print(f"Wrote normalized dataset with {len(df)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
