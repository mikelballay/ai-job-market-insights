from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

import pandas as pd


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    import json

    # lee manejando posible BOM en Windows
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        return []

    # Caso 1: el archivo es un array JSON [ {...}, {...} ]
    if text[0] == "[":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]

    # Caso 2: JSONL (un objeto por línea)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A veces vienen varios objetos pegados en una misma línea "}{".
            # Intentamos separar heurísticamente:
            glued = line.replace("}{", "}~{").split("~")
            if len(glued) > 1:
                for chunk in glued:
                    rows.append(json.loads(chunk))
            else:
                raise
    return rows


def normalize_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for col in ["title", "company", "location", "posted_date", "description", "url", "source"]:
        if col not in df.columns:
            df[col] = None
    if "posted_date" in df.columns:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    return df


def to_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
