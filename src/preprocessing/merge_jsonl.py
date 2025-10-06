from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Lee un JSONL de forma robusta, ignorando líneas vacías o corruptas (con aviso)."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{i} línea inválida, se ignora. Detalle: {e}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(s: Any) -> str:
    """Normaliza strings: quita espacios extra, pasa a minúsculas. Si no es str, lo castea."""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def norm_url(u: Any) -> str:
    """Normaliza URLs (quita anchors, normaliza esquema/host simple)."""
    u = norm(u)
    if not u:
        return ""
    # Quita fragment (#...)
    u = u.split("#", 1)[0]
    return u


def default_key(record: Dict[str, Any]) -> Tuple:
    """Clave por defecto para deduplicar: url si existe, sino (title, company, location, posted_date)."""
    url = norm_url(record.get("url"))
    if url:
        return ("url", url)
    return (
        "tuple",
        norm(record.get("title")),
        norm(record.get("company")),
        norm(record.get("location")),
        norm(record.get("posted_date")),
    )


def make_tuple_key(fields: List[str]):
    """Construye una función clave basada en un listado de campos."""

    def _key(record: Dict[str, Any]) -> Tuple:
        parts = tuple(norm(record.get(f)) for f in fields)
        return ("fields",) + parts

    return _key


def collect_input_files(inputs: List[str]) -> List[Path]:
    """Expande patrones glob y rutas; evita duplicados."""
    files: List[Path] = []
    seen = set()
    for pattern in inputs:
        # Si el patrón coincide con archivos, glob lo expande; si es ruta, también la recoge
        matches = glob.glob(pattern)
        if not matches and os.path.exists(pattern):
            matches = [pattern]
        for m in matches:
            p = Path(m).resolve()
            if p not in seen and p.is_file():
                files.append(p)
                seen.add(p)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Une múltiples JSONL con ofertas y elimina duplicados de forma robusta."
    )
    parser.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="Lista de rutas o patrones glob de entrada (ej: data/raw/*.jsonl data/raw/remoteok_*.jsonl)",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Ruta de salida JSONL combinado (ej: data/raw/remoteok_merged.jsonl)",
    )
    parser.add_argument(
        "--dedup-on",
        dest="dedup_on",
        nargs="*",
        default=None,
        help="Campos para deduplicar (ej: --dedup-on url) o (title company location posted_date). "
        "Si no se especifica, usa: url -> (title,company,location,posted_date).",
    )
    args = parser.parse_args()

    in_files = collect_input_files(args.inputs)
    if not in_files:
        raise SystemExit(
            "No se encontraron archivos de entrada con los patrones/rutas proporcionados."
        )

    # Decide la función de clave para deduplicar
    if args.dedup_on:
        key_fn = make_tuple_key(args.dedup_on)
        print(f"[INFO] Deduplicando por campos: {args.dedup_on}")
    else:
        key_fn = default_key
        print(
            "[INFO] Deduplicando por URL (si existe) o por (title, company, location, posted_date)."
        )

    seen = set()
    merged: List[Dict[str, Any]] = []

    total_in = 0
    for fp in in_files:
        count_file = 0
        for rec in read_jsonl(fp):
            total_in += 1
            count_file += 1
            k = key_fn(rec)
            if k in seen:
                continue
            seen.add(k)
            merged.append(rec)
        print(f"[INFO] {fp} -> {count_file} filas leídas")

    out_path = Path(args.out_path)
    write_jsonl(out_path, merged)
    print(f"[OK] Escribí {len(merged)} ofertas únicas (de {total_in} totales) en: {out_path}")


if __name__ == "__main__":
    main()
