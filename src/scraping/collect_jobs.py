from __future__ import annotations

import argparse

from .sources.base import Query
from .sources.mock import MockScraper

# Fuentes opcionales
try:
    from .sources.remoteok import RemoteOKScraper
except Exception:
    RemoteOKScraper = None

try:
    from .sources.remotive import RemotiveScraper
except Exception:
    RemotiveScraper = None


def _make_query(args) -> Query:
    """
    Crea Query por posición para evitar errores de nombre de campo.
    Soporta cualquier firma de Query con 2 primeros campos: (texto, limit).
    """
    try:
        # Posicional: texto, limit (independiente del nombre real de atributos)
        return Query(args.query, args.limit)
    except TypeError:
        # Fallback: intenta sin limit por si Query solo tiene 1 arg
        return Query(args.query)


def main():
    parser = argparse.ArgumentParser(description="Collect jobs into JSONL from selected source.")
    parser.add_argument(
        "--query", dest="query", required=False, default="", help="Texto de búsqueda"
    )
    # Aceptamos --location para compatibilidad con README/tests, aunque no se use en mock
    parser.add_argument(
        "--location", dest="location", required=False, default="", help="Ubicación (opcional)"
    )
    parser.add_argument("--out", dest="out", required=True, help="Ruta de salida JSONL")
    parser.add_argument(
        "--source",
        dest="source",
        required=False,
        default="mock",
        choices=["mock", "remoteok", "remotive", "auto"],
        help="Fuente de scraping",
    )
    parser.add_argument(
        "--limit", dest="limit", required=False, type=int, default=50, help="Máximo de filas"
    )
    args = parser.parse_args()

    q = _make_query(args)

    if args.source == "mock":
        rows = list(MockScraper().fetch(q))

    elif args.source == "remoteok":
        if RemoteOKScraper is None:
            raise SystemExit("RemoteOKScraper no disponible.")
        rows = list(RemoteOKScraper().fetch(q))

    elif args.source == "remotive":
        if RemotiveScraper is None:
            raise SystemExit("RemotiveScraper no disponible.")
        rows = list(RemotiveScraper().fetch(q))

    elif args.source == "auto":
        rows = []
        if RemotiveScraper is not None:
            try:
                rows = list(RemotiveScraper().fetch(q))
            except Exception:
                rows = []
        if not rows and RemoteOKScraper is not None:
            try:
                rows = list(RemoteOKScraper().fetch(q))
            except Exception:
                rows = []
        if not rows:
            rows = list(MockScraper().fetch(q))
    else:
        raise SystemExit(f"Fuente desconocida: {args.source}")

    from ..common.io import write_jsonl

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} jobs to {args.out} using source='{args.source}'")


if __name__ == "__main__":
    main()
