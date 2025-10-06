from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass
class Query:
    text: str
    location: str | None = None
    limit: int = 100


class BaseScraper:
    source_name: str = "base"

    def fetch(self, query: Query) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
