from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseScraper, Query


def _get_query_text(q: Query) -> str:
    for attr in ("query", "q", "term", "text"):
        v = getattr(q, attr, None)
        if v:
            return str(v)
    return ""


def _get_query_limit(q: Query) -> int | None:
    return getattr(q, "limit", None)


def _strip_html(html: str) -> str:
    # Limpieza rápida sin dependencias: elimina tags y contrae espacios
    txt = re.sub(r"<[^>]+>", " ", html or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


@dataclass
class _Job:
    title: str
    company: str
    location: str
    date: str
    description: str
    url: str


class RemotiveScraper(BaseScraper):
    """
    API pública Remotive:
      GET https://remotive.com/api/remote-jobs?search=<query>
    Respuesta: {"jobs": [ ... ]}
    """

    API_URL = "https://remotive.com/api/remote-jobs"

    def __init__(self) -> None:
        super().__init__()
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://remotive.com/",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _get_json(self, q_text: str) -> Dict[str, Any]:
        params = {}
        if q_text:
            params["search"] = q_text
        resp = requests.get(self.API_URL, headers=self._headers, params=params, timeout=25)
        resp.raise_for_status()
        return resp.json()

    def _to_iso_date(self, s: str | None) -> str:
        if not s:
            return datetime.now(timezone.utc).date().isoformat()
        try:
            # Formato típico: "2025-08-17T15:19:38"
            dt = datetime.fromisoformat(s)
            return dt.date().isoformat()
        except Exception:
            return datetime.now(timezone.utc).date().isoformat()

    def fetch(self, query: Query) -> Iterable[Dict[str, Any]]:
        q_text = _get_query_text(query)
        limit = _get_query_limit(query)
        data = self._get_json(q_text)
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        out: List[Dict[str, Any]] = []

        for j in jobs:
            title = (j.get("title") or "").strip() or "Unknown"
            company = (j.get("company_name") or "").strip() or "Unknown"
            location = (j.get("candidate_required_location") or "Remote").strip()
            posted_date = self._to_iso_date(j.get("publication_date"))
            description = _strip_html(j.get("description") or "")
            url = (j.get("url") or "").strip()

            out.append(
                {
                    "title": title,
                    "company": company,
                    "location": location,
                    "posted_date": posted_date,
                    "description": description,
                    "url": url,
                    "source": "remotive",
                }
            )

            if limit and len(out) >= int(limit):
                break

        return out
