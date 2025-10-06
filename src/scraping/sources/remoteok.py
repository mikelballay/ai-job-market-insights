from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import requests
from bs4 import BeautifulSoup  # (no se usa ya, pero lo dejamos si otros scrapers lo necesitan)
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseScraper, Query


class RemoteOKScraper(BaseScraper):
    """
    Scraper basado en la API pública JSON de RemoteOK para evitar 403 al scrapear HTML.

    Docs no oficiales: GET https://remoteok.com/api  -> lista de jobs (primero suele ser metadata)
    Campos útiles habituales en cada job:
      - "position" (título), "company", "location", "tags" (lista),
      - "url" (ruta relativa), "apply_url" (a veces), "description", "date" (ISO).
    """

    API_URL = "https://remoteok.com/api"
    BASE_URL = "https://remoteok.com"

    def __init__(self) -> None:
        super().__init__()
        try:
            self.ua = UserAgent()
            ua = self.ua.random
        except Exception:
            ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        self._headers = {
            "User-Agent": ua,
            "Accept": "application/json, text/plain, */*",
            "Referer": self.BASE_URL + "/",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _get_json(self) -> List[Dict[str, Any]]:
        resp = requests.get(self.API_URL, headers=self._headers, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        # A veces el primer elemento es metadata (dict con "legal" o similar); filtramos para quedarnos con jobs
        if isinstance(data, list):
            jobs = [x for x in data if isinstance(x, dict) and ("position" in x or "title" in x)]
        else:
            jobs = []
        return jobs

    def _norm(self, s: Any) -> str:
        return (s or "").strip()

    def _to_iso(self, val: Any) -> str:
        """
        Intenta convertir a ISO8601. RemoteOK suele dar "date" (ISO) o "epoch".
        """
        if not val:
            # fallback: ahora mismo en UTC
            return datetime.now(timezone.utc).date().isoformat()
        # Si ya parece ISO:
        if isinstance(val, str) and len(val) >= 10 and val[4] == "-" and val[7] == "-":
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                pass
        # Epoch en segundos:
        try:
            ts = float(val)
            return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except Exception:
            return datetime.now(timezone.utc).date().isoformat()

    def _build_url(self, job: Dict[str, Any]) -> str:
        # Preferimos 'url' absoluto si lo trae; si es relativo, lo unimos al BASE_URL.
        u = job.get("url") or job.get("apply_url") or ""
        u = str(u)
        if u.startswith("http"):
            return u
        if u.startswith("/"):
            return self.BASE_URL + u
        if u:
            return self.BASE_URL + "/" + u
        return self.BASE_URL

    def _match_query(self, job: Dict[str, Any], q: str) -> bool:
        """Filtro simple: busca la query en título, descripción, tags y company (lower)."""
        if not q:
            return True
        ql = q.lower()

        title = job.get("position") or job.get("title") or ""
        desc = job.get("description") or ""
        comp = job.get("company") or ""
        tags = job.get("tags") or []

        hay = " ".join(
            [
                str(title).lower(),
                str(desc).lower(),
                str(comp).lower(),
                " ".join([str(t).lower() for t in tags]),
            ]
        )
        return ql in hay

    def fetch(self, query: Query) -> Iterable[Dict[str, Any]]:
        """
        Devuelve dicts con este esquema:
          - title, company, location, posted_date (YYYY-MM-DD), description, url, source='remoteok'
        Respeta query.limit si viene informado.
        """
        jobs = self._get_json()
        out: List[Dict[str, Any]] = []

        for j in jobs:
            if not self._match_query(j, query.q or ""):
                continue

            title = self._norm(j.get("position") or j.get("title"))
            company = self._norm(j.get("company"))
            location = self._norm(j.get("location") or j.get("region") or "Remote")
            posted_date = self._to_iso(j.get("date") or j.get("epoch"))
            # description suele venir en HTML; lo convertimos a texto plano breve
            raw_desc = j.get("description") or ""
            try:
                text_desc = BeautifulSoup(raw_desc, "html.parser").get_text(
                    separator=" ", strip=True
                )
            except Exception:
                text_desc = str(raw_desc)
            url = self._build_url(j)

            row = {
                "title": title or "Unknown",
                "company": company or "Unknown",
                "location": location,
                "posted_date": posted_date,
                "description": text_desc,
                "url": url,
                "source": "remoteok",
            }
            out.append(row)

            if query.limit and len(out) >= int(query.limit):
                break

        return out
