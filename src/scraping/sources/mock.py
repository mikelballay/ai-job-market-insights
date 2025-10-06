from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable

from .base import BaseScraper, Query

TITLES = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Risk Data Analyst",
    "MLOps Engineer",
    "Quant Researcher",
    "Computer Vision Engineer",
]
COMPANIES = [
    "Aurelius Analytics",
    "BlueWave AI",
    "IberData",
    "Madrid ML Labs",
    "Quantia",
    "RiskFlow",
]
LOCATIONS = [
    "Madrid, Spain",
    "Barcelona, Spain",
    "Valencia, Spain",
    "Remote, EU",
    "Seville, Spain",
    "Bilbao, Spain",
]
SKILLS = [
    "Python",
    "SQL",
    "Pandas",
    "PyTorch",
    "TensorFlow",
    "Docker",
    "Kubernetes",
    "Airflow",
    "AWS",
    "GCP",
    "Spark",
    "PowerBI",
    "Tableau",
    "MLflow",
]

DESCR = (
    "We are looking for a {title} to join {company}. "
    "You will build data products using {skills}. "
    "Experience with {extra} is a plus. "
    "Knowledge of statistics and ML best practices required."
)


class MockScraper(BaseScraper):
    source_name = "mock"

    def fetch(self, query: Query) -> Iterable[Dict[str, Any]]:
        random.seed(42)
        today = datetime.utcnow().date()
        for i in range(query.limit):
            title = random.choice(TITLES)
            company = random.choice(COMPANIES)
            location = random.choice(LOCATIONS)
            skills = ", ".join(random.sample(SKILLS, k=random.randint(4, 7)))
            extra = random.choice(SKILLS)
            days_ago = random.randint(0, 28)
            posted = today - timedelta(days=days_ago)
            yield {
                "title": f"{title} ({query.text})",
                "company": company,
                "location": location,
                "posted_date": posted.isoformat(),
                "description": DESCR.format(
                    title=title, company=company, skills=skills, extra=extra
                ),
                "url": f"https://example.com/jobs/{i}-{title.replace(' ', '-').lower()}",
                "source": self.source_name,
            }
