
.PHONY: install lint test scrape build app precommit

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

lint:
	ruff check .
	black --check .

test:
	pytest -q

precommit:
	ruff check --fix .
	black .

scrape:
	python -m src.scraping.collect_jobs --source mock --query "data scientist" --location "Spain" --limit 50 --out data/raw/mock_jobs.jsonl

build:
	python -m src.preprocessing.build_dataset --in data/raw --out data/processed/jobs_features.parquet

app:
	streamlit run src/dashboard/app.py

features:
	python -m src.preprocessing.clean_text --in data/processed/jobs_features.parquet --out data/processed/jobs_clean.parquet
	python -m src.preprocessing.extract_skills --in data/processed/jobs_clean.parquet --out data/processed/jobs_features_skills.parquet

train:
	python -m src.analysis.train_role_classifier --in data/processed/jobs_clean.parquet --out-model models/role_clf.pkl --out-report reports/role_clf_report.txt --out-cm reports/role_clf_cm.png
