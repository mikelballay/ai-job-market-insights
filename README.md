
# AI Job Market Insights — Fase 1 (Fundaciones + Scraping MVP)

Este starter te guía paso a paso para dejar **Fase 1** lista: estructura profesional, scraping MVP y dataset normalizado.

## 🚀 Arranque rápido

Mac/Linux:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\pre-commit.exe install
```

## 📦 Estructura
```
data/
  raw/         # JSONL crudo
  processed/   # Parquet normalizado
models/
reports/
src/
  common/
  preprocessing/
  scraping/
tests/
```

## 🔧 Scraping (MVP con datos sintéticos)
```bash
make scrape
```

## 🧹 Normalización → Parquet
```bash
make build
```

## ✅ Calidad
```bash
make precommit
make test
```

---

## 🛣️ Paso a paso (detallado)

### 1) Entorno e instalación

Mac/Linux:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

Windows:
```powershell
python -m venv .venv
.\n+.venv\Scripts\python.exe -m pip install --upgrade pip
.
.venv\Scripts\python.exe -m pip install -r requirements.txt
.
.venv\Scripts\pre-commit.exe install
```

### 2) Scraping de validación (mock)
```bash
python -m src.scraping.collect_jobs --source mock --query "data scientist" --location "Spain" --limit 50 --out data/raw/mock_jobs.jsonl
```

### 3) Normalizar y convertir a Parquet
```bash
python -m src.preprocessing.build_dataset --in data/raw --out data/processed/jobs_features.parquet
```

### 4) Lint + Tests
```bash
ruff check . && black --check .
pytest -q
```

### 5) Dashboard (Fase 3)

```bash
streamlit run src/dashboard/app.py
```

Notas:
- El modelo `models/role_clf.pkl` ha sido entrenado con `src/analysis/train_eval.py`. El dashboard incluye una compatibilidad para cargar funciones de ese script al deserializar el modelo.
- Si cambias el script de entrenamiento, asegúrate de mantener los nombres de funciones del pipeline o ajusta el dashboard en `src/dashboard/app.py`.

### 6) Próximos pasos
- Implementa un scraper real en `src/scraping/sources/` heredando de `BaseScraper`.
- Documenta ética/TOS en README.
- Objetivo: ≥ 1.500 ofertas crudas esta semana.



