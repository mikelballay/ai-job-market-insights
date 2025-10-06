
# Phase 2 — NLP & Baselines

## 1) Instalar nuevas dependencias
```powershell
pip install -r requirements.txt
# si quieres lematización:
# python -m spacy download en_core_web_sm
```

Añade al final de `requirements.txt`:
```
(scikit-learn y matplotlib vienen en patches/requirements.append.txt)
```

## 2) Generar features
```powershell
python -m src.preprocessing.clean_text --in data/processed/jobs_features.parquet --out data/processed/jobs_clean.parquet
python -m src.preprocessing.extract_skills --in data/processed/jobs_clean.parquet --out data/processed/jobs_features_skills.parquet
```

## 3) Entrenar baseline
```powershell
python -m src.analysis.train_role_classifier --in data/processed/jobs_clean.parquet --out-model models/role_clf.pkl --out-report reports/role_clf_report.txt --out-cm reports/role_clf_cm.png
```

## 4) Notebook de EDA
Abre `notebooks/02_eda_and_baselines.ipynb`.
