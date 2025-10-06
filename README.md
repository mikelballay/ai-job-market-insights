# 🧠 AI Job Market Insights  
**End-to-end data pipeline to analyze and predict AI-related job roles and required skills**

![dashboard](docs/dashboard.png)

## 📘 Overview
**AI Job Market Insights** is a complete data science workflow that collects **real job offers** from online sources, cleans and processes them, extracts relevant **skills**, trains a **machine learning model** to predict job roles, and visualizes insights through an **interactive Streamlit dashboard**.

This project replicates a **real-world ML lifecycle** — from web scraping to model deployment — focusing on the **AI job market**.

---

## ⚙️ Features
✅ Scrapes job offers from **Remotive** and **RemoteOK**  
✅ Cleans, deduplicates, and merges raw `.jsonl` files  
✅ Extracts **skills and keywords** using NLP  
✅ Trains a **job role classifier** (XGBoost + Scikit-learn)  
✅ Generates reports and **confusion matrices**  
✅ Provides an **interactive dashboard** built with Streamlit  

---

## 📂 Project Structure
ai-job-market-insights/
│
├── data/ # Raw and processed datasets
├── models/ # Trained models (.pkl)
├── reports/ # Evaluation reports
├── src/ # Source code
│ ├── scraping/ # Web scraping scripts
│ ├── preprocessing/ # Cleaning, merging, feature extraction
│ ├── analysis/ # Training & evaluation
│ └── dashboard/ # Streamlit app
└── requirements.txt


---

## 🚀 Quickstart

### 1️⃣ Setup environment

python -m venv .venv
.venv\Scripts\activate        # (Windows)
# source .venv/bin/activate   # (Mac/Linux)
pip install -r requirements.txt
2️⃣ Collect real job data

Copiar código
python -m src.scraping.collect_jobs --query "machine learning" --out data/raw/remotive_ml.jsonl --source remotive --limit 100
python -m src.scraping.collect_jobs --query "data scientist" --out data/raw/remotive_ds.jsonl --source remotive --limit 100
python -m src.scraping.collect_jobs --query "mlops" --out data/raw/remotive_mlops.jsonl --source remotive --limit 100
python -m src.scraping.collect_jobs --query "quant" --out data/raw/remotive_quant.jsonl --source remotive --limit 100
3️⃣ Merge and build dataset

Copiar código
python -m src.preprocessing.merge_jsonl --in data/raw/remotive_*.jsonl --out data/raw/remotive_merged.jsonl
python -m src.preprocessing.build_dataset --in data/raw/remotive_merged.jsonl --out data/processed/jobs_remotive.parquet
python -m src.preprocessing.extract_skills --in data/processed/jobs_remotive.parquet --out data/processed/jobs_remotive_features.parquet
4️⃣ Train and evaluate model

python -m src.analysis.train_eval --in data/processed/jobs_remotive_features.parquet --out-model models/role_clf.pkl --out-report-train reports/train.txt --out-report-test reports/test.txt --out-cm-test reports/cm.png --clf auto --test-size 0.2 --cv 5
5️⃣ Launch interactive dashboard

streamlit run src/dashboard/app.py
📊 Example Results
Metric	Train F1-macro	Test F1-macro
Score	1.000	0.914

The model achieved strong generalization on real job data collected from Remotive, demonstrating the viability of skill-based classification for AI-related roles.

🧰 Tech Stack
Python 3.12

pandas, numpy, scikit-learn, xgboost

Streamlit (dashboard)

Requests, BeautifulSoup (scraping)

Joblib (model persistence)

👨‍💻 Author
Mikel [@mikelballay]
🎓 Data Science & Machine Learning Student — Universidad Carlos III de Madrid & University of Florida
📫 LinkedIn · GitHub

🏁 Next Steps
Add salary prediction models (regression task).

Integrate semantic embeddings (Sentence-BERT) to enhance text understanding.

Deploy dashboard online via Streamlit Cloud or Hugging Face Spaces.
