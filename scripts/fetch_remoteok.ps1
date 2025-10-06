# scripts/fetch_remoteok.ps1
# Recolecta ofertas de RemoteOK para varias queries, une JSONL, procesa parquet, extrae skills y entrena.

# --- Config ---
$py = "python"  # si tu venv está activo, vale "python"
$limit = 50
$rawDir = "data/raw"
$mergedJsonl = Join-Path $rawDir "remoteok_merged.jsonl"
$onlyDir = Join-Path $rawDir "remoteok_only"
$outParquet = "data/processed/jobs_remoteok.parquet"
$outFeat = "data/processed/jobs_remoteok_features.parquet"

$reportTrain = "reports/role_clf_report_train_real.txt"
$reportTest  = "reports/role_clf_report_test_real.txt"
$outCM       = "reports/role_clf_cm_test_real.png"
$outModel    = "models/role_clf.pkl"

# --- Queries recomendadas ---
$queries = @(
  # Data
  "data scientist", "data engineer", "data analyst", "business intelligence", "big data",
  # ML/AI
  "machine learning", "ml engineer", "ai engineer", "deep learning", "nlp", "computer vision",
  # MLOps / Infra
  "mlops", "mlops engineer", "devops ai", "model deployment", "kubernetes", "docker",
  # Quant / Risk
  "quant", "quant researcher", "quant analyst", "risk analyst", "risk data", "financial engineer",
  # Otros
  "analytics engineer", "research scientist", "junior data", "data internship"
)

# --- Asegurar carpetas ---
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $outParquet) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $outFeat) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $reportTrain) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $outModel) | Out-Null

# --- Scraping por query ---
foreach ($q in $queries) {
  $safe = ($q -replace '[^\w\-]+','_').ToLower()
  $out = Join-Path $rawDir ("remoteok_{0}.jsonl" -f $safe)
  Write-Host ">> scraping: '$q' -> $out"
  & $py -m src.scraping.collect_jobs --query "$q" --out $out --source remoteok --limit $limit
}

# --- Merge JSONL (dedup por URL o (title,company,location,posted_date)) ---
Write-Host ">> merging jsonl -> $mergedJsonl"
& $py -m src.preprocessing.merge_jsonl --in "$rawDir/remoteok_*.jsonl" --out $mergedJsonl

# --- Mover a carpeta que espera build_dataset (directorio, no archivo) ---
New-Item -ItemType Directory -Force -Path $onlyDir | Out-Null
Move-Item -Force $mergedJsonl $onlyDir

# --- Build dataset -> parquet ---
Write-Host ">> building dataset -> $outParquet"
& $py -m src.preprocessing.build_dataset --in $onlyDir --out $outParquet

# --- Extraer skills ---
Write-Host ">> extracting skills -> $outFeat"
& $py -m src.preprocessing.extract_skills --in $outParquet --out $outFeat

# --- Entrenar / evaluar ---
Write-Host ">> training/evaluating model"
& $py -m src.analysis.train_eval --in $outFeat --out-model $outModel --out-report-train $reportTrain --out-report-test $reportTest --out-cm-test $outCM --clf auto --test-size 0.2 --cv 5

Write-Host "`n✅ Pipeline completo. Abre el dashboard y apunta al parquet real:"
Write-Host "   streamlit run src/dashboard/app.py"
