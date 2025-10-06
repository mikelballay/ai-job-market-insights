from __future__ import annotations

import pathlib
import subprocess
import sys


def test_mock_scrape_and_build(tmp_path: pathlib.Path):
    out_jsonl = tmp_path / "mock.jsonl"
    out_parquet = tmp_path / "jobs.parquet"

    cmd_scrape = [
        sys.executable,
        "-m",
        "src.scraping.collect_jobs",
        "--source",
        "mock",
        "--query",
        "data scientist",
        "--location",
        "Spain",
        "--limit",
        "20",
        "--out",
        str(out_jsonl),
    ]
    assert subprocess.run(cmd_scrape, check=True).returncode == 0
    assert out_jsonl.exists() and out_jsonl.stat().st_size > 0

    cmd_build = [
        sys.executable,
        "-m",
        "src.preprocessing.build_dataset",
        "--in",
        str(tmp_path),
        "--out",
        str(out_parquet),
    ]
    assert subprocess.run(cmd_build, check=True).returncode == 0
    assert out_parquet.exists() and out_parquet.stat().st_size > 0
