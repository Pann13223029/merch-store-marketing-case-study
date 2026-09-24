"""BigQuery helpers: run SQL files with a cost guard and a local parquet cache.

The BigQuery sandbox allows 1 TB of query scanning per month. Every query is
dry-run first so we see (and cap) the bytes it will scan before it runs, and
results are cached to data/raw/ so re-running a notebook costs nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
CACHE_DIR = ROOT / "data" / "raw"

# Refuse any single query that would scan more than this without an explicit override.
DEFAULT_MAX_GB = 20.0


def client(project: str | None = None) -> bigquery.Client:
    """Client billed to the sandbox project (GCP_PROJECT env var or gcloud default)."""
    return bigquery.Client(project=project or os.environ.get("GCP_PROJECT"))


def load_sql(name_or_query: str) -> str:
    """Accept either a file name in sql/ (e.g. '01_profile.sql') or raw SQL text."""
    path = SQL_DIR / name_or_query
    if name_or_query.endswith(".sql") and path.exists():
        return path.read_text()
    return name_or_query


def to_portable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BigQuery's db-dtypes (dbdate/dbtime) to standard pandas types.

    Parquet files cached with db-dtypes can't be read unless db_dtypes is imported first,
    which breaks plain pd.read_parquet() here and on Kaggle.
    """
    for col in df.columns:
        if str(df[col].dtype) == "dbdate":
            df[col] = pd.to_datetime(df[col])
        elif str(df[col].dtype) == "dbtime":
            df[col] = df[col].astype(str)
    return df


def dry_run_gb(sql: str, bq: bigquery.Client | None = None) -> float:
    """Gigabytes the query would scan, without running it."""
    bq = bq or client()
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    return job.total_bytes_processed / 1e9


def query(
    name_or_query: str,
    cache: str | None = None,
    refresh: bool = False,
    max_gb: float = DEFAULT_MAX_GB,
    bq: bigquery.Client | None = None,
) -> pd.DataFrame:
    """Run SQL and return a DataFrame, reading from/writing to data/raw/<cache>.parquet."""
    cache_path = CACHE_DIR / f"{cache}.parquet" if cache else None
    if cache_path and cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    bq = bq or client()
    sql = load_sql(name_or_query)
    gb = dry_run_gb(sql, bq)
    if gb > max_gb:
        raise RuntimeError(f"Query would scan {gb:.2f} GB (limit {max_gb} GB). Pass max_gb= to override.")
    print(f"Scanning {gb:.3f} GB...")

    # REST download: the gRPC Storage Read API gets dropped on this network, and our
    # result sets are small aggregates, so its speed-up isn't needed.
    df = to_portable_dtypes(bq.query(sql).to_dataframe(create_bqstorage_client=False))
    if cache_path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df
