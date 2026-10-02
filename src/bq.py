"""BigQuery helpers: run SQL files with a cost guard and a local parquet cache.

The BigQuery sandbox allows 1 TB of query scanning per month. query() dry-runs
each query first so we see (and cap) the bytes it will scan before it runs (execute(), used for the
BigQuery ML scripts, skips the dry run; see its docstring), every job carries the same cap as
maximum_bytes_billed so BigQuery itself refuses to bill more, and query() results are cached to
data/raw/ so re-running a notebook costs nothing.
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
    """Accept either a file name in sql/ (e.g. 'audit/a03_lab_table_vs_public.sql') or raw SQL text.

    A name ending in '.sql' is always read as a file, so a missing or misspelled one raises
    FileNotFoundError instead of being sent to BigQuery as SQL text.
    """
    if name_or_query.endswith(".sql"):
        return (SQL_DIR / name_or_query).read_text()
    return name_or_query


def capped_job_config(max_gb: float) -> bigquery.QueryJobConfig:
    """Job config that makes BigQuery fail the job, at no charge, rather than bill more than max_gb."""
    return bigquery.QueryJobConfig(maximum_bytes_billed=int(max_gb * 1e9))


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

    sql = load_sql(name_or_query)
    bq = bq or client()
    gb = dry_run_gb(sql, bq)
    if gb > max_gb:
        raise RuntimeError(f"Query would scan {gb:.2f} GB (limit {max_gb} GB). Pass max_gb= to override.")
    print(f"Scanning {gb:.3f} GB...")

    # REST download: the gRPC Storage Read API gets dropped on this network, and our
    # result sets are small aggregates, so its speed-up isn't needed.
    job = bq.query(sql, job_config=capped_job_config(max_gb))
    df = to_portable_dtypes(job.to_dataframe(create_bqstorage_client=False))
    if cache_path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def execute(name_or_query: str, max_gb: float = DEFAULT_MAX_GB, bq: bigquery.Client | None = None) -> None:
    """Run SQL that returns no rows, such as the BigQuery ML CREATE MODEL scripts, under the same cap as query().

    These statements are not dry-run: a dry run doesn't reliably estimate what CREATE MODEL will bill
    (BigQuery ML works out the bytes for some model types only during training). The cap is applied
    as maximum_bytes_billed alone, so BigQuery fails the job, at no charge, if it would bill more than max_gb.
    """
    sql = load_sql(name_or_query)
    bq = bq or client()
    bq.query(sql, job_config=capped_job_config(max_gb)).result()
