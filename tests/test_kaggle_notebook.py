"""The committed Kaggle notebook must match what kaggle/build_notebook.py builds from the current repository.

Its helper cells are the src/ modules copied in unchanged, so a change to src/, to the session SQL or to the
builder's text without a rebuild would leave the published copy out of date. Cell types and sources are compared;
outputs, execution counts and metadata are not (the committed copy is the executed one).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "kaggle" / "build_notebook.py"
NOTEBOOK = ROOT / "kaggle" / "merch_store_marketing_case_study.ipynb"
TUNING = ROOT / "data" / "processed" / "tuning_results.json"
STALE = "Kaggle notebook is stale: run python kaggle/build_notebook.py"


def load_builder():
    """Import kaggle/build_notebook.py without running its __main__ block (importing writes nothing)."""
    spec = importlib.util.spec_from_file_location("build_notebook", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_kaggle_notebook_matches_a_fresh_build():
    fresh = load_builder().build_cells()
    committed = nbformat.read(NOTEBOOK, as_version=4).cells
    assert len(committed) == len(fresh), f"{STALE} ({len(committed)} cells committed, {len(fresh)} built)"
    for i, (old, new) in enumerate(zip(committed, fresh)):
        assert old.cell_type == new.cell_type, f"{STALE} (cell {i}: {old.cell_type} committed, {new.cell_type} built)"
        assert old.source == new.source, f"{STALE} (cell {i} differs; first line: {new.source.splitlines()[0][:80]!r})"


def test_builder_uses_the_tuned_boosting_settings():
    # the top gradient-boosting configuration by validation PR-AUC in the committed tuning results
    tuning = pd.read_json(TUNING)
    best = tuning[tuning.model == "gradient_boosting"].sort_values("cv_pr_auc", ascending=False).iloc[0]
    assert load_builder().BOOSTING_PARAMS == best.params, \
        "the Kaggle model settings differ from the tuning winner: update BOOSTING_PARAMS in kaggle/build_notebook.py"


def test_importing_the_builder_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = NOTEBOOK.stat().st_mtime_ns
    load_builder()
    assert NOTEBOOK.stat().st_mtime_ns == before
    assert list(tmp_path.iterdir()) == []
