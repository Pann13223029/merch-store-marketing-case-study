"""Tests for src/cache.py: hits and misses, what the key covers, and refresh."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from src.cache import cached, file_digest


class Counter:
    """A compute() stand-in that counts its calls and returns the call number."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.calls


def frame() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "channel": ["Direct", "Social", "Referral"],
                         "day": pd.to_datetime(["2017-05-01", "2017-05-02", "2017-05-03"])})


def test_second_call_with_the_same_inputs_is_a_hit(tmp_path):
    count = Counter()
    first = cached("step", lambda: count(), frame(), np.arange(5), cache_dir=tmp_path)
    second = cached("step", lambda: count(), frame(), np.arange(5), cache_dir=tmp_path)
    assert (first, second, count.calls) == (1, 1, 1)
    assert [p.name.split("-")[0] for p in tmp_path.iterdir()] == ["step"]


def test_cached_result_round_trips(tmp_path):
    result = {"scores": np.linspace(0, 1, 7), "table": frame()}
    cached("step", lambda: result, cache_dir=tmp_path)
    again = cached("step", lambda: result, cache_dir=tmp_path)
    np.testing.assert_array_equal(again["scores"], result["scores"])
    pd.testing.assert_frame_equal(again["table"], result["table"])


@pytest.mark.parametrize("change", [
    lambda df: df.assign(x=[1, 2, 4]),                       # one value
    lambda df: df.rename(columns={"x": "y"}),                # a column name
    lambda df: df.astype({"x": "float64"}),                  # a dtype
    lambda df: df.set_axis([0, 1, 5]),                       # the index
    lambda df: df.iloc[:2],                                  # a row dropped
], ids=["value", "column", "dtype", "index", "rows"])
def test_a_changed_data_frame_misses(tmp_path, change):
    count = Counter()
    cached("step", lambda: count(), frame(), cache_dir=tmp_path)
    assert cached("step", lambda: count(), change(frame()), cache_dir=tmp_path) == 2


def test_a_changed_series_misses(tmp_path):
    count = Counter()
    s = frame().x
    cached("step", lambda: count(), s, cache_dir=tmp_path)
    assert cached("step", lambda: count(), s, cache_dir=tmp_path) == 1
    assert cached("step", lambda: count(), s.rename("y"), cache_dir=tmp_path) == 2
    assert cached("step", lambda: count(), s + 1, cache_dir=tmp_path) == 3


@pytest.mark.parametrize("changed", [
    np.array([0.0, 1.0, 2.5]),                  # one value
    np.array([0, 1, 2]),                        # the dtype
    np.array([[0.0, 1.0, 2.0]]),                # the shape
    [0.0, 1.0, 2.0],                            # a list instead of an array
], ids=["value", "dtype", "shape", "type"])
def test_a_changed_numpy_array_misses(tmp_path, changed):
    count = Counter()
    cached("step", lambda: count(), np.array([0.0, 1.0, 2.0]), cache_dir=tmp_path)
    assert cached("step", lambda: count(), np.array([0.0, 1.0, 2.0]), cache_dir=tmp_path) == 1
    assert cached("step", lambda: count(), changed, cache_dir=tmp_path) == 2


def test_object_arrays_are_keyed_by_value(tmp_path):
    count = Counter()
    cached("step", lambda: count(), np.array(["a", "b"], dtype=object), cache_dir=tmp_path)
    assert cached("step", lambda: count(), np.array(["a", "b"], dtype=object), cache_dir=tmp_path) == 1
    assert cached("step", lambda: count(), np.array(["a", "c"], dtype=object), cache_dir=tmp_path) == 2


def test_name_params_version_and_inputs_inside_dicts_are_part_of_the_key(tmp_path):
    count = Counter()
    base = dict(params={"n_boot": 2000, "seed": 42}, version="v1", cache_dir=tmp_path)
    scores = {"a": np.zeros(3), "b": np.ones(3)}
    cached("step", lambda: count(), scores, **base)
    assert cached("step", lambda: count(), scores, **base) == 1
    assert cached("other", lambda: count(), scores, **base) == 2
    assert cached("step", lambda: count(), scores, **{**base, "params": {"n_boot": 2000, "seed": 7}}) == 3
    assert cached("step", lambda: count(), scores, **{**base, "version": "v2"}) == 4
    assert cached("step", lambda: count(), {"a": np.zeros(3), "b": np.full(3, 2.0)}, **base) == 5


def test_the_steps_own_code_and_function_inputs_are_part_of_the_key(tmp_path):
    assert cached("step", lambda: 0.10, cache_dir=tmp_path) == 0.10
    assert cached("step", lambda: 0.20, cache_dir=tmp_path) == 0.20   # another constant in the step: a miss
    assert cached("step", lambda: 0.10, cache_dir=tmp_path) == 0.10   # the first entry again
    assert cached("step", lambda frac=0.30: frac, cache_dir=tmp_path) == 0.30   # default arguments count too

    count = Counter()
    top_10 = lambda y: y * 0.10   # noqa: E731
    top_20 = lambda y: y * 0.20   # noqa: E731
    assert cached("metric", lambda: count(), top_10, cache_dir=tmp_path) == 1
    assert cached("metric", lambda: count(), top_10, cache_dir=tmp_path) == 1
    assert cached("metric", lambda: count(), top_20, cache_dir=tmp_path) == 2


def test_decorated_functions_are_keyed_by_the_function_they_wrap(tmp_path):
    def checked(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @checked
    def precision(y, s):
        return 1

    @checked
    def recall(y, s):
        return 2

    count = Counter()
    assert cached("metric", lambda: count(), precision, cache_dir=tmp_path) == 1
    assert cached("metric", lambda: count(), recall, cache_dir=tmp_path) == 2


def test_refresh_recomputes_and_replaces_the_entry(tmp_path):
    count = Counter()
    assert cached("step", lambda: count(), np.arange(3), cache_dir=tmp_path) == 1
    assert cached("step", lambda: count(), np.arange(3), refresh=True, cache_dir=tmp_path) == 2
    assert cached("step", lambda: count(), np.arange(3), cache_dir=tmp_path) == 2
    assert len(list(tmp_path.iterdir())) == 1


def test_inputs_it_cannot_fingerprint_raise(tmp_path):
    with pytest.raises(TypeError, match="fingerprint"):
        cached("step", lambda: 1, object(), cache_dir=tmp_path)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_file_digest_follows_the_contents(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("x = 1\n")
    before = file_digest(path)
    path.write_text("x = 2\n")
    assert file_digest(path) != before and len(before) == 16
