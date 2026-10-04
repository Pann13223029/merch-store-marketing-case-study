"""On-disk cache for the slow steps of notebook 04 (model refits, cross-validation, bootstraps, permutation importance).

cached() keys each result by everything it depends on: the step's name, its inputs (data frames, arrays, functions,
plain values), the step's own code, an optional version string (e.g. file_digest() of the module the step calls), and
the versions of Python, NumPy, pandas and scikit-learn. Any change to one of them misses the cache and recomputes, so
nothing needs clearing by hand; old entries just stay behind until data/processed/cache/ (gitignored) is deleted.

Values the step's code reads from outside (closure and notebook variables) are not seen by the key: pass each one as an
input or in `params`. Results are pickled, so only load a cache this machine wrote.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import platform
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "processed" / "cache"
LIBRARIES = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
             "scikit-learn": sklearn.__version__}

T = TypeVar("T")


def _plain(obj: Any) -> Any:
    """JSON fallback for the plain values json.dumps() can't write itself."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"cached() can't fingerprint a {type(obj).__name__}; pass its data or a plain value instead")


def _feed_code(h: hashlib._Hash, code: types.CodeType) -> None:
    """Add a function's bytecode, the names it uses, and its constants (nested functions included) to the hash."""
    h.update(code.co_code)
    h.update(repr(code.co_names).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _feed_code(h, const)
        elif isinstance(const, frozenset):   # set order depends on the string hash seed, which changes per process
            h.update(repr(sorted(map(repr, const))).encode())
        else:
            h.update(repr(const).encode())


def _feed(h: hashlib._Hash, obj: Any) -> None:
    """Add obj to the hash, tagged by type so that, e.g., [1, 2], (1, 2) and an array of 1 and 2 stay distinct."""
    if isinstance(obj, pd.DataFrame):
        h.update(b"frame")
        _feed(h, [[str(c), str(t)] for c, t in obj.dtypes.items()])
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
    elif isinstance(obj, pd.Series):
        h.update(b"series")
        _feed(h, [str(obj.name), str(obj.dtype)])
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(f"array {obj.dtype} {obj.shape}".encode())
        if obj.dtype == object:   # the raw bytes of an object array are pointers, so hash the values instead
            h.update(pd.util.hash_pandas_object(pd.Series(obj.ravel()), index=False).to_numpy().tobytes())
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, types.FunctionType):
        obj = inspect.unwrap(obj)   # decorated functions (e.g. scikit-learn's metrics) share their wrapper's code
        h.update(f"function {obj.__module__}.{obj.__qualname__}".encode())
        _feed_code(h, obj.__code__)
        _feed(h, [obj.__defaults__, obj.__kwdefaults__])
    elif isinstance(obj, Mapping):
        h.update(f"mapping {len(obj)}".encode())
        for key, value in obj.items():
            _feed(h, key)
            _feed(h, value)
    elif isinstance(obj, (list, tuple)):
        h.update(f"{type(obj).__name__} {len(obj)}".encode())
        for value in obj:
            _feed(h, value)
    else:
        h.update(b"value " + json.dumps(obj, default=_plain).encode())


def fingerprint(*parts: Any) -> str:
    """sha256 (hex) of the parts, in order."""
    h = hashlib.sha256()
    for part in parts:
        _feed(h, part)
    return h.hexdigest()


def file_digest(*paths: str | Path) -> str:
    """Short sha256 of the files' contents, for use as cached()'s `version` (e.g. the module a step calls)."""
    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def cached(name: str, compute: Callable[[], T], *inputs: Any, params: Any = None, version: str = "",
           refresh: bool = False, cache_dir: str | Path = CACHE_DIR) -> T:
    """compute()'s result, read from <cache_dir>/<name>-<key>.pkl when that file exists and `refresh` is False.

    Otherwise runs compute(), saves the result there, and returns it. The key covers `name`, `version`, the library
    versions, `params` (plain values), compute's code, and `inputs` (data frames, series, arrays, functions, or plain
    values, also inside lists, tuples and dicts). compute takes no arguments, so it is usually a lambda over the
    inputs.
    """
    key = fingerprint(name, version, LIBRARIES, params, compute, inputs)
    path = Path(cache_dir) / f"{name}-{key[:12]}.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as f:
            return pickle.load(f)
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    # write in full before taking the real name, so a run stopped mid-write leaves no broken entry
    partial = path.with_suffix(".partial")
    with partial.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(partial, path)
    return result
