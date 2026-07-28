"""Optional ctypes bridge to Project Inazuma's dependency-free C++ kernel."""
from __future__ import annotations

import ctypes
import math
import platform
from array import array
from pathlib import Path

_LIB = None
_TRIED = False


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    suffix = ".dll" if platform.system() == "Windows" else ".dylib" if platform.system() == "Darwin" else ".so"
    path = Path(__file__).resolve().parent / ".native" / f"libinazuma_vector{suffix}"
    if not path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(path))
        fn = lib.inazuma_cosine_pairs
        fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                       ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_size_t
        _LIB = lib
    except (AttributeError, OSError):
        _LIB = None
    return _LIB


def available():
    return _load() is not None


def cosine_pairs(vectors, threshold, *, pair_limit=None, per_source_limit=None,
                 max_output_pairs=5_000_000):
    """Return (pairs, evaluated, truncated), or None for unsuitable inputs."""
    lib = _load()
    rows = len(vectors)
    if lib is None or rows < 2 or rows > 0xFFFFFFFF:
        return None
    dimensions = len(vectors[0]) if vectors else 0
    if dimensions <= 0 or any(len(vector) != dimensions for vector in vectors):
        return None
    possible = rows * (rows - 1) // 2
    evaluated_cap = min(possible, pair_limit) if pair_limit else possible
    capacity = min(evaluated_cap, max_output_pairs)
    if capacity <= 0:
        return ([], 0, bool(pair_limit))
    flat = array("d")
    try:
        for vector in vectors:
            flat.extend(float(value) for value in vector)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in flat):
        return None
    left = (ctypes.c_uint32 * capacity)()
    right = (ctypes.c_uint32 * capacity)()
    scores = (ctypes.c_double * capacity)()
    evaluated = ctypes.c_size_t()
    truncated = ctypes.c_int()
    flat_buffer = (ctypes.c_double * len(flat)).from_buffer(flat)
    emitted = lib.inazuma_cosine_pairs(
        flat_buffer, rows, dimensions, float(threshold), max(0, int(pair_limit or 0)),
        max(0, int(per_source_limit or 0)), left, right, scores, capacity,
        ctypes.byref(evaluated), ctypes.byref(truncated))
    pairs = [(int(left[i]), int(right[i]), float(scores[i])) for i in range(emitted)]
    return pairs, int(evaluated.value), bool(truncated.value)
