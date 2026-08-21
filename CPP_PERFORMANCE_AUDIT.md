# C++ performance audit

## Scope

This audit covers the repository's top-level Python modules and transformer
package (about 75,000 lines), using source inspection, an AST loop-complexity
inventory, the existing mirror/database benchmark, and deterministic synthetic
benchmarks. It deliberately does not scan `AI_Children/` or load private memory
stores.

The main conclusion is: port a small shared vector-search kernel, not whole
modules. The dominant CPU opportunities are repeated cosine comparisons in
graph construction and clustering. Audio, image, GUI, database, and file
pipelines are mostly native-library-bound or I/O-bound already.

## Implemented optimization pass

The first optimization pass now includes:

- a dependency-free C++ cosine-pair kernel loaded through the Python standard
  library's `ctypes` module;
- `build_native.py`, which builds the optional library without pybind11, NumPy,
  or another binding package;
- automatic native dispatch for worthwhile, bounded synapse batches, with the
  original Python implementation retained as a fallback;
- cached vector norms in memory and meaning clustering; and
- inverted tag/entity indexes for experience-graph candidates, with a dense
  graph safeguard that falls back to a single exhaustive scan.

Updated 1,000-by-64-vector medians were 0.115-0.117 seconds for native synapse
construction, 1.70 seconds for memory clustering, and 1.45 seconds for meaning
clustering. Relative to the original measurements, these are approximately
31x, 2.2x, and 3x faster. Against the newly improved cached-norm Python
synapse fallback (1.50 seconds), the native end-to-end path remains 12.8x
faster, including conversion and Python result handling.

Build the optional local library with:

```text
python3 build_native.py
```

## Reproducible results

Commands:

```text
python3 -m benchmarks.benchmark_compute_hotspots --sizes 100,250,500,1000 --dimensions 64 --repeats 3
g++ -O3 -march=native -std=c++20 benchmark_cpp_vector_scan.cpp -o /tmp/ina-vector-bench
/tmp/ina-vector-bench 1000 64 5
python3 -m benchmarks.benchmark_memory_graph --root /tmp --records 2000 --payload-bytes 1024
```

Median wall time on the audit machine:

| Real project function | 100 vectors | 250 | 500 | 1,000 |
|---|---:|---:|---:|---:|
| `memory_graph.build_synaptic_links` | 0.0365 s | 0.2171 s | 0.8786 s | 3.5618 s |
| `memory_graph.cluster_fragments` | 0.0398 s | 0.2377 s | 0.9364 s | 3.7571 s |
| `meaning_map._cluster_encoded` | 0.0439 s | 0.2672 s | 1.0665 s | 4.3732 s |

All cases use normalized 64-element vectors and force the full worst-case
candidate scan. Increasing the item count by 10 increased time by roughly 100,
confirming quadratic behavior.

The standalone contiguous `float32` C++ cosine scan took 0.00942 seconds for
499,500 comparisons at 1,000 vectors and 0.23766 seconds for 12,497,500
comparisons at 5,000 vectors: about 53 million comparisons/second. Compared
with the 1,000-vector Python synapse case, the arithmetic-only lower bound is
about 378 times faster. A production extension will gain less because it must
validate inputs and construct result objects. Passing one contiguous batch per
call is therefore essential.

The existing SQLite mirror benchmark achieved 7,656 fresh records/second and
16,024 verified restart skips/second for 2,000 records with 1 KiB payloads.
That path is I/O/database dominated and is not a first C++ target.

## Ranked candidates

### 1. Shared vector neighbor-search kernel

Highest value. The same native boundary should serve:

- `memory_graph.build_synaptic_links`
- `memory_graph.cluster_fragments`
- `meaning_map._cluster_encoded` and word-vector matching
- `logic_map_builder.build_logic_neural_map`

Use contiguous `float32` vectors and integer row IDs. Expose coarse operations
such as `threshold_pairs`, `nearest_cluster`, and bounded top-k, returning
compact arrays. Keep policy decisions, tags, IDs, logging, persistence, and
cadence limits in Python for interpretability.

The bigger win is algorithmic: honor pair budgets early, use blockwise matrix
multiplication for moderate sets, and consider an approximate-neighbor index
only for large sets. C++ makes an exhaustive quadratic scan much faster but
does not make its growth sustainable.

### 2. Experience-graph edge construction

`memory_graph.build_experience_graph` compares every event pair and repeatedly
intersects tag/entity sets. Prefer an inverted index from tag/entity to event
IDs in Python first; that removes most unnecessary comparisons and is likely a
larger win than a direct C++ translation. Port only compact set-intersection
work if profiling still shows it hot after indexing.

### 3. Symbol-word matching and expression lookup

`meaning_map.cluster_symbols_and_generate_words` repeatedly scans word vectors.
`expression_log.find_best_symbol_word` also re-encodes component fragments
inside its word loop. Cache encodings and word centroids first, then route its
batched nearest-vector query through the shared native kernel. Do not port the
surrounding semantic and persistence logic.

### 4. Audio feature extraction (defer)

`audio_digest` and `speech_activity` look numeric, but FFT, dot products,
percentiles, and array operations already run in NumPy's compiled code. The
Python STFT frame loop may be removable with a strided NumPy view. A custom C++
port is justified only if end-to-end audio profiling shows array/list
conversion—especially `mel_db.tolist()`—is material.

### 5. Rendering, mesh generation, vision, and storage (do not port now)

- Mesh primitives are cached and handed to NumPy/pyqtgraph/OpenGL.
- Vision uses OpenCV and NumPy native kernels.
- SQLite mirroring, JSON persistence, archive handling, and raw-file ingestion
  are dominated by serialization and filesystem latency.
- GUI and scheduler code benefits more from reducing work per event and keeping
  blocking operations off the UI thread than from C++.

## Suggested implementation boundary

Create one optional extension, for example `inazuma_native`, with a pure-Python
fallback that has identical results. Build it with CMake and pybind11, but keep
the extension optional so the runtime remains portable.

The first production function should accept a 2-D contiguous `float32` buffer
plus threshold/top-k/pair-limit options and return row-index pairs, scores, and
statistics. It should release the GIL during calculation, check budgets between
blocks, and avoid Python callbacks inside the hot loop.

Before merging a native implementation:

1. Add equivalence tests for thresholds, zero vectors, mixed vector lengths,
   direction generation, caps, stable ordering, and truncation statistics.
2. Benchmark end-to-end calls including conversion and result construction.
3. Require a meaningful gain at realistic burst sizes; tiny batches may be
   faster in Python because boundary overhead dominates.
4. Preserve the existing Python fallback and the project's bounded,
   event-driven cadence.

## Environment note

The checked-in virtual environment currently lacks NumPy, pytest, and pybind11.
The standard-library test discovery ran 27 tests but could not complete the
suite because 11 test modules failed to import (mostly missing pytest/NumPy).
The new Python benchmark compiles and runs with the system Python because the
two measured modules have fallbacks for these dependencies.
