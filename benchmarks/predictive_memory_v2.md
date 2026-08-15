# Predictive Layer Memory Benchmark V2

## Purpose

This benchmark compares the historical predictive symbol lookup with the
bounded-memory implementation. It exists to make the memory tradeoff explicit
and reproducible without running prediction continuously or touching Ina's
live prediction state.

## Implementations

### V1 — full store load

V1 deserializes the complete `symbol_words.json` file before selecting a
candidate. The current private store is approximately 1.52 GB on disk and was
observed at approximately 2.75 GB of process memory during prediction.

V1 also materializes and sorts every fragment path before selecting the ten
newest fragments.

### V2 — compact semantic index

V2 reuses the compact semantic index shared with the logic layer. The full
symbol store is streamed only when that index must be rebuilt. Recent fragment
selection uses a fixed-size top-N heap, so it retains only the requested paths.

Neural-map health checks also count top-level entries with a streaming reader
instead of deserializing the maps.

## Recorded Results

Measurements were taken on 15 August 2026 against the current private store.
Private paths and memory contents are intentionally excluded.

<!-- benchmark-results:start -->
| Measurement | Historical V1 | Measured V2 |
|---|---:|---:|
| Symbol store size | 1.52 GB | 1.52 GB source, 1.61 KB cached index |
| Observed prediction memory | approximately 2.75 GB | not applicable to isolated lookup |
| Isolated indexed lookup peak RSS | not rerun | 19.2 MB |
| Isolated indexed lookup elapsed time | not rerun | 0.11 ms |
| Indexed candidates | not recorded | 3 |
<!-- benchmark-results:end -->

The predictive module import comparison (83 MB for V1 and 38 MB for V2) and
newest-10 fragment selection measurement (154 MB and 1.52 seconds across
406,740 fragments) are separate diagnostic probes, so the automated lookup
table does not overwrite them.

V1 was not rerun against the private 1.52 GB store because its known behavior
requires a multi-gigabyte allocation. This is recorded as a historical
observation rather than presented as a fresh controlled measurement.

## Correctness and Capability Checks

- Streamed and loaded candidate scoring select the same winner in the bounded
  V1/V2 fixture.
- A fresh compact index is reused without rescanning the source store.
- Recent-fragment selection returns the same newest-first result.
- The benchmark and tests do not write to Ina's live prediction state.

## Reproduction

Run the bounded V2 lookup explicitly:

```bash
python benchmark_predictive_memory.py PATH_TO_SYMBOL_WORDS_JSON --version V2 \
  --update-report benchmarks/predictive_memory_v2.md
```

The script updates only the marked table and emits the same measurement as
compact JSON on standard output for local tooling. It never writes a JSON
result artifact or includes the private source path in the report.

V1 remains available for small synthetic fixtures:

```bash
python benchmark_predictive_memory.py PATH_TO_SYNTHETIC_SYMBOL_WORDS_JSON --version V1
```

Do not invoke V1 on a large live store merely to reproduce the historical
failure mode.
