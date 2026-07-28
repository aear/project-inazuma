# Refactoring and maintainability audit

## Implemented in this pass

### Shared vector math

Eight modules carried behavior-equivalent cosine implementations. They now keep
their existing public functions for compatibility but delegate to
`vector_math.cosine_similarity`. The length-aligned variant used by the meaning
map already matched `symbol_word_utils.cosine_similarity`, so it delegates
there instead. The intentionally different intuition-engine variant remains
local.

This gives the numeric behavior one maintenance point without forcing broad API
changes or introducing NumPy as a dependency.

### Cached logic-graph norms

`logic_map_builder.build_logic_neural_map` previously recalculated both vector
norms for every neuron pair. Norms are now calculated once per neuron. A
500-vector, 64-dimension equivalent scan improved from 0.891 seconds to 0.379
seconds (2.35x) with identical scores.

### Earlier graph refactors retained

The same audit session also left these related improvements in place:

- cached norms in memory and meaning clustering;
- inverted tag/entity indexes for sparse experience graphs, with a dense-case
  safeguard;
- a bounded optional C++ cosine scan with an exact Python fallback; and
- isolated dependency-free build, benchmark, and equivalence-test modules.

## Recommended next refactors

### 1. Split orchestration from policy in the largest functions

The highest maintenance risk is not line count alone, but mixed responsibilities:

- `model_manager._update_meta_arbitration_signal` (about 690 lines)
- `house_viewer._build_furniture_catalog` (about 640 lines)
- `early_comm.early_communicate` (about 590 lines)
- `memory_graph.build_fractal_memory` (about 430 lines)
- `meaning_map.cluster_symbols_and_generate_words` (about 270 lines)

Extract pure decision helpers first, leaving I/O, logging, and state mutation in
thin orchestration functions. Each extraction should be behavior-tested; a
mechanical split would only move complexity between files.

### 2. Consolidate configuration access gradually

There are more than a dozen `load_config` implementations and several `_load_config`
variants. `storage_layout.load_config` and `model_manager.load_config` already
act as de facto shared entry points. Introduce one read-only configuration API
with explicit cache and error behavior, migrate leaf modules first, and avoid a
single large migration that could create import cycles.

### 3. Finish persistence-helper convergence

The repository still contains many local JSON load/write helpers despite
`io_utils.atomic_write_json` and `io_utils.load_json_dict`. Migrate state files
that require durability first. Keep streaming JSONL and very large memory files
on their existing specialized paths.

### 4. Add explicit subsystem boundaries

`model_manager`, GUI modules, and memory code import one another extensively.
Small protocol modules for status logging, runtime-state access, and scheduler
requests would reduce import-time coupling and make headless testing easier.
Avoid a generic service-locator object; narrow typed functions are more
inspectable.

### 5. Keep optimization boundaries coarse

The native vector kernel is useful because one call processes a whole bounded
batch. Do not move tag policy, persistence, logging, or cadence decisions into
C++. Those remain clearer and safer in Python.

## Validation notes

The shared vector formula was checked against the prior implementation for
empty, unequal-length, and normal vectors. The new unit tests are dependency
free. Some broad imports can still block on the external project drive, and the
full test suite still requires unavailable pytest/NumPy dependencies; focused
regression and compile checks are therefore the reliable validation path in the
current environment.
