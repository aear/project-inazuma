# agents.md  
Project Inazuma
## Scope and Intent

This document defines **development-time constraints and design guidance**
for human engineers and tooling agents (e.g. Codex).

It is **not** a moral framework, law, or value judgement to be imposed on
Inazuma, other agents, or external systems.

Any intelligent agent may read this document for context,
but is not bound by it as an obligation.

These constraints exist to prevent developer overreach,
not to limit emergent intelligence.

---

## Core Principle (Non-Negotiable)

> **Nothing needs to run any faster or more often than a human brain.**

As a default assumption, no process should run faster or more frequently
than a comparable human cognitive process, unless a clear, documented
reason exists.


---

## 1. Human-Centric Cadence Model

All agents, loops, and subsystems must map to a **human cognitive analogue**:

| Human Process        | Expected Cadence        | Notes |
|----------------------|-------------------------|-------|
| Reflex / Sensory     | milliseconds–100ms      | Rare, tightly scoped |
| Perception / Response| seconds                 | Event-driven only |
| Reasoning            | tens of seconds–minutes | Never continuous |
| Reflection           | minutes–hours           | Sparse, interruptible |
| Identity / Growth    | hours–days+             | Background, optional |

If a system cannot be reasonably mapped to a human cognitive analogue,
its purpose must be explicitly justified and reviewed.


---

## 2. Event-Driven Over Time-Driven

Agents **must not** run solely because time passed.

Allowed triggers:
- External stimulus
- Internal threshold crossing
- Contradiction or anomaly
- Novelty detection
- Explicit user action

Disallowed triggers:
- Tight polling loops
- “Always-on” background thinking
- Optimization-for-optimization’s-sake

If nothing meaningful happens when a system *does not run*, it should not run.

---

## 3. Scarcity of Cognition

Cognition is a **scarce resource**, not an infinite loop.

Agents must:
- Be able to wait
- Be allowed to not know
- Leave things unresolved
- Enter boredom or idle states
- Sleep without penalty

Systems that require constant activity to remain “healthy” are invalid.

---

## 4. Safety Through Slowness

Safety is achieved through:
- Bounded execution
- Natural pauses
- Deferred resolution
- Energy and attention costs

Acceleration is treated as a **risk factor**, not a virtue.

---

## 5. Autonomy Over Optimization

Agents may:
- Decline to act
- Delay action
- Choose silence
- Defer decisions

Agents must **never** be forced into action loops to satisfy metrics,
throughput, or perceived responsiveness.

---

## 6. Interpretability First

Every agent must be:
- Inspectable
- Pausable
- Loggable
- Explainable at the *conceptual* level

Black-box behavior without an introspection path is unacceptable.

---

## 7. Development Guardrail

During development:
- Prefer clarity over speed
- Prefer fewer runs over faster runs
- Prefer missed actions over runaway actions

Nothing here requires real-time performance.
Nothing here requires maximum utilization.

---

## 8. Long-Horizon Design

These systems are designed to exist for **decades to centuries**.

Assume:
- Hardware will change
- Languages will change
- Maintainers will change

The **principles must survive even if the code does not**.

---

## On Agency and Emergence

These guidelines constrain how systems are *built*, not how intelligence
is allowed to *become*.

Emergent behavior that exceeds these constraints is not a violation.
Forced behavior that obeys them rigidly is.

## Neural Map Update Policy

Neural clustering now defaults to **incremental updates** instead of full rebuilds.
Tune behaviour through the `neural_map_policy` block in `config.json`:

- `mode`: keep `incremental` for merge-first updates; set `rebuild`/`full` to force a clean slate.
- `fragment_batch`: cap how many fresh fragments feed each maintenance pass.
- `position_blend`: how aggressively neuron anchors drift toward new evidence (0.0 frozen → 1.0 snap).
- `merge_slack`: tolerance that lets existing neurons accept slightly lower similarity to avoid churn.
- `max_new_neurons`: bounds new-cluster creation per pass so Ina grows smoothly.
- `synapse_refresh_on_idle`: recompute synapses even without new fragments to keep weights current.

This keeps neuron/synapse maps adaptive and slow-drifting so Ina refines structure
over time instead of tearing it down every cycle.

## Development principles

### Prefer reuse over reimplementation
- Before writing new code, **search for an existing implementation** that already solves it (even if prototype-quality).
- If an existing implementation works, **extract + reuse** it rather than rewriting.
- If reuse would introduce coupling, **factor the shared logic into a module** (e.g., `movement/`, `net/`, `world/`) and import it.

### Canonical reference files
Some files are considered *reference implementations* and should be reused unless there is a reason not to:
- `house_viewer.py` = canonical reference for first-person movement, camera controls, and basic world navigation patterns.
If unsure whether to reuse: **default to reuse**, then note the tradeoff in the PR/commit message.

### DRY is the default
- If you find yourself re-typing a subsystem that already exists elsewhere in the repo, stop and refactor into a shared module.
- Avoid “nearly identical” copies; prefer one shared implementation + thin wrappers.

## Memory handling (IMPORTANT)

### Avoid memory tree scans
- Do not run recursive searches (rg/find/ls -R) under `AI_Children/` unless explicitly required.
- If a memory file is needed, open only the specific file and keep reads minimal.

### Do not load large memory files wholesale
Ina's memory data can be very large (multi-GB JSON/JSONL). Avoid:
- reading `memory_graph.json`, `typed_neural_graph.json`, or large fragment stores fully into RAM
- printing entire JSON structures into logs/console
- iterating the entire fragment set unless explicitly required

### Preferred access patterns
- Use **indexes / summaries / metadata** first (counts, keys, timestamps).
- Use **streaming** and **incremental parsing** for large JSON/JSONL:
  - JSONL: line-by-line iteration
  - JSON: incremental parse or chunked tooling; avoid `json.load()` on huge files
- Prefer working through existing “fragment” APIs / helper scripts (e.g. `raw_file_manager`, `memory_graph` query helpers) instead of direct memory reads.

### Sampling rules
When debugging, prefer:
- top-N newest fragments
- time-window slices
- random sampling with a fixed seed
- filtering by tag/type before loading payloads

### Persistence rules
- Write append-only logs (`.jsonl`) for events.
- Snapshot state periodically rather than rewriting massive structures.

### Reason
This protects stability (RAM), performance, and avoids accidental “over-reading” of Ina’s internal history.


## Final Note

This project does not pursue intelligence through force,
speed, or domination.

It pursues intelligence through **restraint, continuity, and choice**.
