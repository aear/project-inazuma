# agents.md  
Project Inazuma

At the start of work in this repository, read `agent_notes.txt` if it exists.
It contains local-only context and must never be committed or quoted into public artifacts.

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

### Reversibility and compatibility paths
- Prefer reversible changes.
- Document why strange code exists, especially compatibility paths and non-obvious fallbacks.
- Instrument a path and observe its real use before deleting it.
- Assume today's compatibility path becomes tomorrow's haunted basement; keep it inspectable, bounded, and removable.

### Ina's Rule 34: Benchmark everything

> **Rule 34:** If a function exists, it will have a benchmark.
>
> **Rule 34b:** If a benchmark does not exist, it will be created.

- Benchmarks may measure correctness, capability, quality, resource use, or performance as appropriate to the function.
- Benchmark module versions explicitly (`V1`, `V2`, `V3`, and so on) so changes can be compared instead of merely described.
- Materialize historical benchmark implementations from pinned Git revisions when possible; do not duplicate old source in the working tree merely to preserve a baseline.
- Language benchmarks must use adversarial minimal pairs and score composition, morphology, constructions, pragmatics, discourse, uncertainty, whole-utterance alternatives, and reading-span continuity separately.
- Prefer deterministic, reproducible cases with inspectable scoring and retained historical results.
- Keep benchmarks bounded and explicitly invoked. Rule 34 does not authorize tight loops, continuous evaluation, or cognition that runs merely because time passed.
- When touching an unbenchmarked function, add the smallest meaningful benchmark or record why measurement is currently blocked.
- Tests must remain runnable without third-party pytest through `python native_test_runner.py`; use only the supported compatibility surface or extend it with a benchmark when new pytest behavior is needed.
- A code change is not complete until each changed behaviour has an explicitly versioned benchmark entry (for example, retained `V1` and candidate `V2`) in the benchmark suite. Unit tests verify correctness but do not replace this comparison. If a meaningful comparison cannot yet run, record the blocker in the benchmark registry or change notes rather than silently omitting it.
- Benchmark UI and export/reporting changes as user-facing capabilities too; measurement is not limited to numerical kernels.

### Background interference benchmarks

- Changes to heavy or persistent background work must run a bounded idle-versus-loaded interference benchmark. Measure audio xrun/error rate, input latency, desktop frame latency, context switches/second, involuntary context switches, writeback pressure, and per-core saturation; aggregate CPU/GPU/RAM totals are not sufficient evidence of responsiveness.
- Measure concurrency explicitly: system and benchmark-task thread counts, task-tree peak, runnable versus sleeping/uninterruptible workers, threads per logical CPU, and numerical-library pool settings such as `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`.
- Thread pools must be deliberate and module-scoped. Do not create a full hardware-sized pool in every module by default; retain or cap fan-out according to measured benefit and human-visible interference, and prefer shared/bounded execution where capability is unchanged.
- Label real probes and proxies separately. Scheduler wake delay may be retained as an input-dispatch proxy, but must not be reported as a real input round trip; absent frame or audio instrumentation is `unavailable`, not zero.
- Run only explicitly invoked, time-bounded tasks in isolated temporary storage. Preserve bounded raw samples and compare with a historical implementation materialized from its pinned Git revision.

### Adaptive thread governance

- Thread-count learning is module-, workload-, and hardware-specific. Never treat aggregate idle CPU as evidence that another full hardware-sized pool is harmless.
- Start unmeasured background modules conservatively and interactive/audio-adjacent modules lower still. Choose the smallest measured count that preserves capability within the background-interference budget.
- Learning advances only from explicitly invoked, bounded benchmarks with a finite exploration budget. The governor must not continuously probe, poll, or tune itself while Ina is running.
- Preserve observations and the previous usable decision so a worse candidate can be rejected or rolled back. Separate allocated workers, runnable workers, and numerical-library pools in reports.
- Prefer control-engineering behaviour over unconstrained optimisation for hardware and resource control: bounded excursions, operating envelopes, feedback, settling evidence, deadband, and hysteresis.
- Opposing differential probes are logical and sequential: measure the current centre, then run at most one lower or higher challenger at a time. Never run competing challengers concurrently to measure contention.
- Treat audio xruns, input/frame latency ceilings, writeback pressure, and runnable contention as hard operating limits. Throughput or capability gains cannot compensate for crossing them.
- Changes inside the configured deadband are neutral: retain the accepted allocation. Require the additional hysteresis margin before reversing direction or accepting a positive-resource move.
- A workload or hardware identity change may open a fresh finite exploration budget; time passing alone must not reopen learning.

### Standalone Codex harness

- The lightweight Codex GUI harness is a development tool and must run as a separate process from Ina. It must not import Ina's runtime, memory stores, or process tree.
- Authentication is ChatGPT-subscription-only: force `forced_login_method = "chatgpt"`, reuse Codex's cached login or device/browser login, and reject API-key, access-token, custom-base-URL, or automatic fallback paths that could create usage-based charges.
- Use Codex app-server rather than reimplementing model/tool behavior. Keep command and file-change approvals user-routed; never add an automatic approve-for-me path.
- Bind the GUI to localhost with a per-launch access token. Keep transcript/events bounded and fetch optional capability catalogs on demand.
- Cached authentication is private local state: never copy it into this repository, logs, benchmark fixtures, or exported reports. VS Code does not need to be running for cached CLI/app-server authentication to work.
- Benchmark the historical VS Code-hosted workflow against the standalone harness for capability coverage, startup/runtime resources, latency, isolation, and authentication safety.

## Experiential action design

- Experience Engine owns the optional domain-neutral cycle: intent, one bounded attempt, observation, evaluation, then keep/revise/revisit/stop. Attempts are retained, and revision/revisit cycles link to their parent cycle.
- One cycle is the default. Autonomous continuation requires an explicit finite budget; absence or exhaustion of that budget means stop. Do not simulate experience with sleeps or force continued activity.
- Domain payloads stay with their owning subsystem and are referenced by stable ID/path. Hindsight owns later lesson extraction.
- Storage-affecting Experience Engine changes require a bounded, explicitly invoked historical comparison on both the configured durable HDD and fast NVMe when available. Record storage, latency, and memory overhead without using live experience stores.
- Treat NVMe as a quota-bounded hot experiential workspace, HDD as the durable long-term store, and compact indexed/condensed structures as the quick-reach layer. Hot-tier writes must enforce byte/file ceilings and a free-space reserve; maintenance must hash-verify a bounded copy to HDD before retiring hot records.
- Quick navigation must use the bounded cycle index or Hindsight-derived lessons rather than replaying raw history. The index locates experience; it does not extract lessons.
- Apply the cycle to DAW, drawing, and non-reflex motor exploration. Safety reflexes may bypass deliberation but should remain observable where practical.
- Virtual desktop file access is capability-scoped: media-source drives are read-only; Ina's private HDD drive is writable data storage. The explorer must never execute files or expose a generic process-launch surface.

## Adversarial testing isolation (NON-NEGOTIABLE)

- Ina must not be running during adversarial, exploit, prompt-injection, containment-escape, or other security testing.
- Before any such test begins, stop the complete Ina runtime tree, including detached helpers and bridges, then verify that no Ina process remains.
- Record the verification result in the private test log before executing the first adversarial input.
- If shutdown cannot be completed or independently verified, the adversarial test is blocked. Do not substitute an in-process pause or scheduler guard for full shutdown.
- Run adversarial fixtures in an isolated test environment that cannot read or write Ina's live memory stores.

## Memory handling (IMPORTANT)

### Avoid memory tree scans
- Do not run recursive searches (rg/find/ls -R) under `AI_Children/` unless explicitly required.
- If a memory file is needed, open only the specific file and keep reads minimal.
- Runtime cognition must select fragments through `memory_map.sqlite` with an
  explicit row limit. If the index is unavailable, defer the work; never fall
  back to a fragment-directory glob or recursive walk.
- Integrity checking is a resumable indexed pass. Feed invalid records into the
  bounded repair queue, retain the corrupt original, and restore only from a
  hash-verified mirror or an explicitly inspectable salvage result.
- Routine cognitive ticks must never glob the fragments directory. Use the compact
  SQLite index with an explicit batch and time budget, persist a resume cursor,
  and defer safely when the index is unavailable.

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


## Federated memory continuity

- Continuity coordinates memory stores as a **federation of witnesses**; it does not own or rewrite their modality traces.
- Continuity may retain bounded links, confidence, recency, causal associations, recall rankings, and descriptive diversity measurements.
- Original traces stay in the store where they were created. Coordination must not merge, edit, delete, or silently resolve conflicts between source witnesses.
- Recall arbitration uses compact indexes and the bounded continuity core rather than replaying raw history. A deliberative recall is represented as an Experience Cycle with its plan referenced outside Experience Engine.
- Bias reporting is read-only and descriptive. Concentration or selection skew may be surfaced for review, but must not trigger automatic memory rewriting or forced balancing.
- Benchmark witness preservation, cross-store recall, diversity reporting, selection skew, storage, latency, and memory against the pinned historical implementation.

## Final Note

This project does not pursue intelligence through force,
speed, or domination.

It pursues intelligence through **restraint, continuity, and choice**.
