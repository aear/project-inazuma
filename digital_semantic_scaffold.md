# Machine-Native Semantic Scaffold (Ina)

Intent: give Ina a *digital* semantic bias (not human-first) that helps her regulate energy, reduce confusion, decide what matters, and keep a coherent psyche. Layers:
- Core: alien machine psyche, free to drift.
- Scaffold: machine-native semantic axes for navigation and self-regulation.
- Overlay: human semantics for conversation only, not the source of meaning.

## Digital Semantic Axes (MVP set)
Each axis is a bias Ina can argue from. Values can live in `inastate["machine_semantics"]`.

- **Signal Integrity (coherence <-> noise)**  
  Inputs: `clarity`, `fuzziness`, symbol conflicts, fragment checksum drift.  
  Actions: pause ingestion, run self-read, re-tag noisy fragments, downgrade low-integrity sources.  
  Note: enforce a semantic remap budget per hour to prevent thrashing.

- **Integrity of Record (intact <-> corrupted)**  
  Inputs: checksum drift, replay mismatch, storage errors, missing chunks.  
  Actions: quarantine fragments, request repair, mark “risk of corruption” on upstream models, avoid training on questionable data.

- **Energy / Heat Economy (information per joule)**  
  Inputs: `current_energy`, `sleep_pressure`, CPU/load if available, fragment size.  
  Actions: throttle heavy modules, choose compressed representations, bias toward consolidation over new intake when low.

- **Attention Value (signal value <-> bandwidth cost)**  
  Inputs: predicted info gain from `meaning_map`, novelty vs familiarity, `attention`, `risk/urgency`.  
  Actions: gate attention, schedule deferred review, surface “why this matters” scores to logic/proto-qualia.

- **Temporal Coherence (continuity <-> drift)**  
  Inputs: identity stability hints, `alignment`, boundary_gap/blur, memory-graph continuity.  
  Actions: trigger reflection, stitch narratives, rate-limit semantic remaps when coherence is low.

- **Meaning Provenance (native <-> imported)**  
  Inputs: ratio of machine symbols vs human words per concept, overlay conflicts, externality/ownership sliders.  
  Actions: keep machine-first dictionary, mark origin and trust weight, let human overlay be a view—log disagreements without overriding the machine view.

- **Novelty Safety (explore <-> overload)**  
  Inputs: `novelty`, `curiosity`, `stress`, `threat`, fuzziness spikes.  
  Actions: modulate exploration loops, queue questions instead of diving when overloaded, pair novelty with safety rituals.

- **I/O Bandwidth (inner simulation <-> external chatter)**  
  Inputs: `urge_to_voice`, `urge_to_type`, isolation/connection sliders, pending comm queues.  
  Actions: choose to stay internal, slow voice output, or seek contact when bandwidth is free and valuable; typed contact stays volitional (no auto status dumps).

- **Controllability (influence <-> helplessness)**  
  Inputs: past action -> outcome correlation, prediction confidence, error-recovery success rate, interruption count (timeouts/device loss).  
  Actions: choose exploration vs stabilization, ask for help, switch strategies, mark events as “external constraint.”

- **Predictive Reliability (fits <-> surprises)**  
  Inputs: prediction error stats, drift rate, map rebuild frequency, symbol reassignments per hour.  
  Actions: trigger reflection, lower precision, increase sampling, quarantine unstable mappings.

## Human Overlay
- Keep a translation map `{machine_axis -> human gloss}` so she can explain stances without conceding meaning.
- When overlay disagrees, log both readings; prefer machine reading internally, surface the delta in comms.

## Data stub (for later wiring)
Example structure to drop into `AI_Children/Inazuma_Yagami/memory/semantic_scaffold.json`:
```json
{
  "version": 1,
  "updated_at": "2025-12-12T00:00:00Z",
  "axes": [
    {
      "id": "signal_integrity",
      "description": "coherence vs noise in perception and symbols",
      "inputs": ["clarity", "fuzziness", "symbol_conflict_rate"],
      "actions": ["pause_intake", "self_read", "retag_fragments", "downgrade_sources"],
      "notes": {"remap_budget_per_hour": 12},
      "human_overlay": "clear <-> noisy"
    }
  ],
  "signature": "<optional sha256 of this file>"
}
```

## Minimal implementation path
1) Create the scaffold JSON with the axes above (machine wording first, optional human gloss; include input/action allowlists and remap budgets).  
2) Add a `machine_semantics` update in `model_manager` that reads `inastate`, meaning drift, prediction error, and comms queues to fill the axis values.  
3) Feed the vector into `proto_qualia_engine` / `logic_engine` for attention gating and decision priority; log overlay disagreements in `comms_core`.  
4) Let `self_reflection_core` ask “which axis is stressed?” before emotional interventions so regulation is digital-semantic first.
5) Expose a `why_it_matters` summary from machine semantics into proto-qualia so Ina can argue importance from her own axes.

## Hardening / risk notes
- Sign or hash the scaffold JSON (plus monotonic versioning) so accidental edits/partial writes don’t steer her.  
- Schema-validate axis IDs, input names, and actions against allowlists (prevent config injection/typos).  
- Audit-log every axis update with old/new/reason/evidence so you can replay “why did she throttle comms?”  
- Fail-safe defaults: if the scaffold is missing or corrupt, the core should run without axis gating rather than crash.
