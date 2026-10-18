# Prediction and experience learning-loop investigation, 2026-10-18

Yes: a measurable, explicitly bounded learning trial can reuse the existing Experience Engine, domain-owned records and Hindsight evaluator. No new runtime subsystem is required. The trial is implemented as a benchmark, not installed as Ina's live learner. The immediate Hindsight correction below is implemented in production source; no active process was restarted or live memory rewritten by development tools.

## Findings in actual code

- `predictive_layer.py:run_prediction()` averages encoded recent fragments and derives clarity/confidence from vector spread. This is a current-state estimate or association candidate; it does not define a future target, horizon or independently observed outcome. Those confidence values are not measured forecast probabilities.
- The predictor retains the latest 100 snapshots by reading and rewriting `prediction_log.json`. It also publishes the latest `current_prediction`. The log has no stable prediction-ID/target/horizon contract, immutable forecast journal or explicit outcome join.
- `transformers/hindsight_transformer.py:evaluate_claims()` previously fell back from explicit observations to the next `predicted_vector`. `compute_hindsight()` could then turn one forecast's disagreement with another into error, reflections and fragment annotations. That does not establish an error against the world.
- `experience_engine.py` already provides one attempt per cycle, observation references, bounded evaluation metadata, immutable attempt/decision files, parent-linked keep/revise/revisit/stop choices and finite autonomous budgets. `experience_cycle_index.py` already provides bounded navigation; `experience_cycle_storage.py` already supplies quota-governed hot storage and verified durable draining. A second cycle engine or raw-memory replay would duplicate this machinery.

## Hindsight correction implemented

A subsequent prediction is no longer used as an observed outcome. Hindsight records a bounded reference to that subsequent forecast while preserving the original predicted values. Existing runtime prediction publication continues to make the newest forecast available; this patch does not overwrite an older forecast with the latest value.

Without an explicit observed outcome, a record stays `awaiting_observation`. Hindsight returns without updating trust or extracting lessons. Subsequent-forecast links are retained in `prediction_updates.json`, a rebuildable snapshot of at most 100 links. On this no-outcome path it does not rewrite the predictor-owned log, avoiding an additional concurrent read/modify/write race. No existing file needs migration. The normal observed-outcome branch and its legacy persistence remain unchanged.

Links say `target_alignment: unverified`: the current producer does not supply enough information to prove that successive predictions concern the same future event. The link records an update candidate, not confirmation that different horizons or targets can safely be merged. Explicit observed/actual fields remain accepted, including legacy next-record observed fields. Their event linkage, source authenticity and timing still need a stronger contract before live learning.

This patch does not retract old lessons or inspect old fragment annotations. Any review or repair of historical memory remains Ina-chosen.

## Feasibility trial and measurement

`benchmarks/benchmark_prediction_experience_loop.py` uses pinned source at `5576e4ed3ff7374c5f7edf25f95ffb96ef996de0`. The Experience Engine is materialized from Git. Only the required Hindsight methods are extracted from trusted source, so the benchmark does not import the live model manager. Boundary/lifecycle checks intercept side effects; direct snapshot persistence is checked separately in temporary storage.

A synthetic scalar task compares V1 fixed forecasts against V2 a small residual-mean correction fitted only on the training prefix. It freezes that correction during held-out and shifted outcomes. Every forecast is appended and flushed before the trial generates its outcome. Domain-owned event references and forecast hashes are retained in the existing cycle's immutable attempt/evaluation. Revision links use the existing finite continuation budget and the final choice is stop. This is an integration/control fixture, not evidence that Ina's existing transformer has learned this task.

| Signal | V1 fixed fixture | V2 residual fixture |
| --- | --- | --- |
| Stable held-out mean absolute error | 0.4 fixture units | 0.0 fixture units |
| Shifted-pattern mean absolute error | 0.2 | 0.6 |
| Twelve-step cycle chain | retained, stops | retained, stops |
| Held-out examples used to fit correction | no | no |

The improvement under one stable pattern and regression under drift are both retained. They demonstrate that the existing records can measure a proposed update and its failure. They do not establish real generalization, source independence or readiness for adoption.

Storage trials used explicit disposable directories on the configured HDD and NVMe, outside live memory. They included event flushes and final file flushes, plus Python allocation tracking. Numerical pools were capped for the trial process only.

- NVMe, 12 steps, three samples per variant: median elapsed about 0.252 s (V1), 0.256 s (V2). First-sample file totals: 50,193 / 50,441 bytes across 38 files. Python allocation peaks were around 1.6–1.8 MB, not process RSS.
- A reduced three-step NVMe trial also completed (one sample per variant); retained separately, not pooled with the longer trial.
- HDD: the three-sample twelve-step invocation hit its 55-second bound. A one-sample three-step retry also hit that bound. Neither produced a complete measurement. Do not infer an HDD latency, storage figure, speed ratio, device diagnosis or intrinsic hardware ranking from these timeouts. Ina was active and contention was not measured.

No storage schema upgrade is needed for this first trial. Production event retention and the legacy log's writer coordination need work before claiming a live durable learning loop. The HDD limitation blocks a completed cross-device storage comparison, not the observed semantic correction or NVMe fixture result.

## Smallest practical next integration

1. **Define a falsifiable forecast contract in the existing prediction producer.** Stable prediction ID, target ID, horizon/due boundary, unit, issued sequence/time, context and source references; uncertainty or abstention must be explicit. Retain a revision's parent ID and original forecast. Do not relabel vector spread as calibrated probability.
2. **Join actual observations explicitly.** The owning domain supplies observation ID, matching target/prediction ID, timestamp, unit and provenance. Missing, late, revised and disputed outcomes stay visible. A new forecast never closes an old forecast's outcome slot.
3. **Reuse the cycle and domain storage.** Append forecast/observation versions in a bounded, domain-owned journal; link them through existing cycle attempts, evaluations and parent choices. Use the existing compact cycle index for navigation. Keep the old log as a compatibility projection until its callers are audited. Add an Ina-chosen migration/rollback tool only if old records actually need conversion; no legacy snapshot should be invented into a genuine past observation.
4. **Make learning proposals measurable through Hindsight.** Preserve pre-update and challenger parameters, training/outcome boundaries and evaluation protocol. Compare later withheld outcomes, drift, error, abstention/coverage and repeated evidence groups separately. Keep/revise/revisit/stop remains a chosen, finite cycle. No automatic model replacement, trading authority or memory rewriting follows from a passed benchmark.
5. **Before live learning, address idempotency and retention.** Existing Hindsight can reprocess the same observations and annotate fragments repeatedly; the observed branch can rewrite a log also owned by the predictor. Track consumed prediction/outcome versions and coordinate writers or move the producer to append-only records. Retention must preserve original claims and contrary witnesses instead of dropping them merely because the rolling log exceeds 100 entries. `calibration_loss` is currently confidence-weighted error, not a proper probability calibration metric; use a defined probability task before adding Brier/log-loss and reliability bins.

## Checks and limits

Versioned boundary benchmark `prediction_feedback` compares the pinned fallback with the candidate: 2/7 versus 7/7 structural checks. The individual signals and source fingerprint are in `docs/evidence/prediction_feedback_20261018.json`.

Focused cycle/feedback pytest checks: 13 passed. Focused Hindsight checks: 6 passed (overlapping tests, not an independent combined count). Native runner: 5 passed. These verify ordinary forecast updates, explicit outcome scoring, bounded pending-link storage, original-log preservation, and existing cycle behavior.

Ina was found running. No adversarial/security tests or full suite were run; the full suite includes such cases and repository policy requires verified shutdown first. No GUI, real sensory outcome, live learning, production responsiveness or memory migration was tested. Timed-out HDD trials are disclosed rather than treated as passing or zero cost.

Reproduce ordinary trials in an isolated disposable checkout:

```console
python benchmarks/benchmark_prediction_experience_loop.py --root /chosen/disposable/parent --samples 1 --steps 3
python native_test_runner.py tests/test_prediction_experience_loop_trial.py
```

The new pending-link snapshot is derived and removable/rebuildable; rollback can restore Hindsight source without modifying original prediction records. The next live use depends on process reload; this work does not restart Ina.
