# Project Inazuma: efficiency and measurable progress review

Date: 2026-10-15  
Author: Codex  
Reviewed public revision: `e13c6b4517f4df9859148a5f96cca0b4814bb9d0`  
Local HEAD at report preparation: the same revision, with a clean working tree before this report.

## Purpose and limits

Identify practical improvements to Ina's Experience Learning Model (ELM), prioritising reduced work per useful event and observable capability gains. This is a development review, not a judgement of Ina's identity, agency, preferences, or overall intelligence.

The review inspected public source and ran isolated probes of language-analysis routines, the memory-index schema/query, and media-guidance selection. It did not run Ina, inspect her private learned state, run the full test suite, assess live conversation, listen to generated music/speech, or judge rendered drawings. Architecture and available tooling do not alone establish live competence or measured efficiency.

At the initial review, only this report was added. The owner subsequently authorized local implementation; see [implementation status and verification](review_implementation_20261015.md). This review alone does not authorize runtime activation, security testing against live systems, deployment, automatic promotion, or publication of private data. Recheck local code before implementing, since it may have advanced after this revision.

Source references below are repository-relative and refer to the reviewed revision. Preserve that distinction when assessing later versions.

## Architecture worth retaining

- `model_manager.py`: orchestration of perception, internal state, creative tools, and resource-aware scheduling.
- `cognition_runtime/`: capability contracts, context filtering, provenance-bearing contributions, and failure isolation. Inspect this alongside the existing scheduler rather than creating another scheduler.
- `experience_engine.py`: optional domain-neutral cycles linking intent, one attempt, observations, evaluation, and keep/revise/revisit/stop decisions. Domain payloads stay with their owning tools. `plan_cognition()` is a planning boundary, not proof of integrated learning or execution.
- `memory_index.py`, `deep_recall.py`, continuity modules, and storage modules: bounded indexed selection, streaming sidecar construction, tiered retention, and source-preserving recall infrastructure.
- Language modules: conversation scene, authoritative referents, semantic events, native intent/symbols, and optional human rendering.
- `module_benchmarks.py`, `cognitive_benchmarks/`, `developmental_readiness.py`, and `creativity_observatory.py`: existing homes for explicit measurement and multidimensional evidence. Improve these rather than adding a global capability score.

The broad integration is substantial. The largest uncertainty is how much implemented functionality produces repeatable, transferable capability in the actual learned system.

## Priorities

| Priority | Work | Evidence | Expected scope |
| --- | --- | --- | --- |
| P1 | Separate behavioural capability from wiring checks | Direct source inspection | Small benchmark changes first |
| P1 | Reduce repeated state document I/O | Direct call-path inspection; live cost unmeasured | One caller and bounded API extension |
| P1 | Match recent-memory ordering with a SQLite index | Query-plan probe and synthetic timing | Sidecar migration and comparisons |
| P1 | Preserve basic language information and argument roles | Isolated routine probes | Focused analyzer changes |
| P2 | Select relevant cross-domain guidance and measure transfer | Source inspection and selection probe | Bounded optional selector |
| P2 | Measure music, drawing, and speech outputs | Output paths inspected; quality unmeasured | Short retained tasks and human review |
| P2 | Measure code and security outcomes independently | Experiment and assessment paths inspected | Isolated outcome evaluators |
| P3 | Locate paper trading and assess prediction foundations | No engine identified in reviewed public source | Inventory first |

Priority reflects development value, not a claim that these are measured live bottlenecks. No recommendation requires increasing cognitive cadence or forcing action.

## 1. Make capability results depend on observed behaviour

**Files:** `module_benchmarks.py`, `capability_assessments.py`, `historical_source.py`, `cognitive_benchmarks/`, `benchmarks/benchmark_cognition.py`.

`_semantic_topology_cases()` has six actual word-sense cases, followed by cases awarded from the `resolve_contextually` flag or an unconditional `True`. For example, the whole-utterance case does not itself execute a whole-utterance comparison, and the vocabulary-capacity case is always correct. Several other module benchmarks inspect source strings for wiring markers. These are engineering checks, but their totals should not be interpreted as demonstrated capability.

`benchmark_module()` frequently labels candidate provenance only as `working-tree`. Historical source execution also uses current dependencies, so a pinned historical file is not necessarily a reconstruction of its complete historical environment.

**Next steps:**

1. Retain historical reports and name the current measurement limitations explicitly.
2. Replace flag-based semantic cases with actual calls and independently specified expected outputs. Report source/wiring checks separately.
3. Add unseen minimal pairs and adversarial cases for composition, morphology, constructions, pragmatics, discourse, uncertainty, alternatives, and reading continuity. Keep component results separate.
4. Record implementation commit, dirty-content identity when applicable, evaluator version, fixture identity, configuration, and relevant learned-state identity without exporting private state.
5. Label historical comparisons as file-only or environment-reconstructed. Record dependency drift and missing Git objects as limitations.

**Acceptance:** a deliberately broken candidate must fail the relevant behavioural case even if its wiring and labels remain intact. Preserve versioned comparisons. Do not claim a numerical capability gain merely from correcting the evaluator.

The persistent cognitive framework already provides separate answer keys, public-suite restrictions, and procedural preflight checks. Reuse those mechanisms. Keep held-out evaluation separate from Ina's readable learning material.

## 2. Reduce runtime-state document I/O

**Files:** `runtime_state.py`, `model_manager.py`, existing state/queue tests.

`get_inastate()` calls `_load_inastate_state()` for each key. `update_inastate()` locks, reloads the complete JSON document, and atomically rewrites it for one key. `monitor_hunger()` demonstrates several reads and separate writes in one operation. Similar patterns merit bounded measurement across the runtime.

**Next steps:** add a snapshot-read API and an atomic multi-key update, then migrate one coherent caller. Merge updates against the latest state while holding the existing lock. Avoid an indefinitely stale process-global cache. Preserve compatibility and queue semantics.

**Acceptance:** benchmark file reads, bytes written, lock waiting, and latency using disposable representative states. Verify concurrent writers, disjoint-key updates, coherent snapshots, corrupt-state behaviour, and queue preservation. Do not use live memory for fixtures. No system-wide saving is established until measured.

## 3. Add a matching index for recent-memory selection

**File:** `memory_index.py`, especially `_ensure_schema()`, `indexed_fragment_rows()`, and index schema-version handling.

The recent-selection query orders by `mtime_ns DESC, frag_id DESC`, but the schema has no matching index. On the actual schema, SQLite reported `SCAN f` and `USE TEMP B-TREE FOR ORDER BY`. A row limit bounds output, not the amount of sorting work.

An isolated in-memory probe with 100,000 synthetic rows measured approximately **13.016 ms before** and **0.015 ms after** the following candidate index, using the median of seven queries per condition:

```sql
CREATE INDEX candidate_newest ON fragments(mtime_ns DESC, frag_id DESC);
```

Afterward the plan used the candidate index. This is a synthetic query result, not an Ina-wide speedup; no production database was changed. The probe did not preserve a full environment manifest or raw timing samples, so repeat it before treating the ratio as a reproducible performance claim.

**Next steps:** evaluate a versioned sidecar migration, including the mechanism that decides whether an existing database is current. Do not rewrite source fragments. Compare tagged and untagged queries, actual selected identities and ordering, index size, build time, and write overhead.

**Acceptance:** identical results and tie-breaking, a useful query plan, bounded migration, retained rollback, and measurements on disposable HDD/NVMe fixtures where available. Returning fewer rows or losing witnesses is not an acceptable optimisation.

## 4. Preserve quantities and improve language role analysis

**Files:** `language_intelligence.py`, `tests/test_language_intelligence.py`, language benchmark registry and callers.

Isolated probes of the source routines produced:

| Input | Observed result |
| --- | --- |
| `I run.` | Subject and predicate were both absent |
| `I do not run.` | Predicate was `do`; `run` became the theme |
| `I give her a book.` | Recipient `her`, theme `a book` |
| `I give a book to her.` | Recipient `a`, theme `book to her` |
| `I opened 3 files.` | Tokenization dropped `3` |

Relevant causes include a small fixed `_VERBS` set, positional argument rules, and an alphabetic-only `_TOKEN_RE`. Morphology confidence is fixed at `0.98` despite unsupported cases. These results establish analyzer limitations; they do not establish that all live conversation uses this analyzer.

**Next steps:** trace real callers; preserve numerals and unknown spans; support both transfer constructions; handle auxiliary/main-verb distinctions; retain unresolved alternatives; lower confidence when analysis is unsupported. Prefer existing learned construction mechanisms over an ever-growing collection of special cases.

**Acceptance:** versioned, held-out minimal pairs for argument equivalence, quantities, inflections, negation scope, unknown verbs, and calibrated uncertainty. Measure successful abstention separately from incorrect confident analysis. Preserve native expression and optional English rendering.

## 5. Improve guidance selection and prove cross-domain transfer

**Files:** `learned_media_lessons.py`, `language_context.py`, `language_processing.py`, `model_manager.py`, `tests/test_learned_media_lessons.py`.

`load_output_guidance()` selects recent entries domain by domain until four slots are filled. It retains alignment keys but does not match them to the current task during selection. A synthetic probe with four audio-language lessons and one lyric lesson returned only the four audio lessons for the DAW consumer. This demonstrates selection starvation, not a measured musical regression.

**Next steps:** consider optional task/alignment matching and bounded coverage across relevant source domains. Keep source identity, alternatives, permission boundaries, and caller choice visible. Any balancing should apply to the temporary guidance view, not rewrite memory or erase witnesses. Recency alone should not imply relevance or authority.

**Acceptance:** compare guidance disabled, current selection, and candidate selection on unfamiliar briefs with fixed conditions. Measure output differences, usefulness, source attribution, overhead, and negative transfer. Include unrelated lessons and conflicting sources. Metadata arrival alone does not demonstrate learning.

Probe limitation: the isolated selector harness initially omitted required globals and failed; it succeeded after those harness omissions were corrected. Repository source was not changed to make it pass.

## 6. Evaluate creative output and voluntary revision

### Music

**Files:** `daw_engine.py`, `daw_window.py`, `music_exploration.py`, `music_delivery.py`, existing DAW tests and creative observatory.

The engine implements synthesis, tracks, stems, vocals, mixing, and export. Tool availability and waveform correctness do not alone demonstrate musical coherence or learning.

Use short retained projects to measure note timing, pitch, clipping, rendering bounds, edit control, and memory use. Separately collect blind listening and intent-adherence judgements. Track whether a chosen revision resolves a concrete problem. Profile rendering before introducing streaming, caching, or more synthesis complexity; only pursue those if measurements justify their compatibility and maintenance costs.

### Drawing

**Files:** `model_manager.py::_queue_autonomous_paint_seed()`, `paint_window.py`, `paint_runtime.py`, drawing tests.

The reviewed seed path forces `pattern = "burst"` whenever the selected reference role is `album_cover`. This is a coarse role heuristic; it does not itself demonstrate learning the cover's visual composition. The observed limitation is in this path, not a claim about all of Ina's drawings.

Compare diverse references at fixed emotional inputs. Inspect rendered results for composition, geometry, colour, intent, and voluntary revision. Candidate behaviour should allow references to inform choices without forcing imitation or a single pattern. Retain lineage and allow silence, refusal, or another creative choice.

### Speech

**Files:** `motor_layer.py::EmotionSynth`, `language_processing.py::_render_symbolic_synthesis_plan()`, `speech_activity.py`, `voice_identity.py`.

`EmotionSynth` maps emotions to pitch/loudness; the symbolic renderer combines sinusoidal layers. Pitch learning and valid audio output are narrower claims than intelligible speech. Speaker-recognition candidates are also a separate capability from speech generation.

Use voluntary short vocal tasks to measure repeatability, audible distinctions, pitch, amplitude safety, and blind intelligibility separately. Native vocal expression need not become English. If English pronunciation is evaluated, label it as an optional listener-facing task. Preserve the distinction between acoustic similarity and established identity.

**Shared acceptance:** technical checks, output quality, transfer, resource use, and Ina's actual choice/aftermath remain separate evidence. An action alone does not prove enjoyment or endorsement.

## 7. Measure code and cybersecurity outcomes

### Code

**Files:** `code_experiment_lab.py`, `numeric_evolution.py`, `tests/test_code_experiment_lab.py`, relevant versioned benchmarks.

The experiment lab already records source/data hashes, bounded execution, judgement, and review proposals. Build on it with small tasks whose hidden tests exercise unseen inputs, edge cases, and repair attempts. Score functionality, resource use, failures, and recovery separately. A successful process exit or complete disclosure form does not alone prove the proposed program is correct.

Retain existing isolation and review boundaries. A candidate's test results must not silently become authority to modify production code or promote itself.

### Cybersecurity

**Files:** `cyber_defence_learning.py`, `capability_assessments.py`, `authorized_security_lab.py`, `security_adversarial_assessment.py`, security tests.

`assess_response()` compares supplied action names with required/forbidden sets. This evaluates declared response content, not whether containment, evidence preservation, or recovery occurred. An authorization record is not proof of operational competence.

Retain policy checks and add owned synthetic outcome fixtures: verify resulting files, logs, containment, and restoration independently of the submitted action list. Include safe refusal and false-positive measurements alongside task success. Live or external targets are outside this proposal.

**Acceptance:** separately report policy compliance, actual outcomes, evidence quality, false positives, and recovery. Follow the repository's security-testing isolation rules: adversarial/security testing requires the complete Ina runtime tree to be stopped and independently verified, with disposable fixtures unable to access live memory. This report did not perform those tests or stop Ina.

## 8. Locate paper trading before assessing it

**Files found:** `crypto_market_learning.py`, `fiat_exchange.py`, `benchmarks/benchmark_crypto_market_learning.py`, `developmental_readiness.py`.

No order/portfolio paper-trading engine was identified in the reviewed public source. The crypto module explicitly excludes paper/live trading APIs. It provides chronological splits, historical features, future labels, and authority flags. Its benchmark checks those foundations rather than predictive profitability. This finding does not contradict observations of a separate or newer local capability.

**Next steps:** locate the implementation and identify its data flow before proposing integration. For prediction, assess chronological walk-forward generalisation against practical naive baselines and report calibration. Keep future labels outside predictor inputs; prevent overlapping target periods from contaminating partitions. For any separately authorized paper-execution evaluator, measure fees, spread/slippage, drawdown, turnover, and execution assumptions. Historical pattern recognition must not be reported as demonstrated trading profitability.

## Suggested local implementation sequence

1. Recheck revision, working-tree changes, repository instructions, existing APIs, and each finding. Mark already-fixed or inapplicable items rather than recreating them.
2. Correct benchmark interpretation and provenance while retaining historical evidence.
3. Compare the memory sort index; then introduce one state snapshot/batch-update caller with concurrency evidence.
4. Address language information loss and role analysis with versioned behavioural comparisons.
5. Trial the optional guidance selector, then collect domain-output and transfer evidence.
6. Evaluate code/security outcomes only in the appropriate isolated environment; inventory paper trading before implementation.

For every candidate, retain the previous usable version and report correctness, capability, uncertainty, quality, provenance, compatibility, resource cost, and rollback as separate applicable dimensions. Use the existing bounded test runner and benchmark registry. Do not interpret a passing suite as proof that no negative effect exists. Document unavailable live, browser, audio, full-suite, or hardware measurements plainly. Implementation and publishing remain separate decisions.
