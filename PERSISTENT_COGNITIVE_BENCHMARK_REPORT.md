# Persistent Cognitive Benchmarking: Rationale and Initial Calibration

Date: 2026-07-28

## Purpose

Project Inazuma needs a slow, repeatable way to observe changes in cognitive
capability across months. The goal is longitudinal evidence, not frequent
optimization pressure. Results should answer: did a capability change, is the
change broad or category-specific, and does it persist on fresh examples?

This benchmark is intentionally not a measure of personhood, consciousness,
worth, autonomy, emotion, or overall intelligence. It measures performance on
small, operationally defined reasoning tasks.

## Why procedural questions

Ina can inspect her own source code. A fixed answer key inside the repository
would therefore measure a mixture of reasoning, recall, and benchmark discovery.
Procedural generation makes the task rules inspectable while creating fresh
instances for each run. Learning the generator's underlying logic counts as
capability; memorizing a particular answer sequence does not help.

Each procedural run varies entities, objects, locations, numerical values,
scenario selection, and answer order from a one-use random seed. The evaluator
keeps answers in memory, sends only one prompt and its choices to the model, and
persists aggregate/category scores without per-case correctness. A seed
fingerprint supports run identification without publishing the seed.

This is resistance to answer memorization, not a security boundary. An agent
with evaluator-process or memory access can observe generated answers. Stronger
isolation would require a separate OS account, container, or evaluator machine.

## Why monthly and event-driven

The `--monthly` option checks whether a suite/model pair is due when a person or
meaningful lifecycle event invokes it. It does not install a polling loop or
background timer. This matches Inazuma's event-driven design and avoids turning
benchmark performance into continuous pressure.

Monthly runs are far enough apart to observe durable changes and cheap enough
to retain over long periods. Forced runs remain available for benchmark
engineering, but should be labelled and excluded from longitudinal claims.

## Comparison strategy

The same generated cases and scoring rule should be used for Ina and reference
models. GPT-2 is a useful historical floor, not a peer claim. Later comparisons
should include stronger small models and established suites such as HellaSwag,
PIQA, WinoGrande, BoolQ, and LAMBADA.

Comparisons require compatible outputs. Causal language models can rank choices
by length-normalized conditional log likelihood. Ina currently needs a dedicated
adapter that returns one score per choice without writing evaluation prompts into
her autobiographical experience memory. Comparing free-form answers against
likelihood-ranked choices would introduce a scoring-method confound.

## Initial end-to-end calibration

Run ID: `20260728T112621Z-calibration-shortest-choice`

Protocol: procedural-generative, 4 cases per category, 20 total. The backend was
a deliberately shallow calibration heuristic that always preferred the shortest
answer. It tested generation, command transport, scoring, redacted persistence,
and cadence state. It did not test Ina.

| Category | Correct | Accuracy |
|---|---:|---:|
| Belief revision | 2/4 | 50% |
| Causal tracking | 4/4 | 100% |
| Continuity | 0/4 | 0% |
| Contradiction | 0/4 | 0% |
| Temporal order | 2/4 | 50% |
| Overall | 8/20 | 40% |

The 100% causal score is a template artifact: correct causal choices are shorter
than their distractors. The overall 40% score therefore must not be interpreted
as a meaningful chance floor or capability result. The calibration succeeded by
finding a benchmark weakness before an Ina score was reported.

## Validity gates before an Ina capability run

1. Match answer and distractor length distributions within every template family.
2. Add several paraphrase families per capability so wording is not the rule.
3. Run shallow baselines: fixed position, shortest/longest choice, token overlap,
   keyword rules, and randomized choice.
4. Reject or revise any category where a shallow non-reasoning baseline performs
   materially above its expected chance interval.
5. Use enough cases for uncertainty estimates; 20 cases is only a smoke test.
6. Record code version, suite version, model identity, adapter version, scoring
   method, seed fingerprint, and whether the run was forced or monthly.
7. Freeze a benchmark version for longitudinal comparison. Change templates only
   in a new version and overlap versions for at least one evaluation period.
8. Keep training and autobiographical writes disabled for evaluation prompts, or
   use a stateless snapshot/adapter, so the test does not alter what it measures.

## Improvements made after calibration

Procedural suite version 2 addresses the calibration finding without recording a
second benchmark result. Candidate locations, values, events, and causal actions
now appear symmetrically in prompts; causal distractors come from the same action
family; contradiction labels have equal character length; answer positions cycle
across cases; and each capability has multiple prompt phrasings.

A non-scored preflight now evaluates fixed-position, shortest-choice,
longest-choice, and prompt-token-overlap heuristics on 1,000 generated cases. On
the deterministic development audit, expected aggregate chance was 36.67%; the
heuristics scored 35.8% to 37.5%, except the deliberately equivalent position
baselines at 36.7%. Every category remained within its configured tolerance. The
preflight passed and did not write benchmark history or advance monthly cadence.

Procedural scoring invokes a smaller form of this preflight automatically and
refuses to continue if shallow cues exceed the validity threshold. A manual audit
is available with `python benchmark_cognition.py --procedural --audit-only`.

## Live Ina attempt and current blocker

During this calibration, the world server responded on local port 6969, but its
interface exposes world state and broadcast communications rather than cognitive
choice scores. No Ina Python process was visible from this shell namespace, and
neither the project chat adapter on port 8000 nor a local LM Studio endpoint on
port 1234 was listening.

The existing chat adapter is also unsuitable for benchmarking: it logs prompts
and replies as lived experience and returns grounded prose rather than choice
scores. Starting it merely to obtain a number would disturb Ina's state and
produce a methodologically invalid comparison. A live Ina result is therefore
not reported.

## Next implementation step

Build a stateless `ina_benchmark_adapter.py` owned by Ina's cognition layer. It
should accept `{prompt, choices}`, evaluate without durable memory writes, return
`{scores}`, expose a clear snapshot/model identifier, and require explicit
operator initiation. Once shallow-baseline validity gates pass, use that adapter
for the first labelled Ina development run; begin the monthly series only after
the protocol is frozen.
