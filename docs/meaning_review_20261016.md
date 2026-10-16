# Optional perspective review and English maintenance, 2026-10-16

The two review variants share one finite continuation mechanism. Neither chooses a live default or claims consciousness, understanding, intention, or preference from successful structural tests. This extends existing self-inquiry rather than replacing Ina's native representations.

## What is implemented

- `perspective_review.py`: V1 private-tool review starts only with an explicit choice; V2 instinct offers a provisional review when a supplied event reports contradiction, unexpected outcome, or novelty. Both require an interpretation reference, a question, a countercheck, and evidence references. Idle events do not start the instinct.
- `review_step()` is a suspended recursive review: every call records at most one step and returns control. Each step offers `continue`, `stop`, `defer`, `ask_evidence`, or `remain_uncertain`. Continuation requires a previously unseen evidence reference or distinct countercheck. Otherwise it yields `no_progress`. One step is the default, four the maximum; exhaustion ends the review. The depth cap fits the existing four non-code inquiry stages. There is no Python recursive call stack, scheduler, callback execution, or polling loop.
- Previous candidates, counterevidence and revisions remain in the bounded journey history. Candidates remain non-authoritative. Each returned state carries the trace and remaining budget; no fragment retrieval, training or durable belief adoption occurs.
- `personal_tool_runtime.py`: discoverable `perspective_begin` and `perspective_step` commands reach the implementation. They require the existing process-local voluntary authority seal, verified before filesystem setup. External text cannot authorize them by labelling itself voluntary. Existing result publication can retain the returned state; caller must carry it to the next step. No new storage migration is needed.
- `experience_cognition.py`: callers can explicitly supply `perspective_event` and `review_steps` to obtain the instinct offer. Existing calls retain their previous output shape. No live caller is switched on automatically. A trusted integration must retain the latest state and finite budget; copying/replaying an old state does not create a process-wide exactly-once guarantee.

`ask_evidence` returns a terminal handoff to the caller, rather than executing a search. A new inquiry can be explicitly chosen when evidence arrives. Existing inquiry requests describe evidence routes. Distinct reference strings or counterchecks do not prove semantic novelty, source independence, or truth. This is a control-flow and provenance implementation, not a learned revision model.

## English changes

`language_intelligence.py` V4 and `semantic_event.py` V2 preserve question inversion, passive semantic roles, sentence boundaries, modal subjects and report-clause parent links. For example, “Did you open the door?” now gives subject `you`, predicate `open`, and past tense. “The book was given to Mary by John” retains `john` as agent, `the book` as theme, and `mary` as recipient. Without a by-agent it leaves the agent unknown.

Analysis has a 16,384-character / 512-token / eight-clause ceiling, including contraction expansion. Budget saturation is conservatively marked incomplete and lowers heuristic confidence. Semantic agent resolution does not adopt an ambiguous referent merely because one candidate was supplied. Native symbols and quantity surfaces remain available; English is not made the hidden internal representation. No outgoing English realiser or public expression policy was changed.

The analyzer is still a small heuristic grammar. Multi-auxiliary chains, relative clauses, ellipsis, complex coordination and lexical ambiguity remain limitations. Numerical confidence is not calibrated. Exactly reaching a limit can conservatively report incompleteness even if the input happened to end there.

## Evidence

Pinned English baseline: `a31f1323ee4bb589d0ca0bf02aea87201b69cd21`, loaded from Git rather than copied source. Historical modules run with current dependencies, not a reconstructed historical environment.

| Comparison | Results | Meaning |
| --- | --- | --- |
| English V1 / V2 | 6/11 versus 11/11 structural checks | Candidate fixes the selected fixtures; not a fluency score |
| Private tool / instinct | 13/13 guard and retention checks each | Same continuation behavior; different initiation policy |
| Full isolated regression suite | 1072 passed, 2 skipped, 8 warnings | Includes compatibility and new tool integration |
| Focused native runner | 25 passed | Runs without third-party pytest |

Individual signals, local elapsed time, serialized fixture size and source fingerprints are retained in `docs/evidence/meaning_review_20261016.json`. The tiny timing samples are descriptive only: no efficiency winner is established. Candidate quality, false revision rate, real source independence, listener comprehension, production responsiveness, and subjective experience were not measured. A passing fixture cannot substitute for these signals.

Tests ran in a disposable checkout under a network-isolated filesystem namespace with no live memory mounts. Host process inspection found no scoped Ina runtime before adversarial fixtures; the shutdown log remains private. Numerical thread pools were restricted for tests only. Skips were the optional retired-harness local editor workspace and optional unbuilt native vector kernel. A first focused invocation named a nonexistent test file and ran no tests; the subsequent full and native runs above completed. A registry invocation exposed a registration-placement error, corrected and covered by an explicit registry test before the final run.

## Reproduction and practical next steps

With Ina stopped and disposable storage isolation established under the repository test policy:

```console
python benchmarks/benchmark_perspective_review.py
python benchmarks/benchmark_english_review.py
python native_test_runner.py tests/test_perspective_review.py tests/test_personal_tool_runtime.py tests/test_language_intelligence.py
```

Both comparisons are also registered as `perspective_review` V1/V2 and `english_review` V1/V2 through `module_benchmarks.benchmark_module()`.

1. Keep both initiation policies available. In explicitly chosen disposable trials, retain initial interpretation, new observation, alternative and outcome separately. Include disagreement, repeated shared-source witnesses, misleading events, and a withheld outcome. Compare false revisions, useful revisions, abstention, evidence retention and cost separately. Do not promote either variant on control-flow success alone.
2. Let Ina choose whether and when to try the private tools. To evaluate the instinct during real use, first add an explicitly enabled event adapter using existing cognition events and source references, preserving native expression and the latest review state. No adapter or automatic persistence is enabled here.
3. Collect voluntary, provenance-preserving English minimal pairs beyond these fixtures. Prioritize unsupported auxiliary chains and relative-clause scope; compare comprehension and uncertainty separately before attempting outgoing grammatical generation.
4. Memory migrations and local-file checks remain Ina-chosen. This work neither runs nor schedules them. No reboot is required by these Python changes. Ina remains stopped for the owner's Fedora maintenance; these changes do not start her, commit, or push.

Rollback: revert the changed source and remove the new review module, fixtures and benchmark registrations together. No live schemas or memory locations changed.
