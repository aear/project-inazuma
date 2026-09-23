# Experience Cognition Coordination

`experience_cognition.py` adapts several useful conditional-computation ideas
for Ina's Experience Learning Model architecture. It coordinates an already
bounded event snapshot; it is not a language model, a memory system, or an
autonomous scheduler.

## Mechanisms

- **Sparse experience routing** scores relevant cognitive subsystems and
  activates at most four. Below-threshold routes and capacity rejections remain
  visible. No qualifying route is a valid result.
- **Independent attention lenses** inspect causal, temporal, affective, social,
  and contradictory relationships without averaging their disagreement away.
- **Selective transient-state propagation** may propagate, hold, or decay a
  derived state. It preserves source references and never edits the source
  experience.
- **Adaptive computation** begins with one pass and grants only a finite number
  of additional passes for explicit novelty, contradiction, uncertainty, or
  stakes. Resolution, exhausted evidence, retained uncertainty, and budget
  exhaustion are all valid stopping conditions.
- **Multi-horizon prediction** keeps immediate, near, and later hypotheses
  separate. Every admitted prediction needs source references and an explicit
  observation that could disconfirm it.
- **Epistemic state** treats `known`, `uncertain`, and `unknown` as successful
  outcomes. An unknown result has no answer, identifies missing evidence, and
  may suggest bounded observations without authorizing them.

## Memory boundary

The coordinator does not read the fragment store, persist a plan, train
parameters, retrieve evidence, or alter Experience Cycle continuation budgets.
Memory remains owned by Ina's existing custom stores and federation of
witnesses. `ExperienceCycleEngine.plan_cognition(...)` is a convenience entry
point and has no storage side effects.

## Text-expression integration

`ThoughtProcessor.prepare_communication(..., cognition_event=...)` can now
coordinate an event before creating its output-neutral expression intent. The
returned communication plan carries both the full bounded cognition plan and
text-expression guidance; the intent carries only a compact epistemic summary.

When that plan is supplied to `TextRealiser`, an `unknown` result permits only
an explicit acknowledgement of not knowing, a request for evidence, or
silence. An `uncertain` result permits qualified expression, a request for
evidence, or silence. Definite text is rejected in both cases. This constraint
is opt-in at the realiser boundary so existing channels do not change output
before their own integration benchmark and review.

## Input contract

An event may provide normalized signals between `0.0` and `1.0`:

```python
event = {
    "signals": {
        "novelty": 0.7,
        "contradiction": 0.8,
        "uncertainty": 0.9,
        "causal": 0.6,
    },
    "required_evidence": ["causal", "temporal"],
    "evidence": {"causal": ["observation:17"]},
}
```

Evidence values are bounded references, not payloads. If `temporal` evidence is
absent here, the epistemic result can be `unknown`, identify `temporal` as
missing, and leave continuation optional.

## Measurement

The `experience_cognition` module benchmark retains an explicit V1 baseline and
scores V2 separately across routing, agency, attention, transient state,
cadence, prediction, uncertainty, and the memory boundary. Unit coverage lives
in `tests/test_experience_cognition.py`; the Experience Cycle integration test
also verifies that planning creates no storage writes.
