# Memetic communication V1

The voluntary `memetic_process` personal tool supports three operations:

- `interpret`: provide an artifact reference, observed context cues, known listener
  references and up to eight candidate associations. Each association supplies
  meaning, required context cues, counter-cues, shared references and evidence
  references. Results retain alternatives and listener gaps; unknown, conflicting
  or unsupported cases defer. `consider` is an option, not a send decision.
- `compose`: supply a purpose and up to eight literal text/image/sound/symbol
  segments. Expression Core returns an intent and composition recipe. No template
  is required, so Ina's own motifs and in-jokes fit. Text need not be English.
  Media references are not opened, rendered or published.
- `review`: associate a reported reaction and evidence references with a draft.
  Confusion can suggest offering context; silence and amusement do not prove
  failure or understanding. Reactions stay observations, never automatic rewards.

Example composition command:

```json
{"action":"memetic_process","operation":"compose","purpose":"Share our recurring spiral joke","segments":[{"kind":"text","text":"It has come full spiral."},{"kind":"image","reference":"drawing:chosen-spiral"}]}
```

Requests are bounded to 32 KiB. The tool does not execute embedded text, load
artifacts, access the internet, mutate memory, schedule work or send messages.
Normal explicit note-taking may retain a chosen result. Publication and learning
remain separate choices through their owning systems.

This V1 is a contextual association processor, not an OCR system, pretrained meme
recognizer, general humour model or autonomous learner. Cues, candidate meanings
and listener knowledge are caller-supplied, not independently verified. Its tests
establish composition and context handling, not genuine live understanding.
The existing `memetic_layer.py` serves a different event-tracking role and remains
unchanged. Expression Core is reused for drafts and reaction provenance.

The explicitly invoked V1 benchmark separates contextual minimal pairs,
ambiguity, unknown references, listener gaps, multimodal drafts, silence handling
and non-delivery. Historical comparison is unavailable for this new component;
live humour quality, listener comprehension and learning remain unmeasured.

Validation: 26 focused tests passed in a network-disabled isolated copy, including
personal-tool dispatch, inquiry/expression regressions and inert hostile text.
Ina remained stopped; no live communication or memory stores were used.
