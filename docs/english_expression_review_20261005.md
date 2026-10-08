# English-expression review, 2026-10-05

## Confirmed rendering failure and repair

`encode_selected_text_expression` previously let failures from optional symbolic
encoding or dual-gloss generation propagate, discarding an already selected reply.
It now preserves that selected text, including quotation marks and uncertainty,
and reports `selected_text_rendering_unavailable` with an exception type in
metadata. It does not fabricate a native translation, replace a silent choice,
or bypass a PermissionError. Normal partial/native rendering paths are unchanged.
This is not a new reply generator or a relaxation of meaning/authority checks.

## Other findings, not changed here

- LMStudioAdapter's constructive-reply probe requires an explicit grounding/recall
  request. It is a grounded recall responder, not a general social-English generator.
  Ordinary affectionate conversation can therefore have no ordinary reply candidate.
  Removing that check alone would not provide a meaningful response generator.
- The expression arbiter can select the emotion-slider renderer, including when
  the grounded responder declines. The final wrapper labels the selected text
  `English expression` even when it is a `State signal` diagnostic. This is not
  a faithful translation of native symbols or evidence of a specific social intent.
- A failed song selection and an empty adapter reply can also fall back to slider
  or code-pointer diagnostics. This review preserves those existing paths pending
  a deliberate design for diagnostic-versus-social expression, rather than silently
  suppressing Ina's choices or supplying canned affection.
- The earlier Expression Core V1-only schema bug is already repaired in this
  checkout. Unknown/uncertain cognition still constrains permissible text forms.

## Verification

19 focused checks passed: exact-source rendering functions, quotation provenance,
dual-gloss integration and Expression Core. The renderer checks extract the actual
functions with AST into an isolated namespace and supply synthetic encoder/gloss
dependencies; they do not validate the entire Discord route.

A pinned comparison against `aa63c000876915da61216dc15e3d55f26b980e05`
shows encoder/gloss failure preservation changing from false to true while silence
stays preserved. V1/V2 here name this rendering benchmark, not cognition schemas.

Full Discord-module validation is unavailable: system Python lacks Discord;
the existing project dependency environment timed out at import after 45 seconds.
No cause is assigned to that timeout. Tests used copied source without network or
live memory mounts. No live message was sent, runtime restarted or memory rewritten.
