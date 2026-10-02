# Cognition refinement V3 — 2026-10-02

## Changes

- Evidence availability requires bounded, nonblank string references rather than
  mere truthiness. Empty entries, numbers and mappings no longer count as witnesses.
- Explicit `counterevidence_references` retain conflict even if a caller omits the
  numerical contradiction signal. Source-origin claims remain caller-supplied and
  unverified; confidence remains a heuristic, not calibrated probability.
- `plan_experience_cognition`, including the existing ExperienceCycleEngine route,
  now returns `hypothesis_comparison`. Supply up to eight `hypotheses` with `id` and
  `claim`, and eight `observation_candidates` with `id`, `question` and
  `expected_outcomes` keyed by hypothesis ID. Pairwise comparison retains predicted
  disagreement, indistinguishability and missing predictions separately. A proposed
  check with complete outcome coverage may be suggested; no check is executed.
- Outcome labels are compared literally after trimming/case-folding. Different
  wording is not proof of different real outcomes. This is a bounded planning aid,
  not semantic entailment, expected information gain or automatic experimentation.
- Expression Core now accepts cognition V1/V2/V3. Previously it rejected the V2
  plans already produced by the runtime. Unrecognised epistemic states now fall
  back to unknown rather than permitting confident expression. Existing supported
  state behavior is retained. The old benchmark's V1-only compatibility assertion
  was updated to enumerate supported schema versions.

## Verification and limits

34 focused tests passed in copied, network-disabled storage, with no live memory
mounted. The first run exposed the expression compatibility failure and included
one nonexistent test filename; these were corrected before the passing rerun.

The standalone cognition comparison materializes historical source from `ee1eb68`.
V3 passes all seven separate checks, retaining the supported-evidence and
shared-origin behaviors while improving malformed-evidence, counterevidence and
proposed-check handling. The historical implementation has no hypothesis-check
component; its absent capability is not an observed bad execution.

Registry checks: cognition V2 9/9, V3 7/7; expression V6 17/17, V7 4/4.
The retained benchmark cases run against current code; only the separately pinned
cognition comparison is a historical source comparison. A pinned historical
expression comparison remains unmeasured; V7 compatibility cases and current V6
regressions are the available evidence, not a claim of full historical parity.

No runtime restart, ongoing background workload, automatic continuation, custom
memory modification or parameter training was added. Live cognition gains,
calibration, independent witness verification, and an AGI readiness claim remain
unavailable. Cases here are deterministic developer fixtures, not held-out evidence
of general intelligence. Runtime consumers must supply meaningful alternatives
and expected outcomes for the new planning capability to help.
