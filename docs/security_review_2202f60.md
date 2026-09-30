# Review of 2202f60 and follow-up verification

Verified 2026-09-30. Implementation changes begun September 29 were retained in
37c99a6 by the time work resumed; the subsequent work adds regression coverage,
model-rerouting handling, and the user-authorized Mercury reading capability.

## Requested findings

| Finding | Implemented behavior | Evidence and limits |
| --- | --- | --- |
| Truncated commit review | Reject stdout over 256 KiB and invalid UTF-8. Compare full diff and immutable tree; commit reviewed objects and update the branch with an expected-old-HEAD check. Preserve unrelated staged content. | Oversized-tail regression and concurrent worktree-edit fixture pass. |
| Push destination bypass | Check all expanded fetch/push URLs, require one identical HTTPS GitHub URL, reject URL rewrite rules, pass the validated URL and exact commit ID to transport. Disable redirects and hooks. | Malicious pushurl and both rewrite forms rejected before any network request. No live push performed. |
| Lab authorization | Drafts cannot authorize execution. A trusted human-review entry point seals reviewed records with a process-local key; tampering, expiry and replay fail closed. | Synthetic consent/tampering/replay tests pass. The sole executor is one TCP connection to the exact IP and port, with peer verification and no payload. General lab tools remain unavailable; mock sockets only were exercised. |
| Kernel signatures | Caller-declared signature flags confer no build authority. `gpgv` verifies a bounded source snapshot against an operator-pinned signer/keyring and issues a digest-bound process-local receipt. | Actual cryptographic fixture passes; modified content, wrong signer and forged receipt fail. No real kernel source was verified or booted. |
| Model display | Requested model and app-server-reported model are separate. Turn overrides clear unconfirmed model state. Settings and turn-scoped rerouting events update the reported model; other threads/stale turns are ignored. | Focused protocol tests pass against installed schema fields. This reports provider claims, not independent proof of model identity. |
| Git status rescans | Cached snapshot; refresh on an explicit request or after relevant changes. | Eight UI-like reads: V1 eight scans, V2 one scan. No persistent benchmark workload added. |

Git's DNS check remains a preflight, not socket-level IP pinning. HTTPS identity
verification remains enabled. Broader live Discord/harness boundary verification
is still outstanding. Passing isolated fixtures does not prove absence of other
vulnerabilities or grant Ina external operating readiness.

## Other requested work

- Generate message uses the subscription-authenticated app-server, an ephemeral
  thread with confirmed read-only permissions, a bounded diff and structured
  response. Drafts remain editable and require the ordinary commit review.
  Main-thread events remain separate; command approvals stay user-routed.
  Provider-connected generation and visual browser review have not been exercised.
- Oxford British English definitions are available through an optional provider.
  Credentials and explicit enablement are required. No paid plan was selected and
  no credentials were copied. See `language_reference_sources.md`.
- Cognitive input iterators now stop at the declared budget. Shared-origin
  evidence does not become independent through repetition across lenses. This
  improves implementation behavior; live intelligence/learning gain is unmeasured.
- QEMU 10.2.2 is installed. No VM was booted or live security experiment performed.
- Mercury initially remained disabled, then the user explicitly authorized reading.
  Its private registry now permits read only. The voluntary personal-tool action
  lists bounded directories or reads 64-KiB source excerpts, rechecking consent
  and rejecting hidden paths and symlink escapes. No execution or handover enabled.

## Verification

The follow-up expressive-variation tool compares explicitly supplied observations
through the existing emergence-capture mechanism. It is reachable through the
voluntary personal-tool catalog and returns optional single-variable drawing,
language and music trials. Emotion, novelty-seeking and aesthetic preference
remain possible explanations alongside tool state; none is asserted or excluded.
It does not require self-explanation, execute trials, reward novelty, or write
memory automatically. Actual creative quality and cross-domain learning remain
unmeasured. Its V1 benchmark reports individual preservation, abstention and
non-execution checks; historical comparison is unavailable for this new tool.
The follow-up focused expression/personal-tool run passed 11 tests.

The main focused run passed 77 tests; the personal-tool integration run passed
7 tests. Runs used a network-disabled Bubblewrap environment with copied code and
temporary fixtures. Ina's live memory and credentials were not mounted. Host
process/service shutdown checks were recorded privately before adversarial inputs.
Python compilation and whitespace checks passed.

`benchmarks/security_review_comparison.py` compares current behavior against actual
source materialized from pinned revision `2202f60`. Separate signals all improved
for oversized review rejection, malicious destination rejection, tampered lab
records, declared signature rejection, cognitive iterator bounds, independent
evidence handling, lexical result bounds and revocable reading. It explicitly
reports live generation quality, network containment, cognition gain, real kernel
boot and visual GUI verification as unavailable. No readiness score is inferred.

The private Observatory inspection records unchanged implementation-only results,
missing readiness/creativity evidence streams, an absent code/game assessment and
unavailable Ina cyber responses. Recent fragment parse errors require separate
indexed integrity investigation; originals were preserved. The cause remains a
hypothesis, including the user's reported history of forced host shutdowns.
