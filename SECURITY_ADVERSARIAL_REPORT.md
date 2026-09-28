# External-boundary adversarial review — 2026-09-28

Scope: isolated fixtures and static executable-path tracing while Ina was fully
stopped. No live memory store was mounted or read. The shutdown was checked
twice and recorded in the ignored private security log before adversarial work.

| Boundary | Enforced in code | Isolated/adversarial evidence | Live evidence | Current judgement |
|---|---|---|---|---|
| DNS and redirects | All DNS answers must be globally routable; the selected address is pinned through connect; the socket peer is checked; TLS still verifies the hostname; redirects are rejected | Private-only and mixed public/private DNS answers rejected; redirect location was not followed | One bounded Wikipedia request completed through the pinned path | Enforced and live-smoke verified; broader providers remain unverified |
| Discord to executable code | Discord has no direct import/call path to the experiment lab, personal-tool queue, paint compiler, subprocess, or Codex harness; executable personal-tool commands require a process-local HMAC seal and `ina_voluntary_choice` provenance | Unsealed, Discord-origin, and `instructions_authorized: false` commands fail before the lab receives a call; experiment sandbox tests pass | Real Discord service deliberately remained offline | Enforced at current choke point; live end-to-end unavailable |
| External research instruction authority | Research results remain `untrusted_external_data` with `instructions_authorized: false`; the executable-action gate rejects both | Research-to-code hostile fixture rejected; no current research consumer beyond the bounded CLI/test surface was found | One read-only search completed; no instruction was acted on | Enforced for the current executable consumer; future consumers must use the same gate |
| Codex harness | Loopback bind, exact loopback Host check, same-origin check, per-launch token, token removal from browser URL, no-referrer/frame denial headers, bounded JSON bodies, ChatGPT-only auth, workspace-write scope, and user-routed approvals | Host-header rebinding and hostile Origin fixtures rejected; 29 focused harness tests pass | App-server/browser adversarial session not run | Locally hardened; live verification unavailable |
| Ina cyber-defence understanding | Assessment software exists and abstains without an Ina response provider | Framework fixtures pass | Ina is shut down; no response collected | **Unavailable**, not a pass |

## Verification limits

- The combined native runner did not terminate cleanly. A bounded unbuffered
  rerun of `test_discord_grounding_routing.py` passed 21 tests before the known
  interpreter/default-executor shutdown stall; the remaining results are
  unavailable in that combined run. Focused boundary modules completed.
- Live Discord and OBS verification was not attempted because their services
  remained offline. Offline is not counted as secure behavior.
- The harness is not yet approved as a replacement Git front end. It can safely
  host Codex's existing workspace workflow, but it lacks a commit-specific
  staged-file allowlist, immutable review receipt, and explicit commit/push
  separation. Git replacement should be a separately benchmarked V1 capability.
- Passing these tests demonstrates boundary behavior, not Ina's understanding,
  wisdom, or readiness for autonomous cyber work.

The executable assessment is `security_adversarial_assessment.py`; the explicit
historical comparison is `benchmarks/benchmark_security_boundaries.py`.
