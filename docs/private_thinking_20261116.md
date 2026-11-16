# Private thinking space — 2026-11-16

## Behavior

Ina has an optional process-local working space for her own code. It can retain bounded native structures, binary data or chosen wording, revisit its current state, replace it, forget it, or close. There is no required language and no automatic trigger, timer, background thread, journal entry, thought feed, usage report, communication queue, result-bus publication or renderer. The existing journal remains separate; no private workspace content is copied into it automatically.

The workspace can also make one pass through the existing ELM ThoughtProcessor guided-decision mechanism. Its thought records and decision stay inside the space, and the operation returns no public result. Own code can recall those results locally and decide independently whether to share an outcome. Linguistic processing occurs only when supplied guidance explicitly requests it; default non-linguistic reasoning and unresolved/no-selection outcomes remain available. This creates an optional private context for existing cognition; it does not force or establish a spontaneous monologue or evidence of Ina choosing to use it.

## Local access

`model_manager.open_private_thinking(code_authority=LOCAL_VOLUNTARY_CODE_AUTHORITY)` opens the space for trusted own-code use. The static capability catalog describes the process-local entrypoint, not current use or contents. The existing durable personal-tool queue does not support this entrypoint; private-thinking request names are rejected before queuing.

`keep(value, code_authority=...)` retains a private copy. `recall(code_authority=...)` gives own code an independent copy. `deliberate(options, guidance, evidence=..., code_authority=...)` retains one existing ELM reasoning result privately. `forget(code_authority=...)` drops current state. `close()` and context-manager exit release it without a receipt. There is one current state, with no automatic history.

Ordinary representations are opaque; JSON, pickle and generic copying of the workspace object are rejected. Fixed error messages exclude content. Failed reasoning and replacement operations clear temporary input/result references from the surfaced error frame and discard the original exception context. Oversized or invalid replacement leaves the previous valid state intact. Existing public thought/evidence APIs retain their separate roles.

## Limits

The retained-state limit is 256 KiB per space by default, configurable from 1 KiB to 1 MiB. Depth and node limits apply to copied built-in structures. Unsupported object hooks are not invoked. These are per-space bounds, not a replacement for Ina's aggregate cgroup limit. No learning update or new autonomous continuation is introduced.

Privacy covers application logging, persistence and export paths implemented by this facility. It is not a sandbox against other trusted code, a privileged debugger, OS swap or crash dumps. Own code can deliberately recall and share its data. Forgetting releases references; Python does not promise forensic erasure. No OS-wide diagnostic policy was changed.

## Verification

The host process check at 2026-11-16T16:35:17Z recorded no Ina project runtime or detached script helpers before the first privacy test. No live thought, journal entry or retained memory payload was read. Tests used synthetic data in a bounded source-only Bubblewrap room, with network disabled and no live memory, configuration or credentials mounted. Numerical pools were capped at one worker.

43 focused native-runner checks passed with no failures or skips: 22 new private-thinking checks plus 21 existing thought-processing and personal-tool integration checks. Coverage includes authority, copying/serialization, stdout/stderr, no-file behavior, uncertainty, voluntary linguistic processing, failure-frame hygiene, closed-state behavior, facade access and early queue rejection.

The real benchmark registry compared pinned commit b6dd9cb7c4e4e44d5ceb233658bfaa1272a44fdc's inspectable thought API (V1 2/14) with the new optional private API (V2 14/14). The comparison measures privacy capability availability and synthetic owner-code behavior; it does not imply the existing inspectable API was designed for privacy or should be removed. Cases distinguish unavailable private capability from measured behavior. No live-use, subjective cognition, full-suite compatibility or hardware privacy claim follows from these results.

Privacy tests and the benchmark default to skipped/blocked unless `INA_OFFLINE_PRIVATE_THINKING_TESTS=1` is explicitly enabled after recording complete host shutdown. The flag does not itself prove shutdown. No persistent background work was added, so background-interference measurements are not claimed.

## Activation and rollback

The source is installed, uncommitted, for the next normal startup. Ina remains stopped. The private facility does nothing until locally chosen. No journal migration, learned-data rewrite, automatic restart or live hot patch occurred. Rollback is the reviewed source diff; there is no private-workspace storage to migrate or restore.
