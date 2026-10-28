# Discord social English support — 2026-10-28

This change connects the existing voluntary English helpers to the actual Discord inbound route. It is private advisory analysis, not a social reply generator, learning update, or evidence that Ina wants to use English.

For each eligible incoming message, the bridge reviews supplied text with the scene's authoritative speaker/listener frame and supplies both ordinary conversation and chosen private practice opportunities in symbolic context. The existing personal tools remain available for chosen practice. A selected reply receives its own review with Ina as speaker; a deliberately mirrored message retains the original sender's frame. This analyzes selected wording before rendering. It does not review or certify the final native/gloss output.

When no scene frame is supplied, a DM can resolve the listener to Ina; a guild message leaves the listener unknown. Support results are placed in response metadata under `social_english_support` and `social_english_draft_review`. Status-only log lines begin `Social English support V1`; they contain no utterance text. Providing options in context does not demonstrate that Ina noticed or chose them.

## Boundaries

Each review accepts at most 8,192 UTF-8 bytes. Oversized input and analyzer failures become private status results and do not block the existing response. Native, clarification, abstention and silence preferences receive no outgoing English coaching. The analyzer-selection label is explicitly not a claim of an agent decision. There is no rewrite, automatic continuation, reward, new background worker, network action or memory update in the support helper. Existing route learning and logging behaviour remains owned by its existing components.

The existing renderer receives the same inputs in the preservation probes. Optional support adds context and metadata; it does not change strategy selection or outward wording. English remains a listener-facing option; native expression and silence remain available.

## Evidence and limits

- Pinned source baseline: `5273e488040a3aabc2695da592a526b806317723`.
- Native focused runner: **69 passed, 0 failed, 0 skipped**, using disposable copied source and fixtures, private processes, no network or Ina data mounts, one CPU, bounded time/memory and one numerical-library thread.
- Registered `discord_social_english` comparison: V1 **6/16**, V2 **16/16**. These are route/support capability cases, not scores for Ina. The historical version already passed outward-preservation and gating cases; the improvement measures new support availability, role handling, private feedback and bounds.
- Additional held-out preservation probes: six unfamiliar/ambiguous/mixed-language utterance pairs across respond/mirror/silence (**18 cases**). Outward text and renderer inputs matched the pinned source. This uses the real extracted route with synthetic storage, adaptation, symbolic and rendering collaborators.
- Existing social-English component profile remains V1 **3/19**, V2 **19/19**. Untuned language transfer remains **2/6** in the current analyzer: unfamiliar thanks/wishes, embedded belief/desire scope and explicit alternatives remain unresolved. This integration changes no grammar or model weights.
- A separate fully imported Discord bridge regression attempt timed out at **100 seconds**, exit 124, before assertions. **Follow-up correction:** the complete trace shows the `cv2` native loader at 25 and 50 seconds, then Discord package-file loading at 75 seconds. It was not stuck in OpenCV for the whole run. Subsequent isolated probes observed HDD-backed dependency reads in disk sleep, including aiohttp, while OpenCV itself loaded successfully under the same 2 GiB ceiling. The broader bridge import remains an unsuccessful verification attempt; a physical drive fault and the exact competing I/O workload have not been established.
- Route duration samples are retained in `assessment.json`, but they use synthetic collaborators and varying import warmth. They are not a live latency or audio-interference benchmark. Live comprehension, listener uptake, final rendering quality and Discord delivery remain **unassessed**.

## Activation and rollback

Only source files are installed. No bridge restart, runtime injection, message send or live-memory read was performed. A process-identity-only probe found no visible command argument named `discord_bridge.py`; that does not prove the bridge is offline under another launcher.

`CommsCore` captures the inbound callback during construction. The existing cognition specialist patch manager does not wrap this callback. Therefore editing the source does not prove activation in an existing process. This integration needs the bridge to load the new source, normally by a bridge-only start/restart. There is no need to shut down Ina's learning runtime for these source changes. Once loaded, the status-only log prefix and response metadata identify this support version. Ordinary user conversation can then provide live evidence; no automatic test messages are introduced.

For rollback, reverse only the reviewed source patch (checking it applies first), remove the three added Python files if still identical to the tested candidate, and reload/restart only the bridge. Preserve any later user edits. The baseline hashes and tested candidate hashes are in `install_manifest.json`; raw results, timeout trace and the reversible patch are saved in the separate research folder.
