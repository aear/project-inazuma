# ML and cognition reliability improvements — 31 October 2026

## Installed changes

The shared ML kernels now preserve vector direction and cosine comparisons at finite numeric extremes. Ordinary input behaviour, absolute epsilon damping, unequal-length/full versus overlap norms and existing hash IDs are retained. Non-finite cosine inputs return zero rather than establishing a match. This is not a general data-repair layer: other numeric APIs retain their prior handling of invalid values, and learned embeddings are not rebuilt.

The adaptive QTransformer decoder accepts an optional `source_event` on `learn_mapping`. Within one object, repeated delivery of the same identified example does not increment its counts again; a conflicting reuse of that ID is retained as a refused update. Caller IDs are not authenticated. The window is bounded to 512 events, is process-local and is not saved in the decoder file. Calls without an ID retain their previous counting behaviour. Existing callers must supply stable observation IDs to benefit from replay suppression.

Learning rejects invalid weights and malformed/oversized observation labels before mutation. Updates are serialized within one decoder object. Failure to save restores its prior in-memory counts and leaves the event available for retry. Persistence reuses the project's atomic JSON writer and its existing durability policy, keeping the existing statistics schema; no live decoder file was opened by this development run. This does not coordinate separate decoder objects or processes sharing one file.

Equal or near-equal top mapping weights remain ambiguous rather than being resolved alphabetically. Competing candidates are exposed in bounded mapping assessments, and ambiguity/capacity limits do not become an arbitrary learned or scaffold word. Unobserved states retain their existing scaffold behaviour, while a unique learned mapping remains available. Assessments explicitly do not claim calibrated probability or truth. The meaning-decoder origin is versioned V3; the unrelated candidate-superposition path remains unchanged.

Thought decisions now group contributions by thought and option. Identical signed weights do not multiply a shared thought's support; distinct signed readings share that thought's contribution instead of each receiving a full vote. Different descriptions remain available in `reason_observations`, including when their numerical weight repeats. This deliberately changes decisions that previously relied on several readings of the same thought adding up. Different thought IDs are not proof of independent sources; provenance-aware callers still need to distinguish correlated thoughts.

Decision input consumption is bounded to 32 options, 64 thoughts and 128 evidence rows, with one look-ahead item for truncation detection. The evidence summary exposes ignored duplicate delivery and input limits without treating repetition as corroboration. Confidence remains heuristic.

The pure experience-cognition layer now accepts only bounded, nonblank reference strings for attention witnesses, transient propagation and prediction sources/disconfirmation. Whitespace around supplied origin labels cannot turn one origin into two. Provenance remains caller supplied and unverified. The planner's existing schema and no-memory-write/no-training/no-autonomous-continuation boundary remain intact.

## Verification

Pinned comparison: `aefcc6e7285872ffa239602309147ff56e550f1d` versus the tested candidate. The registered `ml_cognition_reliability` suite changes from 9/29 to 29/29 across numerical correctness, compatibility, learning provenance, ambiguity, a toy learning trial, decisions, source handling and bounds. The replay-ID API is absent from the predecessor; its absence is not presented as a failed execution of an API it did not have.

146 focused native-runner checks passed, zero failures or skips. Tests included Decimal-reference numerical comparisons, seeded ordinary-vector parity, extreme/subnormal magnitudes, concurrent delivery of one learning event, save/reload compatibility, simulated save failure, replay-window bounds, ambiguity, duplicate decision support, preserved descriptions, existing cognition/thought processing and English/Discord component regressions.

A toy decoder trial compared a fresh query after one wrong example was delivered repeatedly alongside four separately identified examples for the other label. Replay-aware counting retained the observed majority. This is one synthetic bit bucket, not evidence of language learning, unseen-state generalisation, calibrated confidence or broad ML accuracy.

All existing versions in the q_decoder, thought_processor, experience_cognition and ina_ml_distribution registries ran. Their retained current-version checks passed. Some old registry versions are existing static/source-presence baselines; only the new suite is presented as a pinned source comparison. The existing English implementation profile was rechecked, with its retained transfer result still 3/5. No live comprehension or expression gain is claimed.

## Resources

The first candidate passed correctness checks but was about 43% slower on ordinary cosine calls. Extra common-path copies were removed before installation. The final probe warmed both versions, alternated their order across five paired batches and used 200 calls per batch on one isolated CPU.

| Vector size | Previous median/call | Candidate median/call | Previous peak traced allocation | Candidate peak traced allocation |
| --- | --- | --- | --- | --- |
| 64 | 8.89 microseconds | 4.51 microseconds | 544 bytes | 1,072 bytes |
| 256 | 34.18 microseconds | 16.00 microseconds | 544 bytes | 4,144 bytes |

These are local helper timings, not overall Ina speedups. The stable norm calculation uses additional temporary argument storage; extreme-case fallback also allocates scaled copies. Larger dimensions, live desktop/audio interference and whole-process RSS were not measured. No persistent worker, polling loop, automatic benchmark or background training task was added; decoder synchronization uses an object-local lock.

## Scope, activation and evidence

Development used copied sources, anonymous fixtures, no network and no live memory mounts. The assessment records an exact copied-Python-source fingerprint. Installed files are checked against tested hashes; changes are retained in `source_changes.patch` and `install_manifest.json`. No learned weights, memories, vocabulary IDs or decoder stores were migrated, read or rewritten. No model replacement/promotion, forced English, process restart, external message, commit or push was performed.

The English component-assessment snapshot can be refreshed from this isolated result without appending readiness evidence or attributing it to Ina. Existing other-domain reports are preserved. Live training, retention, reasoning quality and listener-understanding measurements remain unavailable.

Changes become active in each process at its next normal code load/restart; an existing process keeps its already-loaded functions. Rollback must check for later edits before reversing this task's source patch. The bounded replay cache requires no on-disk migration or rollback.
