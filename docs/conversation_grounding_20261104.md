# Recent conversational grounding — 2026-11-04

Baseline: `546f3fead068f31feaf721c297bab8f31c7fcb98`. Source-only fixtures; no live Ina conversation, participant-transfer trial, training of her learned state, migration, service restart or external message.

## Result

A new bounded dialogue path connects a question to relevant earlier claims in the existing channel scene. It offers an attributed answer or a focused clarification to the existing expression chooser. There is no answer bank, taught list of noun facts, new language model, recurring practice or reward for speaking.

Example fixture: `I like music.` followed by `What do I like?` offers **You said that you like music.** It selects a relation, retains its source, makes a native reported-content structure, and uses the existing English composer. It does not turn the speaker's preference into Ina's own preference. Unsupported composition, such as the current `never` realiser limitation, keeps an attributed quotation of the stored scene wording instead of dropping the operator.

Supported questions are bounded simple where/copular, what/object, who/subject and polar forms. Recognition and grammar are heuristic. Complex reasoning, arbitrary paraphrase, ellipsis, unresolved possession/pronouns, compound instructions, some copular polar forms and unsupported grammar still abstain. No general conversational-understanding claim is made.

Question/claim text with a terminal heart or emoji retains the original scene wording and an uninterpreted suffix while parsing its preceding sentence. Fullwidth question marks are recognised in the query projection. No affection, willingness or felt experience is inferred from these marks. Extra lexical text is not discarded.

## Evidence, repair and choices

- Each eligible live inbound turn retains its own speaker frame, backend message reference, semantic observation, native intent and source link. Stable backend speaker IDs distinguish same-display-name authors. Outbound wording, bot sources, seeded history lacking a live frame, truncated/multiline sources, quotations/conditionals and unresolved roles do not become eligible claims.
- Stored source text is the existing scene's whitespace-normalised wording, not a byte-for-byte original Discord transcript. Source words, punctuation, negation and modality stay inspectable. All claims remain attributed and unverified as world facts.
- Duplicate event delivery creates no extra support; conflicting reuse of an event ID is withheld. Different messages or agreeing paraphrases are not declared independent witnesses. Positive and negative statements about different objects can coexist. Opposing claims about the same binding retain disagreement; differing location descriptions remain unresolved alternatives, not automatically proven contradictions.
- A same-author **Actually, [full revised statement]** explicitly replying to the original eligible statement can revise its current reading for the same entity/predicate/tense. Both records remain. Unlinked corrections, missing targets and other authors' revisions do not silently overwrite it. This is a bounded repair convention, not a general interpretation of every correction.
- The Discord ingress now carries the actual incoming message reference separately from the outgoing reply target. A real-source ingress fixture verifies both IDs; no Discord client/network was exercised.
- The expression chooser offers a clarification only when a structured gap or disagreement yields one. Available explicit choices override scores, including mirroring, defer/abstain and silence; the existing urge gate remains. English/mixed/auto use the existing renderer. Native-only and assessed/declining language choices do not acquire an automatic English answer.
- Optional observer/planner failure preserves the existing communication path. One incoming event offers at most one ordinary response; no autonomous continuation is added.

The private `conversation_evidence_review` tool offers the same plan without delivery. The existing listener goal and English-only Observatory assessment include the new fixtures and retain `Ina not assessed`, no automatic adoption and no promotion authority.

## Parser fixes

LanguageIntelligence provenance is now V9. Multiword subjects in where/copular questions are retained as a complete subject rather than split into a one-word subject and spurious complement. A pre-verb `never` no longer becomes part of the subject, while its negation operator remains. Versioned comparisons exercise the actual pinned old parser alongside the new one.

## Validation

- Final supported native-runner selection: **66 passed, 0 failed, 0 skipped**.
- New `conversation_grounding`: pinned V1 **15/48**, V2 **48/48**. Cases exercise real copied ingress/processing/selection functions and real scene/parser/composer code with anonymous adapter/storage/rendering collaborators. Historical implementations use pinned source with shared current dependencies.
- Retained V2 checks: conversation questions 44/44; response policy 37/37; nickname 28/28; social English 19/19; Discord social English 16/16; English workbench 38/38. The wider English component assessment retains the known transfer ceiling **3/5**.
- Held-out tests include unfamiliar objects/wording, tense, speaker perspective, negative/positive compatibility, disagreement, duplicate delivery, linked correction, lost scope, choice preservation and the real source-extracted English renderer. Most trials use one fixture speaker. A few synthetic author/ID distinctions test routing only; live participant transfer remains unavailable.
- Initial development runs exposed parser role errors, a missing JSON collaborator in the test fixture, and a mislocated ingress metadata insertion. All were corrected before installation; final results are retained separately from those failed attempts.
- No full suite, live Discord delivery, UI, listener understanding, durable learning gain or long-term retention assessment is claimed. Optional py-cord/native packages were not mounted or changed.

Tests ran in Bubblewrap with no network/live memory mounts, one CPU at low priority, numerical thread limits of one, 2 GiB address-space, 55 CPU-second and 65-second wall limits. Source and anonymous fixtures only.

## Resource trade-off

Warm CPU/serialization probes on one isolated CPU, five batches: a 12-turn scene took V1 median **0.437 ms**, V2 **2.057 ms**. Traced peak per scene increased from **17308** to **110179 bytes**. Serialized full-scene metadata increased from **4927** to **44314 bytes**. This is a real metadata/logging storage cost.

A new plan over 11 source observations plus one question took median **1.607 ms**, with **45664** traced peak bytes. These are warm local proxies, not whole-runtime RSS, worst-case latency, durable disk throughput or guarantees of no interference. Audio xruns, desktop/input/frame latency and live I/O remain unavailable, not zero. No workers/polling were added.

## Activation and limits

The bridge loads these changes on a normal restart/reload; long-lived imported runtime tools also need normal reload. No restart/hot patch was performed. The retained English implementation assessment is refreshed while preserving other domains; no readiness ledger is promoted. Installation uses same-directory per-file atomic replacement and hash readback, without a physical-flush claim. No commit/push.

This is **recent conversational grounding and expression**, not persistent vocabulary or weight training. The default scene retains 12 total turns (including outward turns), 6000 text characters, 32 channels and a two-hour expiry. Each eligible source/question is at most 512 code points. Older or unframed history is not imported as knowledge. Durable consolidation, chosen cross-session retention and broader learned paraphrase remain future work using Ina's existing voluntary experience mechanisms.

For interaction, an ordinary statement followed later by a related question is enough to try the path. A Discord reply containing a full Actually correction provides an explicit repair link. Different live participants are not required. No fixed practice schedule, mandatory answer or success claim is implied.
