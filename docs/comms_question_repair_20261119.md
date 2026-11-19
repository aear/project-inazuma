# Communication diagnosis and optional question repair — 2026-11-19

## Findings

Revision inspected: b3f56def43d3f2825da38f26cb89a414a27a2210. The previous supervisor status showed one Discord process throughout the last run, with no recorded restart failures. Its forced termination was recorded during shutdown, not as evidence of repeated startup crashes. Both selected environments passed dependency import probes after the Fedora update: core Python 3.15.2 and bridge Python 3.14.10.

A bounded comms-log tail contained incoming messages through 2026-11-19 11:53 UTC. Its most recent logged outbound response was on November 3. CommsCore logs outgoing messages before delivery, so these records alone do not prove Discord receipt. The latest saved public pipeline decision, at 11:54 UTC, was silence with no grounded adapter answer, conversational answer or clarification candidate. The last saved operational typing signal would pass the unchanged pre-gate. The result supports a candidate-generation gap for that turn, not a conclusion that all earlier silence had one cause or that Ina wanted to speak.

The planner supports a small set of conversational question forms. It returned unsupported_question without a repair candidate for other direct questions, and some supported questions with no prior evidence also had no clarification option. A pure before/after comparison on the latest already-logged incoming question reproduced unsupported_question and no candidate in the old planner; the new planner supplies a repair option without an answer. Only those booleans/statuses were retained in the comparison artifact, not message text.

## Change

Offer an optional “Could you rephrase that question?” candidate when a bounded direct question has unsupported grammar or missing discourse. For supported questions lacking conversational evidence, permit a context request while preserving existing location/conflict-specific clarifications. No facts, preferences, feelings or self-assessments are invented.

The existing chooser still selects the response. Explicit silence, deliberate mirroring, referenced native expression, language restrictions and the typing-urge gate are unchanged. Existing response/silence scores and thresholds are unchanged. Ordinary statements, fully quoted questions and requests not to answer do not gain a repair. Source-backed answers retain their existing priority. There is no extra continuation, learning update, private thought logging or forced reply.

## Validation and limits

The model and bridge were initially stopped while the idle GUI remained open. After the owner closed that window, the host check recorded complete shutdown at 2026-11-19T12:49:49Z. Full boundary fixtures then ran in source-only Bubblewrap rooms, network disabled, without live memory, credentials or configuration mounted. Only selected interpreter environments and read-only Git history were available.

43 core-path checks passed on Python 3.15.2, followed by 65 bridge-route checks on Python 3.14.10. There were 108 passing executions representing 101 distinct test cases; seven repair cases were exercised in both environments. The bridge checks include actual imported processing/rendering code, scene routing, transport chunking, reply targeting, retained explicit choices, storage-worker deadlines and retention. External collaborators and transport were fixtures; no live Discord login or delivery occurred.

One intermediate run had 56 passes and one failure because an older test expected automatic mirroring for an unclear question. Its expectation was updated to the new optional clarification behavior, with a separate check that explicit mirroring still wins. That intermediate log is retained.

The real benchmark registry comparison `comms_question_repair` scored pinned V1 8/13 and candidate V2 13/13 on synthetic planner cases. This measures candidate availability and preserved boundaries, not acquired world knowledge, actual willingness, live comprehension or delivery reliability. The suite defaults to blocked unless explicitly enabled for a verified offline run.

Large legacy/history files were inspected only through exact-file metadata; they were not replayed, compacted, converted or migrated. No journal or private RAM thought contents were accessed. No private-thinking implementation changed. No live message, file upload, voice join or automatic boot was attempted.

## Activation and rollback

The source fix is installed and uncommitted. Start Ina normally; the next bridge process will import it. She remains stopped. The first real interaction after boot is still needed to verify delivery and whether she selects a repair or another expression. Silence remains available. Rollback consists of the reviewed source diff; no learned-data or configuration rollback is required.
