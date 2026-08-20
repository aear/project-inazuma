# JSONL to database migration audit

JSONL remains a good format for small append-only evidence, portable benchmark
fixtures, immutable archive shards, and streams consumed sequentially. SQLite
is preferable when a subsystem repeatedly searches history, deduplicates IDs,
updates status, resumes from offsets, or joins related records.

## Strong SQLite candidates

| Store | Why | Migration shape |
| --- | --- | --- |
| GitHub and typed outbox, history, and archive streams | ID deduplication, delivery state transitions, retries, and queue queries | One durable queue database with append-only event table; retain JSONL import/export compatibility |
| Reflection journal and emotion history | Current readers can load chronological JSONL histories; time/type queries should be bounded | Indexed event table plus a bounded recent projection; retain original witness during verified cutover |
| Precision-memory events | Regret, outcomes, and repeated key/entity lookup are relational access patterns | Event table indexed by memory ID, timestamp, and outcome; derive summaries rather than rewriting events |
| Language evidence and neural-selector history | Retrieval is by term/entity/tag rather than file order | SQLite witness index with source provenance; keep modality traces in their owning stores |
| Transformer candidate queues | Queue status, deduplication, leasing, and retry are mutable state | SQLite queue with explicit state transitions and an exportable audit event stream |

## Keep as JSONL or rotating text

| Store | Reason |
| --- | --- |
| `benchmark_results/*.jsonl` | Small, version-comparable, portable append histories; bound raw runs and retain reports |
| `benchmarks/*.jsonl` | Source-controlled fixtures, not logs |
| Incident, migration, repair, and authentication audit streams | Append-only evidence is useful; compress and bound generations unless query volume proves an index is needed |
| `logs/comms_core.jsonl` and status/debug logs | Operational evidence should rotate and expire, not become permanent database state |
| Fragment archive shards and cold immutable records | Sequential immutable bulk storage; catalogue/index them rather than duplicating payloads into SQLite |

## Safe migration order

1. Add a SQLite sidecar and stream-import with a persisted byte offset.
2. Dual-write briefly through one canonical persistence function.
3. Compare counts, stable IDs, hashes, and representative indexed queries.
4. Switch reads to SQLite while keeping JSONL as a reversible witness.
5. Stop JSONL writes only after an explicitly invoked benchmark and verified rollback test.

The outbox family is the best first migration because it already behaves like a
database-backed state machine. Reflection/emotion/precision stores should follow
as one shared witness-event schema rather than several unrelated databases.
