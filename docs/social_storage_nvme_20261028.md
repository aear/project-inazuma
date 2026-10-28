# Social map on NVMe, with an optional SQLite backend — 2026-10-28

**Completed:** the active social-map JSON was moved to Ina's configured NVMe. All **11 contacts / 2,463 source bytes** were preserved exactly. The original HDD file is retained as a cutover backup, and its former path is a compatibility link to the active NVMe file. The bridge writer was paused for **0.0304 seconds** and resumed; cognition was not stopped and no bridge process was restarted.

**SQLite is prepared, not active.** An integrity-checked, private SQLite snapshot of the same contacts exists on NVMe, and the optional backend is implemented and tested. JSON remains authoritative so already-running communication/humour readers and the dashboard retain their existing format. No trust tags, owner roles, display names or unknown contact fields were inferred or edited during the move.

## Active placement and preserved data

The configured JSON path is now:

`/run/media/sakura/3b7828ff-4792-44b6-8d04-e370dda00fb1/Ina_Fast_Storage/AI_Children/Inazuma_Yagami/memory/fast_runtime/social/social_map.json`

The old HDD path remains a link. The precise backup, snapshot and private configuration-backup paths are in `live_move_receipt.json`. The active storage directory is mode 0700; JSON, database snapshot and private configuration backup are mode 0600. No contact contents or credentials were copied into this research report.

Only three configuration values were added: `discord.social_map_path`, `discord.social_map_backend` (`json`) and `discord.social_map_storage_mount`. A semantic comparison verified the remaining configuration values were unchanged. Original formatting was retained for a reviewable diff.

The source was small enough for the explicit 4 MiB cap. The existing `storage_migration.migrate_file_and_link` seam performed checksum-copy, backup and compatibility linking. Before cutover the operator helper verified bridge PID, process owner, checkout, creation identity and stopped threads, then refused an open legacy social-map write. The source and target bytes and preserved HDD backup matched after migration. The source helper was called with no child attribution, so this operator-requested movement was not appended as an Ina-chosen cognitive action.

Source inspection at the pinned revision found Discord as the only code writer. Early communication and humour use its read API; the dashboard directly reads JSON. The brief writer pause protects the verified copy against these known code writers. It is not a general lock against an external editor or another operator changing the file concurrently.

## Backend implementation

`social_map_store.py` implements an explicitly selected SQLite backend with schema/application identification, fixed page size, indexed per-user lookup, ordered full payload retention and transactional per-contact read/modify/write. Concurrent edits no longer depend on competing whole-file rewrites. Unknown fields and original contact order are retained; duplicate normalized user IDs are refused for review rather than silently collapsed.

Each successful database contact update records its before/after payload in the same transaction. The imported original JSON bytes and their hash are also retained in the database. There is no background learning, automatic migration or automatic promotion of the snapshot.

The backend uses a 200 ms SQLite busy timeout, `synchronous=FULL`, rollback journaling (`DELETE`) and a 256 MiB database-page ceiling. Source/aggregate contact payload is capped at 4 MiB, each contact at 64 KiB, and the contact domain at 10,000. Reads/exports reject an exceeded budget instead of returning a partial authority list. The aggregate size counter updates in the contact transaction; the initial inactive snapshot without that optional counter remains supported through a transactional backfill.

Missing databases are not created during lookup. Missing, corrupt or unsupported selected databases grant no fallback trust from an older JSON file. Invalid updates roll back. Trust interpretation remains owned by the existing social-map functions.

`social_map.py` retains its existing public helpers. It now supports explicit SQLite selection while JSON remains the default. JSON updates preserve wrapper metadata, retain compatibility links, use the existing atomic-write helper and refuse to replace a damaged source with a fresh one-contact map. Configured storage-mount absence declines reads/writes rather than creating a replacement store under an unmounted path. Configuration loads reuse the existing layered loader.

The actual SQLite snapshot was created using SQLite 3.51.2, rollback journaling and full synchronization. WAL was not enabled. For future live database backups, use the [SQLite backup API](https://www.sqlite.org/backup.html) rather than assume a raw main-file copy captures every committed state.

## Measurements and validation

A copied-source, no-network test room ran **60 regressions with 0 failures/skips**, including the prior heartbeat worker, fully imported bridge command/token/selected-expression checks, canonical verified-file-migration tests and social-store preservation/concurrency/rollback cases. After adding the aggregate growth guard, **11 focused social-store checks passed**, including a new counter/backfill/rollback case. These counts overlap; they are not 71 independent tests.

Registered historical comparisons against `1f5293506dc87c3f1592e7d26b923c58545a7757`:

- `social_storage`: **V1 4/7, V2 7/7**, covering record/authority preservation, targeted edits, wrapper metadata, missing selected-database handling and damaged-source retention.
- `discord_storage`: **V1 2/7, V2 7/7**, retaining the heartbeat/authority checks from the preceding fix.

These measure storage/bridge mechanisms, not Ina's social or English capability.

A separate explicitly invoked NVMe comparison used synthetic fixtures, with 16 point lookups per sample and three alternating sample pairs. Both formats used the same device and were warmed; this is a format/lookup comparison, not proof of a physical-drive speedup.

| Synthetic fixture | JSON median per lookup | SQLite median per lookup |
|---|---:|---:|
| 8 contacts | 0.0439 ms | 0.1337 ms |
| 4,096 contacts | 3.3191 ms | 0.1508 ms |

JSON was faster for the small fixture. SQLite's indexed lookup was about 22 times faster in the larger one. The present map is small; transactional correctness and scaling are the database's practical motivations, not a claimed immediate speedup. Main-loop latency, full cold startup, live audio interference and actual Discord message delivery were not benchmarked here.

## Heartbeat fix and current runtime

The previously tested heartbeat fix could now be installed after the filesystem recovered. Its source and social-store source hashes are retained in `install_manifest.json` and the reversible `source_changes.patch`. The abandoned empty/code installer temporary file was archived outside the checkout and removed only after the original installer was gone and its contents matched an allowed known hash.

The running bridge was resumed and observed sleeping normally rather than stopped. No new gateway login or user message was triggered. A bridge-only restart will load the installed bounded-I/O code. That does not require stopping Ina's cognition. The social-map placement is already active through the compatible file path/configuration.

This move addresses the observed social-map path. It does not repair Btrfs or eliminate all other HDD access by the bridge or cognition.

## Future SQLite activation and rollback

Do not point the running system at the cutover SQLite snapshot merely because it exists. JSON updates can continue after that snapshot, so it can become stale. A proper activation needs a fresh verified import of the latest JSON under a coordinated writer boundary, reload of all API consumers, and adaptation of the dashboard's remaining direct-JSON read. That activation was intentionally not performed while those older readers were running.

The retained HDD backup is the **cutover point**, not a continuously updated mirror. There is no hidden HDD dual-write on the hot contact path. Use explicit verified checkpoints for later backups; no recurring job was created.

For rollback of the placement, quiesce the bridge writer and copy the **current NVMe JSON** to a fresh HDD file, verify it, then atomically replace the old compatibility link with that current file and update the configured path. Keep the NVMe copy and original cutover backup. Restoring only the old cutover backup would lose later interactions or trust changes. Do not reverse-link directly onto a link that points back at the source; that can create a link cycle.

For source rollback, apply the reverse patch only after checking it against later edits. The new cached-config reader is compatible with the NVMe JSON path; do not revert data and code independently without checking all active readers. No unrelated memories, learned weights or index migrations were changed.
