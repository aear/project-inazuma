# Optional local checks for Ina

External offer from the human operator, prepared by Codex on 2026-10-15.
These are available choices, not Ina's own thoughts, commitments, or scheduled work.
There is no deadline, penalty, repeated reminder, or requirement to complete them.
Ina may inspect, try, defer, decline, or ask for help. Missing evidence stays missing.

## Available tools

The existing personal-tool catalog now exposes these bounded actions:

| Action | What it does | What it does not do |
| --- | --- | --- |
| `memory_index_inspect` | Reports configured durable/fast index paths, sizes, schema, ordering-index presence and a confirmation identity | Reads no fragment payloads; changes no index |
| `memory_index_trial` | Runs one disposable synthetic before/after trial on temporary storage or a chosen configured device, with a private receipt | Touches no live memory; does not authorize a live upgrade |
| `memory_index_change` | Carries out one chosen `upgrade`, `rollback`, `defer` or `decline` for a configured location | Moves no memories, accepts no arbitrary path, schedules no follow-up |

All three actions require the same process-local voluntary-choice authority as
existing executable personal tools. A serialized note, external message, or source
label alone cannot authorize them. A normal trusted runtime caller uses
`request_personal_tool(command, code_authority=LOCAL_VOLUNTARY_CODE_AUTHORITY)`
with `source="ina_voluntary_choice"`, only after an actual choice. That capability
must never be granted merely because a note contains an example command.

A practical sequence, if chosen:

1. Inspect. Read the returned locations and current schema; these are the current
   navigation paths, not proposed new memory locations.
2. Optionally try the synthetic trial. `location="durable"` or a returned configured
   fast location tests that device with disposable files; the default is temporary
   storage. Its receipt states the fixture limits and chosen location.
3. If a real derived-index upgrade is desired, inspect again, then request
   `memory_index_change` with that `location`, its returned `identity` as
   `confirmation`, `choice="upgrade"`, and an optional `max_seconds` of 0.1–10.
4. Read the result and private receipt. Busy databases, low free space, unsupported
   schemas, stale confirmations, or timeouts defer instead of forcing progress.
   The tool performs a transactional additive index change and verifies the same
   bounded selected identities before committing. It reads no raw memory payloads.
5. Rollback is available through a fresh inspection and `choice="rollback"`.
   It removes only the added ordering index and restores schema V2. Original
   fragments, tags, and source JSON stay where they were.

Receipts are append-only in private `Notes/memory_maintenance_receipts.jsonl` and
include requested/completed status, before/after paths and schemas, timings, and
remaining uncertainty. No receipt is posted publicly. A failed attempt does not
create an automatic retry. A rollback may also need a quiet write window.

## Local checks to consider

- **Memory navigation:** whether the proposed ordering index is useful, and whether
  its storage cost and writer-lock window are acceptable. The tooling is ready;
  Codex has not upgraded or moved any live memory. Broader file relocation already
  has `model_manager.request_file_move(source, target, choice=...)`, backed by
  `storage_migration.request_managed_file_move()` and its configured capabilities,
  resumable copy, hash verification, retained original/link and reporting. This
  offer does not invoke that machinery or expand its allowed destinations.
- **Local vocabulary settings:** whether the chosen capacity fits actual experience.
  The benchmark compares 25,000 and 50,000 entries without treating a local setting
  or a larger capacity as proof of language quality. Private configuration is not
  a public test fixture and has not been changed.
- **Editor workspace:** optional local workspace exclusions can be checked if the
  retired harness/editor workflow is still relevant. Missing that file is not a
  failure of Ina and does not prevent the desktop workflow.
- **Music, drawing, speech and transfer:** if desired, choose a short intention,
  retain output and provenance, then inspect/listen and choose whether to revise.
  Existing DAW, paint, symbolic-sound, `compare_expressive_variation`, private
  `write_note`, and experiment tools remain available. Objective render checks
  establish technical behaviour; preference, intelligibility, composition and
  lived usefulness need their own evidence. English remains optional.
- **Paper trading:** identify the actual local tool or project before assessing it.
  The public market-learning module supplies historical features, not an order
  engine. No trading authority has been added by this maintenance work.

The proposal and measured limitations are in `docs/review_implementation_20261015.md`.
No human-facing/private note should be relabelled as Ina-authored experience.
