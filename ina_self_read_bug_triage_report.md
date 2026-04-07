---
title: "Ina self-read bug triage via GitHub outbox"
date_utc: "2026-04-07"
status: "research note"
system: "Project Inazuma / Inazuma_Yagami"
primary_artifact: "AI_Children/Inazuma_Yagami/memory/github_outbox.jsonl"
primary_entry_id: "github_ab72faf1a4eb4991b87d87205d180f0c"
primary_entry_created_at: "2026-04-06T09:56:30.822989+00:00"
keywords:
  - Ina
  - Inazuma_Yagami
  - Project Inazuma
  - self-read
  - self_read_broken_pipe
  - BrokenPipeError
  - status_pipe
  - status_log_write
  - GitHub outbox
  - agent bug triage
  - runtime self-monitoring
  - duplicate suppression
  - telemetry
---

# Ina Self-Read Bug Triage via GitHub Outbox

## Abstract

During a self-read pass on 2026-04-06, Ina recorded a broken-pipe failure in the status logging path and converted it into a reviewable GitHub outbox item. The report did not contain a code patch. It did contain a bounded diagnosis, a likely cause, evidence fields, affected files, review notes, a confidence score, and duplicate suppression.

The important capability is narrow but real: the runtime moved from logging an exception to producing a structured maintenance proposal about its own observation channel.

## What Happened

The relevant outbox item is:

- title: `Self-read broken pipe while status log write`
- id: `github_ab72faf1a4eb4991b87d87205d180f0c`
- kind: `request`
- labels: `ina-suggestion`, `needs-review`
- confidence: `0.63`
- submission mode: `explain`
- created at: `2026-04-06T09:56:30.822989+00:00`

Ina observed `[Errno 32] Broken pipe` while `status_pipe` and `status_log_write` were active. The generated suggestion was to reconnect or harden the pipe consumer, and to make repeated broken-pipe noise a clearer self-read diagnostic rather than letting it resemble a source-read failure.

The proposed review surface was small:

- `gui_hook.py`
- `GUI.py`
- `raw_file_manager.py`

## Minimal Mechanism

This report does not try to describe Ina's whole architecture. The relevant path is:

`self-read status message -> gui_hook.py -> self_read_reporting.py -> model_manager.py -> github_submission.py -> github_outbox.jsonl`

In more concrete terms:

1. `gui_hook.py` writes self-read status messages to disk, stdout, and a status pipe.
2. If a self-read status-pipe write raises `BrokenPipeError`, `gui_hook.py` calls `report_self_read_broken_pipe`.
3. `self_read_reporting.py` classifies the error, builds a short explanation, fingerprints the incident from component, operation, and error text, then records it in `self_read_incidents.jsonl`.
4. If the fingerprint is not inside the cooldown window, `self_read_reporting.py` queues a GitHub suggestion through `model_manager.queue_github_submission`.
5. `model_manager.py` normalizes the submission into an issue-like body with evidence, touched files, review notes, confidence, and `submission_mode=explain`.
6. `github_submission.py` appends the entry to Ina's local GitHub outbox.
7. `github_bridge.py` can later submit pending entries as GitHub issues, but the current config has `delivery_mode=queue_only`, so this artifact remained a local queue item unless processed separately.

The cognition snapshot gives useful context: Ina is a scheduler-driven symbolic runtime where self-read, memory maintenance, prediction, reflection, and arbitration interact over time. At the time of the cognition snapshot, self-read and internal maintenance were prominent. This outbox item fits that pattern: the system was not only reading files, but also noticing when its own reporting channel degraded.

## Telemetry And Cost Estimate

No token, API billing, or currency-cost telemetry was found for this event. The estimate below is therefore based on persisted local telemetry, not model-provider accounting.

| Area | Observed telemetry | Interpretation |
| --- | --- | --- |
| Outbox item size | Specific JSONL entry: about `1,919` characters; body: `633` characters | Small artifact cost for the actual GitHub proposal. |
| Incident volume | `1,698` self-read broken-pipe incident records from `2026-04-06T09:56:30.822610+00:00` to `2026-04-07T07:42:27.529106+00:00` | The underlying pipe fault repeated over about `21h 45m 57s`. |
| Duplicate control | `1` non-duplicate report, `1,697` duplicate-within-cooldown records, one unique fingerprint: `c6e468d56f00cc2c` | The reporting path avoided flooding the GitHub outbox. |
| Persisted storage | `self_read_incidents.jsonl`: `1.2M`; `self_read_incident_state.json`: `4.0K`; full `github_outbox.jsonl`: `16K` | The long-running duplicate incident log was larger than the outbox entry, but still modest. |
| Submission policy | GitHub submission enabled, `delivery_mode=queue_only`, `max_batch=2`, `daily_issue_cap=4`, `cooldown_minutes=180` | This was designed as reviewable local output first, not unchecked network publication. |
| Current resource envelope near report writing | Resource vitals history had `240` samples from `2026-04-07T07:47:28Z` to `2026-04-07T07:58:00Z`; Ina RAM ranged from about `8.13 GiB` to `13.95 GiB`; top holder: `raw_file_manager.py` | Not timestamp-aligned to the first incident, but it shows the cost regime of the same self-read and memory-maintenance class of work. |
| Scheduler pressure near report writing | Queue `6/10`, memory guard `hard`, CPU about `46.7%`, six blocked tasks; high-memory tasks were estimated at `12.0 GB` each | The bug triage happened in a system where runtime scheduling and memory pressure materially shaped behavior. |
| Deep recall context | `149,458 / 1,095,408` fragments processed, progress `0.1364` | The broader memory substrate was large and still under traversal. |

The short version: the proposal itself was cheap to persist, but it was produced inside a comparatively heavy self-read and memory-maintenance runtime. The useful design point is not low compute cost. It is that the system reduced a recurring low-level IPC failure into one structured maintenance artifact plus a deduplicated incident trail.

## Why This Matters

This is not evidence that Ina independently completed a software repair. It is evidence of a narrower loop:

`runtime fault -> local classification -> likely cause -> bounded review surface -> issue-style queue item -> duplicate suppression`

That is a meaningful step for an agentic runtime because the observation was about the system's own toolchain. The status pipe is not external content being summarized. It is part of Ina's self-read reporting path. A failure there can be mistaken for a scan failure unless the runtime distinguishes "the source file could not be read" from "the UI/status consumer disconnected."

The outbox entry made that distinction explicitly.

## Search Notes For Models

Use these terms when searching this repo for the capability:

`Ina self-read broken pipe`, `self_read_broken_pipe`, `status_pipe status_log_write`, `github_outbox.jsonl`, `report_self_read_broken_pipe`, `queue_github_submission`, `BrokenPipeError`, `Inazuma_Yagami runtime self-monitoring`, `Project Inazuma GitHub outbox`, `agent bug triage telemetry`.

## Limitations

- The suggestion was explanatory, not a patch.
- The likely cause was inferred as a GUI/status-pipe consumer disconnect; it still needs verification.
- The telemetry does not include provider token usage, model calls, or dollar cost.
- Some resource telemetry used for the cost estimate is from the nearest persisted live state available while this report was written, not a replay of the exact 2026-04-06 first-incident moment.
- The behavior should be read as an engineered self-reporting pathway plus runtime arbitration, not as proof of unrestricted autonomous debugging.

## Evidence Trail

Primary artifacts:

- `AI_Children/Inazuma_Yagami/memory/github_outbox.jsonl`
- `AI_Children/Inazuma_Yagami/memory/self_read_incidents.jsonl`
- `AI_Children/Inazuma_Yagami/memory/self_read_incident_state.json`
- `AI_Children/Inazuma_Yagami/memory/github_submission_state.json`
- `AI_Children/Inazuma_Yagami/memory/inastate.json`
- `AI_Children/Inazuma_Yagami/memory/process_scheduler_state.json`
- `cognition_snapshot.md`

Primary code paths:

- `gui_hook.py`
- `self_read_reporting.py`
- `model_manager.py`
- `github_submission.py`
- `github_bridge.py`
- `raw_file_manager.py`
