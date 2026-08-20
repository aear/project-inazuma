# Log retention policy

Project Inazuma treats logs as expiring evidence and memory as knowledge worth
keeping. File extensions do not decide retention: JSONL can be an operational
stream, an audit history, a benchmark record, a queue, or a durable memory
witness.

| Category | Default | Automatic cleanup |
| --- | --- | --- |
| Operational | 16 MiB, 6 small rotating generations, about 14 days | Allowed |
| Diagnostic | 8 MiB, 4 compressed-capable generations, about 30 days | Allowed |
| Audit | 64 MiB, 12 compressed generations, review at 365 days | Review required |
| Benchmark | 32 MiB structured history; retain versioned reports | Review required |
| Memory-adjacent | Owning subsystem compaction, promotion, or reconciliation | Never as a generic log |
| Fixture | Source-controlled benchmark/test input | Never |

`python log_policy.py --policies` performs a bounded, read-only inventory. It
does not enter `AI_Children/`, delete files, compact JSONL, or infer that a log
is memory. The report's `over_size_policy` field identifies candidates for an
explicit maintenance action.

Current routing examples:

- `logs/ina_status.log` and `logs/comms_core.jsonl` are operational.
- `precision_window.log`, crash dumps, and module debug files are diagnostic.
- incident, migration, delivery-history, and authentication-health streams are audit evidence.
- `benchmark_results/*.jsonl` is benchmark history; `benchmarks/*.jsonl` is a fixture.
- reflection, emotion, language-evidence, precision-memory, queues, and neural-selector streams are memory-adjacent even when their names end in `log.jsonl`.

Promotion from evidence into memory must be explicit and structured. Once the
owning subsystem retains the durable fact, its source log remains subject to
the category's expiry policy; the raw log is not permanent merely because it
was once useful.
