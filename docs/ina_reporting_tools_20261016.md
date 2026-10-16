# Ina's chosen work reports and bug diagnostics

Implemented 2026-10-16. The Fedora update is already complete. No restart, service start, memory maintenance, publication, commit or push is performed by this work.

## Available personal tools

The normal private-tool catalog now advertises these commands. They use the existing queue, one-command processing, process-local voluntary authority and result publication. A serialized `source: ina_voluntary_choice` label alone cannot authorize them. Nothing asks Ina to explain her creative motives or report on a schedule; native wording is accepted unchanged.

| Command | Chosen input | Result |
| --- | --- | --- |
| `work_report_write` | `report` mapping | A new private report under the configured personal drive's `Reports` folder |
| `work_report_read` | `report_id` returned by write | One bounded report, including its supplied references and measurements |
| `bug_source_inspect` | `code`, optional `source_reference` | Static Python candidates with source hash and line numbers; no import or execution |
| `bug_file_inspect` | `relative_path` on the existing read-only code drive | Same diagnostics for one explicit non-private `.py` file; no directory scan |

Reports support `kind` = `drawing`, `music`, `experiment`, `capability`, or `bug`. Required fields are `title`, at least one `observations` entry and an explicit `uncertainties` list (which may be empty). Optional fields:

- `interpretations`, `next_steps`, `reproduction_steps`: lists of chosen strings.
- `expected`, `actual`: chosen text.
- `evidence`: up to 16 entries with `reference`, optional `sha256`, `origin` and `independence_group`.
- `measurements`: up to 16 entries with `name`, `value`, `unit`, `method`, `evidence_reference`. Values are text so units, native notation and `unavailable` can be retained. Name and method are required.
- `previous_report`: reference to a prior report. A revision receives a new ID/file; it does not overwrite that report or the underlying creative asset.

An example command payload (the trusted voluntary caller supplies authority separately):

```json
{
  "action": "work_report_write",
  "report": {
    "kind": "drawing",
    "title": "Chosen drawing trial",
    "observations": ["My description, in whichever wording I choose."],
    "uncertainties": ["I have not tested whether another viewer sees the same thing."],
    "evidence": [{"reference": "chosen-artifact-version-reference", "origin": "my drawing trial"}],
    "measurements": [{"name": "viewer interpretation", "value": "unavailable", "method": "not tested"}]
  }
}
```

Drawing and music reports can reference existing creative-version records. Experiments can reference the experiment ID, run result and Experience Cycle. Capability reports can retain separate task measurements, failures, missing checks and contrary evidence from existing assessments. Bug reports have room for source/version references, reproduction steps, expected/actual behavior and proposed next checks. No global capability score, automatic promotion or independently verified evidence is manufactured.

## Bug investigation workflow

1. Choose a source excerpt or one code file. The checker accepts at most 64 KiB, traverses at most 4,096 AST nodes and retains at most 32 findings. Incomplete traversal is disclosed. File reading uses a narrower budget through the existing virtual filesystem; oversized files are rejected before parsing.
2. Review syntax rejections, bare `except` clauses, mutable literal defaults and repeated direct-scope definitions. Repeated names in different classes are not conflated. These are candidates: intentional overloads, shared defaults and recovery paths can be legitimate. No findings does not establish correctness or security.
3. If desired, use existing `experiment_create`, `experiment_run`, `experiment_judge` for a bounded reproduction in the experiment room. This is an explicit choice, not something a source finding executes. Production source edits and promotion are not granted by a report.
4. Save observed outcomes and remaining uncertainty in a bug report, linking later revisions through `previous_report`. Existing GitHub reporting remains separate; these private reports are not submitted externally.

The checker cannot inspect non-Python behavior, prove semantic intent, check runtime state or establish bug-detection recall. Source references and report evidence remain caller-supplied. Readback checks report identity, not cryptographic authenticity; the personal drive remains editable. No raw memory directories, private project folders, environments or symlink paths are accepted by the file checker.

## Verification and implementation

- `ina_work_reports.py`: validation, chosen report creation/readback and bounded static inspection.
- `personal_tool_runtime.py` and `instruction_authority.py`: real command dispatch and authority checks before filesystem setup.
- `ina_desktop/files.py`: backward-compatible `read(..., max_bytes=...)`, including a bounded read if a file grows after the size check.
- `benchmarks/benchmark_work_reports.py` and `module_benchmarks.py`: explicit V1 generic-note baseline pinned to `a31f1323ee4bb589d0ca0bf02aea87201b69cd21`, V2 report/tool candidate. Historical runtime source uses current dependencies; this is not a reconstructed historical environment.
- `tests/test_ina_work_reports.py`: connected tool roundtrips, native text, revisions, unknown measurements, source limits, malformed syntax, scope distinctions, symlinks, private paths and authority rejection.

V1 retains ordinary native notes. V2 passes 14/14 selected reporting/control checks, including all five report kinds and real dispatcher calls. This measures reachability and preservation, not report quality or general bug-finding ability. Single-run fixture timing is retained descriptively; no speed claim is made. Focused native-runner verification: 9 passed. Full isolated pytest regression suite: 1,081 passed, 2 skipped, 8 warnings. Skips remain the optional retired-harness editor workspace and optional unbuilt native vector kernel.

An initial fixture failure exposed a field collision between source code and instruction provenance. Source code now travels under `code`; `source` remains authority provenance. The connected malformed-source fixture verifies the actual code is inspected.

Testing used disposable fixture storage in a network-isolated namespace with no live-memory mounts. A private host process check found no scoped Ina runtime before adversarial tests. No tests of these new tools were run against Ina's live private work. Live report usefulness, false-positive rates on her projects, actual creative quality and runtime bug reproduction remain unmeasured.

Reproduce under the repository's shutdown/isolation policy:

```console
python benchmarks/benchmark_work_reports.py
python native_test_runner.py tests/test_ina_work_reports.py
```

Rollback removes the new commands/authority entries, module, benchmark registrations and tests together; existing notes, assets, experiment records and prior reports can remain. No live memory schema changed.
