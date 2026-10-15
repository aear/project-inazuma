# Review implementation and verification, 2026-10-15

Author: Codex. Baseline: `e13c6b4517f4df9859148a5f96cca0b4814bb9d0`.
This records the implemented portion of the efficiency/capability review and the
separately requested retirement of the standalone harness. It is not evidence
that every review recommendation is complete.

## Implemented

- `runtime_state.py`: uncached detached snapshots and atomic multi-key replacement
  under the existing lock. Single-key updates reuse the same operation. Queue
  claims and read/modify/write counters retain their dedicated operations.
- `model_manager.py::monitor_hunger()`: one coherent read for its direct state
  lookups and one grouped write for hunger, fitness, and update time. The actual
  caller body was compared against the pinned historical body with fixed inputs
  and stubbed external helpers; results matched. Direct state I/O changed from
  six reads/three writes to one read/one write. This excludes helper-internal I/O.
- `memory_index.py`: a composite recency index for new sidecars, schema V3, and an
  explicit additive upgrade for existing V2 sidecars. Existing V2 databases remain
  accepted without migration. `upgrade_recent_index=True` is a maintenance opt-in;
  no live sidecar has been migrated. Existing recall touches survive migration,
  and the source JSON is not reparsed for this upgrade.
- `language_intelligence.py`: analysis V3 preserves quantities and unsupported
  surface symbols, handles basic auxiliary/main-verb distinctions and transfer
  alternation, recognises basic run inflections, and reports lower uncertainty
  confidence for unsupported analysis. Supported heuristic analysis is explicitly
  described as not empirically calibrated. This remains a limited English analyzer.
- `learned_media_lessons.py`: optional `selection="aligned"` and bounded alignment
  keys provide ranked, alternating domain coverage. Default selection remains
  compatible; the candidate is not wired into live callers. Sources are unchanged.
- `module_benchmarks.py`: semantic-topology V3 scores observed word-sense outcomes
  only. Historical V1/V2 mixed behaviour/wiring results remain available. The
  English foundation assessment now uses V3; this is a narrower evaluation, not
  a newly demonstrated increase in language capability.
- `historical_source.py` and benchmark results: code identity records Git HEAD and
  a SHA-256 of tracked Python changes plus untracked Python source. It does not
  identify private learned state, configuration, installed dependencies, or an
  entire historical environment. Missing provenance is reported as unavailable.
- Standalone harness: default launch exits before configuration, authentication,
  browser, or server setup. `--deprecation-status` works offline;
  `--allow-deprecated` retains an explicit historical fallback. Source, helpers,
  security controls, historical reports, and benchmarks remain. `AGENTS.md` and
  README no longer require feature parity with the retired harness. See
  [retirement details](codex_harness_deprecation.md).

## Initial verification

All execution used a disposable source copy with temporary fixture data. The
status-pipe writer was disabled in test launchers so tests could not publish to
Ina's live status channel. No live memory was used or changed.

- System interpreter: 55 focused tests passed, zero failed/skipped, across new
  regression tests and existing memory-index, language, state-queue, lesson,
  intelligence-upgrade, language-context, and recall-index tests.
- Project virtual environment: 31 focused tests passed, zero failed/skipped,
  covering the new regressions plus memory-index, language-intelligence and
  state-queue files.
- An earlier combined virtual-environment run passed 27 tests before timing out
  during collection of a manager-importing test. A separate bounded stack capture
  located the stall in optional Qiskit's native import through `QTransformer`.
  System-interpreter coverage passed without that optional dependency. No Qiskit
  workaround or production isolation change was introduced.
- `review_efficiency` V1/V2 uses the same disposable fixtures against pinned source
  and the candidate. Historical source uses current dependencies, not a restored
  historical environment. Raw samples and environment details are retained in
  [the comparison evidence](evidence/review_efficiency_20261015.json).
- Candidate passed 22/22 targeted checks; historical code passed 7/22. These include
  changed policy and evaluator requirements, not just capabilities. This total is
  not a general intelligence, readiness, or efficiency score.
- On 10,000 synthetic disk-backed rows in temporary storage, median seven-query
  lookup time was about 1.47 ms historical versus 0.025 ms candidate. Database size
  grew from 790,528 to 1,122,304 bytes (about 42%). This is a narrow query sample,
  not an alternating paired trial, a real-store benchmark, or system-wide saving.
- Changed Python compiled and diff whitespace checks passed. The report and source
  references were inspected. No full-suite, GUI, real audio/frame interference,
  live-runtime, HDD/NVMe comparative, adversarial, or security tests were run.

Reproduce the standalone comparison explicitly in isolated temporary storage:

```sh
python -m benchmarks.benchmark_review_efficiency
```

The same V1/V2 comparison is registered as `review_efficiency` in
`module_benchmarks.benchmark_module()`. It does not run periodically.

## Activation and remaining work

No machine reboot is required. A controlled restart of affected Ina processes is
recommended before relying on the new source: existing Python processes retain
already imported code, while newly spawned helpers can import edited files sooner.
No process was stopped, restarted, or signalled by this change.

Existing-sidecar index migration should be evaluated and scheduled in a controlled
maintenance window: index construction can block writers. The new guidance selector
remains opt-in until output/transfer evidence justifies live use.

Live music, drawing, speech, code-generation quality, calibration, and cross-domain
transfer remain unmeasured. No evidence supports automatically changing creative
choices or promoting those domains. The drawing cover-to-burst heuristic remains
pending output evidence. Paper trading still requires locating its implementation.

Security outcome evaluators and adversarial fixtures remain pending. The repository
requires the complete Ina runtime tree to be stopped and independently verified
before such testing, with a private verification log and isolated fixtures unable
to access live memory. A controlled shutdown is required before that work; it was
not attempted here.

Changes are local and uncommitted. Nothing was pushed or published. Rollback is a
source diff review/reversal; existing V2 indexes and original guidance defaults
remain usable, and no live data migration needs undoing.


## Maintenance-window completion

The owner verified shutdown and authorized an isolated maintenance run, then
reserved all live memory/index migrations for Ina's own choice. No live index,
fragment store, or memory location was changed by Codex.

### Checks completed

- Host `/proc` and user services were checked before security testing. No Ina
  runtime remained. The timestamped pre-test record is private outside Git.
- Bubblewrap exposed system libraries, installed dependencies read-only, and a
  disposable source copy. It exposed no host home, mounted data drives, credentials,
  live memory, display/audio session, or external network. The live status pipe was
  disabled in launchers. Tests did not restart Ina.
- The final full isolated pytest run completed: **1,063 passed, 2 skipped**.
  Skips are the optional local editor workspace (offered to Ina) and an unbuilt
  optional native vector kernel. Eight warnings remain; passing tests do not
  establish live Discord, audio, rendering quality, or runtime safety beyond the
  exercised fixtures.
- Initial isolated attempts exposed missing dependencies, missing local-only
  files, stale benchmark directory counts/version unpacking, and one new migration
  helper naming collision. The collision was fixed and tested; the full rerun
  passed. Local vocabulary policy is no longer asserted as a public test fixture.
- Test-only numerical/thread limits allowed Qiskit import to complete. Production
  execution policy was not changed. The broad run uses the actual updated project
  dependency set, without the temporary dependency overlay.
- `py-cord` 2.8.1, `davey` 0.1.7 and PyNaCl 1.6.2 were installed in the project
  virtual environment from downloaded PyPI wheels after isolated compatibility
  testing. Dependency consistency checks passed before and after. A private
  affected-package backup was retained in temporary maintenance storage. Other
  installed dependencies were left at their already-compatible versions.
- The formerly ignored dependency declaration is now distributable, with an
  explicit DAVE dependency. Credential setup is documented in a safe tracked
  guide; existing private credential documentation/configuration was preserved.

### Ina can inspect and act

`memory_maintenance_tools.py` and the personal-tool catalog now provide:

- Read-only inspection of configured index locations, schema, size, and a plan identity.
- One disposable synthetic trial on temporary storage or a chosen configured
  device; it does not open that device's real index and removes scratch files.
- A chosen, transactional index upgrade or rollback, requiring a fresh inspection
  identity and process-local voluntary authority. Busy/unsupported databases,
  insufficient free space, stale plans, and time budgets defer execution.
- Private append-only requested/completed receipts containing before/after paths
  and schemas, retained selection verification, and rollback information.
- Explicit defer/decline with no automatic retry, schedule or continuation.

The `memory_maintenance_tools` V1/V2 benchmark demonstrates historical absence
versus six actual synthetic outcome checks. The new timeout path rolls back the
transaction. This benchmark does not claim Ina has chosen or performed a live
migration. Existing capability-scoped, verified file-movement tools are retained
and referenced in her handoff rather than reimplemented or invoked.

See [Ina's optional local checks and tool instructions](ina_optional_local_checks_20261015.md).
A clearly attributed external offer is delivered in her private Notes area, through
her existing personal-drive file interface. It is not inserted as an Ina-authored
self-question, inner thought, experience, or an executable command queue entry.
The tool catalog becomes discoverable through the existing runtime mechanism at
her next normal startup. No action has been preselected on her behalf.

Live memory migration, private configuration checks, voluntary creative-output
review and live capability measurements remain her choices. Public/synthetic
fixtures are complete for this change; these deferred choices are not failures.
The machine has not been rebooted, Fedora has not been updated, and changes remain
local/uncommitted with nothing pushed.
