# Optional English: report-path review and choice fixes

The inspected report paths (`self_read_reporting`, `storage_migration_report`,
`github_submission.build_issue_body`) combine structured findings, caller-supplied
body text and fixed English templates. Their fluency does not establish a separate
general English generator. No live private reports were scanned for this review.
Evidence, uncertainty and source wording remain useful reusable structure, but a
general social-composition improvement remains unimplemented in this pass.

Two concrete choice bugs were repaired:

- Explicit `native_only` no longer aliases the historical mixed `native` mode.
  Legacy `native`, `auto` and `mixed` behavior remains for compatibility.
- Discord rendering now respects the selector's assessed native-only, clarify or
  abstain result instead of discarding it and reconstructing English layers.
  Rendering-error fallback also declines English for an explicitly native-only
  or abstaining preference. Silence is not replaced with canned wording.

Eight focused checks passed with synthetic dependencies and copied source. The
Discord functions were extracted from their actual source; full module/live-route
verification remains unavailable following the import timeout documented in
`english_expression_review_20261005.md`. A versioned selector comparison uses the
original source from `aa63c000876915da61216dc15e3d55f26b980e05`: native-only changes
from mixed output to native output, while English and mixed choices remain usable.
No runtime restart, live send or memory edits occurred.
