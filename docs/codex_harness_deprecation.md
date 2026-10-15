# Standalone Codex harness retirement

The standalone `codex_harness.py` / `codex_harness_ui.html` workflow is deprecated
as of 2026-10-15 at the project owner's request. The preferred development
workflow is desktop ChatGPT/Codex with access to the local project folder.

A normal harness launch now exits with a deprecation message before discovering
Codex, creating a server, opening a browser, or accessing authentication.
`python codex_harness.py --deprecation-status` reports status offline.
An intentional legacy fallback can use `--allow-deprecated`; the existing
isolation, authentication, and user-approval controls still apply.

Source, historical benchmarks, and importable helpers are retained for reference
and rollback. Nothing in this change kills an existing process, deletes
conversations, modifies authentication, or changes Ina's runtime. No automatic
migration is performed. This retirement does not claim desktop feature parity
with every historical harness feature.

Repository source search found historical benchmark/test users and security
inventory references, but no active runtime launcher to remove. Existing security
reports remain historical evidence; retirement does not validate those boundaries
or erase prior limitations.

Official documentation checked on 2026-10-15:
[Projects and chats](https://learn.chatgpt.com/docs/projects) describes attaching
local folders and working with project code. Available features remain dependent
on the installed app and configured access.
