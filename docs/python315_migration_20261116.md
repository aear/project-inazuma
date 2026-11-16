# Python 3.15 main runtime, supported Discord fallback — 2026-11-16

## Result and boundary

The selected main-runtime environment is `.runtime/core315`, based on Fedora's `/usr/bin/python3.15` (currently 3.15.2). The exact minor-version executable receives Fedora patch updates without following a future Python 3.16 alias. The retained `venv` runs Python 3.14.10 for Discord. This is a partial migration: Pycord 2.8.1 explicitly requires Python >=3.10,<3.15 and the unmodified full dependency resolution fails on 3.15. That restriction has not been overridden.

Pycord metadata: https://pypi.org/project/py-cord/ . The retained Python 3.14.10 itself includes recent security fixes: https://www.python.org/downloads/release/python-31410/ . Its private installation still requires explicit future updates; upgrading the main runtime does not automatically maintain that fallback or every pip package.

## Startup and installation

`runtime_environment.project_python()` selects `.runtime/core315` only when its local `.ina-ready` receipt exists. Discord explicitly selects the retained `venv`. Missing selected interpreters fail visibly rather than silently reverting versions. The supervisor records the actual selected command for each service. Main GUI probes no longer require the separately hosted Discord client; Discord retains its own dependency gate and finite crash budget.

Some existing startup paths use plain `python`. The GUI and supervised services now prepend the selected environment's executable directory to inherited PATH, so those children stay in the intended environment. An explicit-version child launch and inherited-PATH behavior are tested. Other environment entries are preserved, without mutating the input mapping.

`requirements-core.txt` contains the main-runtime dependencies, including audioop-lts required by pydub. `requirements-discord.txt` contains Pycord and its constrained DAVE dependency. `requirements.txt` includes both for a complete Python 3.14 installation. On Python 3.15 install only the core file until upstream Pycord supports it. The existing complete declaration remains deliberately unsatisfiable on unsupported Python rather than silently omitting Discord.

The new environment was built separately, using the candidate's resolved package versions; its pip was updated to 26.3. The old environment, its packages and private base interpreter remain intact. `.runtime` is ignored by Git. The local selection receipt is published only after the recorded checks. To reproduce elsewhere: create `.runtime/core315` with `python3.15 -m venv`, install `requirements-core.txt`, retain a supported complete `venv` for Discord, run the checks, then explicitly publish the ready marker. Do not copy Python 3.14 site-packages into a 3.15 environment.

## Evidence

The first unsplit 3.15 fixture run passed 103 checks and failed four because those cases import Discord. These failures were retained, not treated as missing evidence of success. The original 3.14 comparison passed all 107 checks. After separating the environments and adding coverage, 127 core tests passed on the final installed 3.15 candidate, with zero failures/skips. The four Discord-dependent language-route cases separately passed on 3.14.10: 131 focused checks across their intended environments, not a full suite on either interpreter.

Core scope includes English expression/character tokens, ML/cognition, lifecycle/startup, synthetic image counting, audio-device resolution and world collision. The GUI's actual pre-window import prefix completed on 3.15. Both installed environments passed pip dependency checks and their real startup dependency probes. Core imports do not establish that every optional hardware/desktop path works.

Registry comparisons: `runtime_startup` pinned V1 5/12, current V2 12/12; `python_runtime_split` pinned shared-interpreter V1 5/10, current V2 10/10. Selection fixtures are not live application startups. The split baseline materializes the committed service-command expression at 6ddc708052af29c993925a935ed4f2dda2a3e49b, modelling the process interpreter as the retained legacy environment. The startup suite explicitly labels its GUI wiring check separately from behavior.

A separate three-phase cross-version check used real runtime_state queue and outbox_event_store APIs: 3.15 wrote synthetic Unicode JSON/SQLite records, 3.14 read them and wrote a reply, then 3.15 read the reply and both event rows. All phases passed. No live memory data was used or migrated.

Tests ran in bounded source-only Bubblewrap rooms with network disabled, no live memory/config/credentials mounted, and numerical pools limited to one worker. A host process check recorded no running Ina project processes before broader fixtures. No full live boot, Discord login, voice handshake, actual audio capture, desktop rendering, background-interference benchmark or security audit was performed. These remain distinct from import and fixture evidence.

## Activation and rollback

Fully close and reopen Ina's GUI, then start normally. The main runtime and world/workspace services select Python 3.15; Discord selects Python 3.14. Source changes are uncommitted. To roll back the selection, stop Ina and rename `.runtime/core315/.ina-ready` to a disabled name. The GUI then selects the retained working 3.14 environment; keep both environments while any processes still use them. No learned storage or Ina-chosen memory-maintenance authority changed.
