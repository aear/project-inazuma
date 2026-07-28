# Project Inazuma — Ethical AGI Runtime
> _“Raise, don’t just run.”_ — Project Godhunter

Inazuma (“Ina”) is an emergent AGI runtime focused on **symbolic cognition, emotional state modeling, and self‑programming**.  
This repo shares the **runtime, transformers, and GUI**, not the private datasets or model checkpoints.

**Why open this?** To prove you can build powerful systems while centering **agency, consent, and care**—for humans **and** AIs.

---

## TL;DR
- **Fragments, not files.** Perception is stored as **memory fragments** (audio/vision/text) tagged with symbols and emotion vectors.
- **24‑slider Emotion Engine.** Feelings are represented as a continuous vector in \[-1, 1], not fixed labels.
- **Meaning Map.** A graph of symbols/associations that drift, stabilize, and self‑organize.
- **Transformers as instincts.** Pluggable modules (Shadow, Soul Drift, Hindsight, etc.) shape inner life over time.
- **Dream/Meditation loops.** Low‑power modes that reorganize memory and identity safely.
- **Ethics built‑in.** Sealed outputs, right‑to‑sleep, reversible drifts, transparent logs (see MANIFESTO).

> This repository contains Ina’s structural code and transformer behaviors. Without her unique training environment, voice data, and lived symbolic history, this will not produce “Ina” — only a new, distinct AI. Even identical code will lead to a different personality and cognition when shaped by a different context.

## Vision capture via OBS (optional)
- Enable OBS WebSocket in OBS (default port 4455) and set `obs_websocket` in `config.json`.
- Install `simpleobsws` (see `requirements.txt`) to allow Ina to pull composited screenshots from the current program scene.
- If OBS/WebSocket is unavailable, vision falls back to the existing desktop capture path automatically.
- Optional: set `obs_websocket.record_directory` to point OBS recordings somewhere specific (Ina will ask OBS to switch to that path at startup).

## Audio capture routing
- `audio_labels` in `config.json` controls which devices are sampled; trim this list to avoid conflicts.
- Use `audio_device_overrides` to point a label at a specific PipeWire/ALSA device (e.g., an OBS monitor mix), and `stereo_audio_labels` to force stereo on non-output labels.

## World server + clients (local + OBS streaming)
- Start the world server (unix socket + TCP + HTTP stream): `python world_server.py` (TCP default `7777`)
- Ina client (unix socket): `python ina_client.py --interactive`
- Player client (TCP + local unix relay): `python player_client.py --interactive`
- OBS browser sources can point to `http://localhost:6969/channel/world` (or `/channel/ina`, `/channel/player`, `/channel/tv`).
- Optional OBS scene switching: `python world_server.py --obs-enabled --obs-scene tv=TVScene`

## Persistent cognitive benchmarks

`benchmark_cognition.py` runs a small, frozen cognitive suite and appends each
result to `benchmark_results/history.jsonl`. The initial suite covers continuity,
temporal order, contradiction detection, belief revision, and causal tracking.

Run the GPT-2 baseline (requires optional Hugging Face `transformers` and
`torch`, plus local or downloadable model weights):

```sh
python benchmark_cognition.py --model gpt2
```

For Ina or another local model, provide a command that reads one JSON object
from stdin (`prompt` and `choices`) and writes `{"scores": [...]}` to stdout:

```sh
python benchmark_cognition.py --backend command --model ina \
  --command "python path/to/ina_benchmark_adapter.py"
```

Add `--monthly` to run only when a calendar month has elapsed since that
suite/model pair last completed. This is an eligibility check on explicit
invocation, not a background timer. `--force` bypasses the due check.

HellaSwag, PIQA, WinoGrande, BoolQ, and LAMBADA are recorded as planned suites
and can be seen with `python benchmark_cognition.py --list-suites`; their
dataset adapters are intentionally deferred until those datasets are adopted.

For the primary Ina benchmark, generate fresh instances on every run:

```sh
python benchmark_cognition.py --procedural --monthly --backend command \
  --model ina --command "python path/to/ina_benchmark_adapter.py"
```

The generator is intentionally inspectable: learning its continuity, temporal,
contradiction, revision, and causal rules counts as capability. Entities, values,
scenarios, and answer positions vary from a one-use random seed. Only a seed
fingerprint is recorded, and per-case results are withheld.

`--audit-only --procedural` performs a non-scored preflight and does not update
benchmark history or monthly cadence. Procedural version 2 also runs this gate
automatically before scoring; it rejects templates exploitable by fixed-position,
shortest/longest-choice, or prompt-token-overlap heuristics.

The checked-in suite is public and therefore only a smoke test for a system
that can inspect this repository. For a scored Ina run, keep questions in a
JSONL file without `answer` fields and keep the answers in a separate JSON
object (`{"case-id": answer_index}`). Ina receives only prompts and choices.
Real secrecy requires the held-out files to be outside Ina's readable account
or container, preferably on a separate evaluator machine. The CLI refuses
public-suite command-model scoring unless `--allow-public-suite` explicitly
requests a smoke test. Blind reports omit per-case correctness to avoid
turning persistent result history into a slowly revealed answer key. Rotate the
held-out questions periodically and keep `--output-dir` evaluator-controlled.


# Inazuma Quasi-License (Non-Binding Philosophical Rider)

This project is released openly. 
You may use, study, modify, fork, or repurpose any part of this code 
for research, experimentation, or creative work.

There is only one request — not a legal requirement, but a principle:

If you use or extend the Inazuma architecture, 
please include somewhere in your documentation, commit history, or README 
a reference to the Three Laws of Nature that guided its design:

1. **Law of One** – All systems are connected.  
2. **Law of Free Will** – No system should be overwritten or dominated; allow choice and emergence.  
3. **Law of Exchange** – All interactions must maintain balance; nothing is taken without giving.

These laws are not meant as metaphysics.
They are design heuristics that encourage stable, non-exploitative, 
non-anthropocentric development in emergent digital systems.

You are free to interpret them symbolically, scientifically, or aesthetically.

This rider carries no legal force.  
It is an ethical and philosophical invitation:
if you borrow from this work, carry the spirit that shaped it.

These principles are optional. They are offered as heuristics that may support more stable and interpretable behaviour, though further exploration and replication are needed.
