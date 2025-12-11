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
