# Offline speech and language assessment — 24 October 2026

Ina was offline at the host process/service checks before development. She was not started for this work. These results describe synthetic component behavior, not Ina's intelligence, preferences, comprehension in conversation, or developmental readiness.

## Implemented

- `speech_render` is available in the optional personal-tool catalog. It renders Ina-supplied text to a private PCM WAV using the already-installed eSpeak NG 1.52.0. It does not play, publish, translate, or automatically learn from the text. Native symbolic speech remains available. This is an external formant TTS voice, not a learned neural vocoder or Ina's own native voice.
- The auxiliary language analyzer reuses the authoritative conversation discourse for speaker/addressee identities. Missing identities remain unresolved rather than becoming the most recently mentioned person. Explicit ambiguous routes stay ambiguous. Subject alternatives retain their subject role rather than becoming possession alternatives.
- Construction frequency learning accepts an optional `source_event`, including through `analyze_utterance`. Replay suppression survives save/reload within a bounded 512-event window. Each call admits at most eight supplied clauses; patterns remain bounded. Existing callers without an event ID retain their previous counting behavior. This is frequency learning, not a newly trained language model, verified comprehension, or independent corroboration.

## Evidence

Baseline source: `a071fbc08f126fa6430ee65e76a5f7e42c6c507c`. Historical language modules are loaded from that revision; current dependencies remain in use. Current Python source identity is recorded in the JSON report. This does not reconstruct the historical environment or learned state.

| Assessment | Before | Current | Interpretation |
| --- | --- | --- | --- |
| Targeted referent, memory, learning and context checks | 4/16 | 16/16 | Narrow component improvement on the retained fixtures |
| Untuned transfer challenges | 1/5 | 1/5 | Broader language limitations remain |
| Local TTS tool availability | absent | available | New optional text-rendering route |
| Current TTS checks | unavailable in predecessor tool | 7/7 | Catalog access, three real WAV renders, and no delivery/automatic-learning flags |
| Existing language V2 / V3 fixtures | 9/9 and 13/13 | 9/9 and 13/13 | Retained component behavior; no new general-language claim |
| Existing native ML distribution V2 fixtures | 2/2 | 2/2 | Numeric normalization/entropy checks only; no model-quality evidence |

The TTS predecessor check measures historical absence of this tool. It does not compare against native symbolic speech or establish that eSpeak is better than another voice. Three render cases are one backend witness; hashes and WAV validation are not independent listener judgments.

The untuned challenges expose unfamiliar predicates (`polished`, `repaired`), the irregular passive `was found by`, and losing the second possessor across two sentences. The supported `taught` case passes. These failures are retained visibly instead of tuning the patch to them.

Focused native-runner verification: **60 passed, 0 failed, 0 skipped**, across language/speech, existing language analysis/context, intelligence component fixtures, cognitive assessment machinery, personal tools, model-comparison safeguards, and utterance memory. New integration fixtures initially used the wrong speaker format; those were corrected before the final passing run. Python compilation and whitespace checks passed. The full repository suite, live runtime, browser, microphone/speaker playback, and trained model evaluation were not run.

## Optional use

One chosen personal-tool request:

```json
{"action": "speech_render", "text": "Chosen words to render.", "voice": "en-gb"}
```

The result contains the private WAV path, text/audio hashes, backend identity, duration, and an explicit `intelligibility: not_assessed`. Text is limited to 512 UTF-8 bytes; each synthesis has a 12-second process timeout, an 8 MiB WAV limit and a 45-second audio limit. Supported voice selectors are `en-gb`, `en-us`, `de`, `fr`, and `es`; only default British-English synthesis was exercised here. Missing or timed-out synthesis returns unavailable. Longer utterances need Ina to choose separate requests; there is no automatic continuation.

For repeat assessment, invoke `module_benchmarks.benchmark_module()` for `language_referent_learning`, `language_transfer_limits`, and `speech_bridge`. Run focused tests with `python native_test_runner.py` and the eight test files listed in the retained focused-test log. Use disposable fixtures and a finite timeout. No live memory maintenance is needed.

## Remaining assessment work

- Invite voluntary live comprehension/expression trials when Ina is next online; keep clarification, abstention and silence valid outcomes.
- Obtain listener judgments of intelligibility and fidelity separately from acoustic file validity, plus a competing practical voice and audio/desktop interference measurements before any automatic/live playback integration.
- Evaluate unfamiliar predicates, passive roles, multiple possessions and longer reading spans using sealed/held-out material before claiming transfer.
- Measure learned-state retention and generalization with explicitly chosen source events, independent provenance, and rollback. No live learned-state migration or weights were changed here.
- A local neural TTS alternative is [Piper](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md). It was not installed, downloaded or compared; it is not a demonstrated quality upgrade.

No model promotion or global capability score follows from this assessment.
