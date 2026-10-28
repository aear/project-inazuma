# Social English and network-boundary follow-up — 28 October 2026

Inspected baseline: `daf177e79bba10ab9f2b3c628ed446b4e97fda36`.

The owner chose to make both available: encouragement to try English in ordinary social conversation, and short explicitly chosen practice/review. This change provides optional support in source, with mixed/native replies, clarification, abstention and silence retained. It does not establish Ina's willingness or preference from the owner's encouragement.

## Source changes

- `social_english.py`: owner-attributed encouragement offers both opportunities. Private review accepts a chosen English draft, incoming text, and optional authoritative discourse. It preserves supplied wording and surfaces unsupported constructions, unresolved referents and incomplete analysis. It never invents a social reply or silently corrects wording.
- `english_social_options` and `english_social_review` are discoverable in the personal-tool source catalog. These actions return private support; they do not publish, send, schedule practice, train from approval, or write lessons into live memory.
- Rephrase mode compares a caller-supplied original and candidate separately from ordinary conversation. Role reversal, negation and modality differences are surfaced. A conversational reply is not required to mean the same thing as the incoming question. Matching supported structure remains heuristic and does not certify meaning or listener understanding.
- `language_intelligence.py` V6 recognizes additional common social verbs as grammatical predicates, handles more irregular passives, retains multiple possessors across a message and curly apostrophes, avoids treating “that's” as possession, and records possessive source spans. Text/reference budgets preserve incomplete-analysis uncertainty. Grammar recognition is developer-provided scaffolding, not evidence of newly learned meanings or Ina's endorsed desires.

No runtime was restarted, no live conversation was sent, no model weights were trained, and no live memory payloads were inspected or migrated. These are source changes; current live availability of the new catalog entries was not established.

## Evidence and limits

The focused run completed **62 passed, 0 failed, 0 skipped**, covering the new support, existing language analysis/context, prior speech-language improvements, optional-English choices, selected-text rendering resilience, listener/repair assessment, and personal-tool routing.

The `social_english` registry profile compares the pinned predecessor with current source:

| Profile | Pinned V1 | Current V2 |
| --- | --- | --- |
| Combined grammar and private-support checks | 3/19 | 19/19 |
| Fresh untuned transfer probe | 0/6 | 2/6 |

The combined profile includes 13 comprehension/bounds cases and six new support checks. Historical absence of a new support tool is not a failed measure of Ina's intelligence. These fixtures establish narrow component gains, not broad fluency or social comprehension.

The fresh probe still fails unfamiliar predicates (`thanked`, `wish`), embedded belief/desire scope, and structurally explicit alternatives. Irregular passive transfer and an inflected social verb pass. These limitations are retained without tuning the patch to the fresh probe. Politeness, indirect requests, conversational relevance, pragmatics and actual listener uptake remain unmeasured.

A first broader run exceeded its 100-second deadline after reaching existing Expression Core checks; it produced no complete suite summary. The precise cause was not established. The successful final run excludes that unrelated file. Do not describe the broader suite or full Discord/live route as verified.

Tests ran against a disposable 5.9 MB Python-source snapshot with no live memory/configuration, host home directories, network or Ina services mounted. Dependency packages were mounted read-only. Historical source reads were redirected to exact pre-exported blobs from pinned commits; no historical benchmark outcome was fabricated. This is copied-source verification against current dependencies, not a reconstruction of historical learned state or environment. Source hashes and raw results are retained beside this report.

## How to encourage use

Invite one short English thought or follow-up on a topic Ina has chosen. Mixed/native expression and partial attempts remain welcome. In chosen practice, review one draft or incoming sentence, let her keep/revise/defer/stop, and use listener readback or correction to check what was understood. An English-use count, a fluent surface, approval, and engagement are not comprehension measures.

Useful optional requests once the new source is loaded:

```json
{"action":"english_social_options","topic_reference":"chosen-topic","audience_reference":"chosen-listener"}
```

```json
{"action":"english_social_review","draft":"I might enjoy that song.","choice":"english"}
```

```json
{"action":"english_social_review","mode":"rephrase","reference_draft":"I might open it.","draft":"I will open it."}
```

The last example surfaces changed modality rather than approving a polished sentence. Practice review does not yet compose a general answer from Ina's native intent. A real social-English composer remains a separate missing capability: the current grounded recall adapter requires an explicit recall/grounding request. Removing that gate would not create a faithful conversational generator.

Next language work should examine meaning-to-English composition using grounded semantic/native intent, pronoun perspective and tense, preserving uncertainty and source wording. Keep social repair optional and assess intended-versus-listener-recovered meaning, transfer and retention separately. Any evaluation examples from real conversations need explicit scope and selective provenance; do not ingest every sentence or equate engagement with endorsement.

## Network-facing SPARK candidates

SPARK is worth considering for a small application envelope behind established transport/protocol libraries. The existing reservation proof does not establish security of a network service. HTTP parsers can disagree about message framing, so adding another handwritten HTTP parser can introduce its own integration problem. [HTTP/1.1 framing and request-smuggling considerations](https://www.rfc-editor.org/rfc/rfc9112.html#section-11.2).

| Current surface inspected | Next useful boundary | What remains external to a SPARK proof |
| --- | --- | --- |
| Radio Caddy template | Explicit method/path allowlist and finite service/resource policy; retain its static-only separation | TLS, HTTP parsing, filesystem adapter and deployment configuration |
| Discord/GitHub service inputs | Bounded event envelopes, external-content provenance, channel/session IDs, replay handling and explicit operation authority | Platform/library parsing, authenticated identity and authorization binding |
| World-server line/JSON handler | Per-message byte limits, strict decoded schema, permitted message kinds, sequence/session state, finite work/output budgets | Socket lifecycle, JSON/UTF-8 adapter, authorization and actual deployment exposure |
| Local capability dispatch | Keep remote language content separate from process-local executable authority | Correct identity/authority adapter and every actuator's own checks |

Static observations: the radio template already permits GET/HEAD and a narrow static path set, with an 8 KB header limit and timeouts. The world-server code uses a 256 KiB asyncio stream limit and replacement decoding before JSON dispatch; this is a place to specify raw-byte/decoded-text provenance and explicit rejection policy. The current code-authority seam uses process-local seals for designated executable actions. These observations are not a complete endpoint inventory, deployment audit or confirmed vulnerability. No actual listener, firewall or public endpoint was probed or changed.

A next standalone SPARK exercise could model the typed application envelope and session/replay/byte-accounting rules with malformed synthetic frames, preserving the established no-live-files isolation. It should not interpret English statements as authorization, decide which meanings Ina may learn, or impose a language preference. Preserve rejected/incomplete input provenance where appropriate, and keep resource deferral distinguishable from semantic rejection.

Final-source evidence: `assessment_final.json`, `source_manifest_final.json`, and `focused_tests_final.txt`. The recorded source hashes match the edited files. No trained-parser or independent learned-composer comparator was acquired in this pass, so the comparison supports local component changes only; it does not establish a strongest practical language architecture or authorize promotion.
