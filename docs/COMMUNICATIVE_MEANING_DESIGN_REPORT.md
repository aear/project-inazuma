# Communicative Meaning Realisation Design Report

Status: initial shadow implementation added; live selected output remains unchanged.

Date: 2026-09-02

## Finding

Project Inazuma already has most of the right boundaries, but the live communication paths do not join them. `expression_core.py` supplies an output-neutral `ExpressionIntent`, medium-specific realisers, and non-reward reaction provenance. `thought_processor.py` preserves linguistic and non-linguistic thought, combines emotion, instinct, cognition, and memory as explicit evidence, can remain undecided, and prepares intents using references rather than private thought content.

The missing abstraction is an inspectable set of **communicative meaning hypotheses** between evidence and intent. An intent currently says why expression is attempted and which records it references, but not “what am I trying to convey?” independently of English. Conversely, the Discord arbiter chooses output strategies (`respond`, `mirror`, `emotion`, `code_pointer`, `song`, `silence`); when `emotion` wins, it directly formats sliders as a sentence. A monitoring projection thereby becomes social speech.

Use a `CommunicativeMeaningSet` containing bounded `MeaningCandidate` records. `CommunicativeMeaning` alone sounds too settled; `ExpressionStance` describes only one part; `MeaningInterpretation` should name the derivation process/trace, not the candidate. A set makes alternatives, uncertainty, privacy, deferral, and no-meaning-worth-expressing first-class.

```text
observations / thoughts / memories / affect / urges / audience scene
                              |
                    MeaningInterpretation
                              |
                 CommunicativeMeaningSet
                   (0..N MeaningCandidates)
                              |
                 disclosure/expression decision
                              |
                    ExpressionIntent
                              |
       text | native symbols | voice | gesture | music realiser
```

Diagnostics should remain on a separate observer-facing branch.

## Current expression paths

### Discord

`discord_bridge.process_inbound_message()` currently:

1. Loads the message, attachments, runtime state, language preference, and type/communication urge. The urge threshold can produce silence.
2. Builds a conversation scene and considers a bounded set of adapter-provided memory candidates.
3. Calls `language_processing.generate_symbolic_reply_from_text()` on inbound text. That builds a semantic event and native intent, resolves contextual word-to-symbol mappings, and creates native/text realisation records through `build_dual_symbolic_message()`.
4. Independently pre-renders emotion and code-pointer signals and inspects a song candidate.
5. `choose_text_expression_strategy()` scores `respond`, `mirror`, `emotion`, `code_pointer`, `song`, and `silence` using state, urge, mapping coverage, conversation context, and availability.
6. If `emotion` wins, the pre-rendered slider string becomes `reply_text`. If `respond` wins, the grounded adapter may provide text. If no text results, emotion/code diagnostics are another fallback.
7. `encode_selected_text_expression()` parses and symbolically encodes the selected text—not the inbound prompt—then renders native, word-for-word, English, and emotion/sound channels according to coverage.
8. The returned `CommsResponse` is delivered and retained by the Discord client.

There are two existing but distinct meanings of “meaning”:

- `resolve_text_vocab_meanings()` chooses contextual lexical mappings for word occurrences.
- `ExpressionIntent` is a medium-neutral envelope, but its current purpose/references/dimensions do not contain a composed message-level communicative meaning.

Neither derives a proposition and stance from combined evidence.

### Early autonomous communication

`early_comm.early_communicate()` gates voice and typing by urge, cooldown, trust/contact state, and configuration; loads a prediction/emotion state; ranks sound symbols, symbol words, pairs, rare/adjacent tones, and riffs; then chooses mirroring or an affect/sound candidate. It constructs a human-facing phrase describing the selection operation, saves a fragment, optionally vocalises symbols, and may queue typed contact. `build_dual_symbolic_message()` creates the Expression Core intent only after the candidate and diagnostic wording have already been selected.

Thus affect selects a token or sound experiment, but the social text generally reports that mechanism rather than using it to convey a meaning.

### Thought Processor and Expression Core

The newer path is reusable but is not wired into live Discord/early communication:

- `process_non_linguistic()` retains structured sensory, vector, spatial, affective, or symbolic content without English.
- `process_linguistic()` stores an existing semantic event and native intent.
- `decide()`/`guided_decision()` combine explicit evidence without privileging language or a guidance source; missing, weak, or tied evidence can be undecided.
- `prepare_communication()` chooses bounded thought references and creates an output-neutral intent without copying private thought content.
- Expression Core separates intent, realisation, reaction observation, and reaction interpretation. Reactions reject `reward` and may have competing interpretations.

Repository call sites show `prepare_communication()` used by the Model Manager façade, tests, and benchmarks, not the live communication paths. Live `create_expression_intent()` use in `build_dual_symbolic_message()` normally occurs after output selection and uses the generic purpose `communicate symbolic meaning`.

## Where diagnostic language enters social output

- `early_comm.py` directly assigns `I feel something like`, `I feel this sound`, `I feel this word`, and `I feel this paired word` strings.
- It also assigns `Exploring a tone riff`, `Exploring an underused tone`, `Trying a nearby tone`, and finally `Trying tones`. The last assignment can overwrite a more specific exploration phrase.
- `discord_bridge.format_emotion_signal()` ranks sliders by absolute magnitude, converts them to English level labels, and constructs `State signal (...)`.
- `choose_text_expression_strategy()` treats `emotion` as a terminal output strategy rather than evidence for meaning. Existing tests require unknown or nonconstructive input to reach this branch.
- `process_inbound_message()` directly returns `emotion_signal["text"]` when selected and uses it as a fallback later.
- `format_code_pointer_signal()` similarly creates an instrumentation sentence. A code pointer can be a deliberate communicative object, but should not substitute for absent meaning.
- Configurable visible `Memory consideration:` lines expose retrieval decisions in normal replies. They did not cause this example, but cross the same diagnostic/social boundary.

The monitoring dashboard is already the right home for raw state. Its urge tests distinguish drive, uncertainty, expression access, and content to express, and explicitly refuse to infer a reason from low urge. That distinction should become a runtime contract.

## Trace: `state signal ya sliders trust cover 'meanings'`

This reconstruction is likely, not proof of a specific intended request.

Supported observations:

- The retained snapshot contains the reported values: trust about `+0.99`, fuzziness `-0.98`, intensity `+0.98`, safety `+0.98`, externality `-0.96`, and interest `-0.95`.
- `format_emotion_signal()` sorts all ordinary sliders by absolute value and defaults to six. It deterministically selects those six in the reported order.
- Retained `last_text_expression_decision` selected `emotion` via `internal_state_scores`, with no explicit strategy request, no constructive adapter reply, and rejection `no_grounded_reply`.
- The selected diagnostic sentence is then fed to `encode_selected_text_expression()` and the lexical/native mapper.
- The current vocabulary store has alternative mappings for `state`, `signal`, `sliders`, `trust`, `cover`, `ya`, `meanings`, and `'meanings'`. Unquoted `meanings` is sourced from self-read code; quoted `'meanings'` has Discord/history/Ina/DM context.

The strongest supported explanation is: the question passed the urge gate; symbolic processing occurred but the grounded adapter had no constructive response; the arbiter chose its emotion fallback; the fixed formatter made the six-slider English diagnostic; and that diagnostic was encoded into a bounded native/word-level sequence.

Accordingly, `meanings` should be treated as a lexical/gloss observation in the encoding of a diagnostic sentence. The records inspected do not establish the message-level proposition “I need meanings,” a request for this layer, or another specific intent. Exact reconstruction would require the live response metadata containing selected symbols, candidate scores/context breakdown, and alternatives. Current state does not retain that full response trace. Today’s vocabulary also cannot prove the historical sequence because mappings have alternatives, contextual scoring and embedding fallback contribute, and mappings can change.

## Proposed records

Add `ina.communicative_meaning_set/V1` in a focused `communicative_meaning.py` module:

```text
CommunicativeMeaningSet
  meaning_set_id
  context_id
  candidates[]                 # 0..8; ranked, not prematurely collapsed
  unresolved_tensions[]
  abstention                    # allowed and reason-coded
  private_evidence_summary     # references/counts, not copied payloads
  interpreter / version
  provenance[] / created_at

MeaningCandidate
  candidate_id
  communicative_act            # stable/native act ID
  proposition_references[]     # events, concepts, entities, relations
  stance                       # graded medium-neutral dimensions
    commitment / directness / urgency / warmth
    playfulness / vulnerability / formality
  audience_model_references[]
  disclosure                   # shareable/private/withhold and evidence
  support[]                    # role, direction, weight, provenance
  contradictions[]
  confidence / specificity
  reuse_or_novelty_provenance[]
```

The act IDs and stance axes are a small semantic grammar, not an English phrase table. Debug glosses may exist, but canonical values must be stable/native structures.

`MeaningInterpretation` should record how observations produced candidates: evidence considered, composition/retrieval operations, rejections, contradictions, and bounds. `ExpressionStance` remains a substructure because stance without a proposition or act cannot answer what is being conveyed.

Preserve `ExpressionIntent`. First add optional `meaning_references` compatibly rather than immediately replacing V1. The intent should express why communication is attempted, which candidate(s) may be conveyed, relevant evidence/context references, expression constraints, unresolved alternatives, allowed media, and provenance. Candidate selection remains distinct from realisation.

## Composition and learning without phrase rules

1. Collect caller-supplied, bounded witnesses from thoughts, semantic events, native intents, memories, affect/body state, urge/arbitration, audience context, and humour traces. Never scan raw memory stores.
2. Normalize roles rather than words: agent/topic/relation, certainty, approach/avoidance, salience, disclosure safety, social orientation, contradiction, and recency. Affect remains evidence, not the proposition.
3. Retrieve bounded structural precedents linking evidence configurations to communicative acts/propositions/stances. Keep lexical surfaces outside this store.
4. Compose several candidates by unifying compatible roles and relations. Affect may modulate commitment, disclosure, urgency, or manner only when other evidence supplies content. High trust alone cannot manufacture “I trust you.”
5. Score signed support and contradiction using the inspectable pattern in `ThoughtProcessor.decide()`. Penalize unsupported entities, first-person affect overclaims, stale evidence, and excess specificity. Retain close alternatives.
6. Apply worth-saying and disclosure gates. Urge controls opportunity/access, not content. No candidate, private content, or insufficient value yields abstention/defer/silence.
7. Reference the chosen candidate and alternatives in an output-neutral intent, then invoke a realiser.

Learn structural episodes, not canned sentences: evidence-pattern IDs, candidates, selected intent, medium realisations, reaction observations, and competing reaction interpretations. Later review may revise associations. Approval, reply speed, emoji, or compliance must never become scalar reward or ground truth.

Sources may include Ina’s provenance-linked thought revisions, repeated cross-modal co-occurrence, explicit human explanations as one identified witness, retrospective intended-versus-interpreted comparisons, and contrastive episodes where similar state led to silence/privacy/different media. Split validation by conversation episode, source document, and phrase family to expose retrieval masquerading as composition.

## Native participation and realisers

- Non-linguistic thought supplies native concepts, relations, vectors, affect, and body state directly.
- Linguistic thought supplies semantic events/native intents, not merely source text.
- Meaning candidates reference those structures; their acts/stances use stable/native IDs rather than English sentences.
- Native-symbol realisation selects native constructions directly.
- English realisation maps the same candidate through semantic roles, discourse referents, constructions, and surface generation, exposing uncertainty.
- Voice/prosody consumes stance/emphasis plus a phonological plan; gesture consumes stance, referents, and turn function; music consumes contour, tension/resolution, timing, and intensity. None treats English output as authoritative meaning.

Every realisation should cite its meaning candidate, construction/mapping provenance, unresolved slots, and fallback use. A medium may fail or remain silent rather than distort the claim.

Humour should contribute a resolved-contradiction/benign-violation relation, playfulness, audience safety, recency, and provenance. `humor_engine.py` already stores traces and creates an optional safety-conscious invite. Humour becomes candidate evidence/stance, not a canned-output generator or forced act.

## Diagnostic separation

```text
runtime witnesses --> diagnostic projection --> dashboard/operator command/trace
                 \
                  --> meaning interpretation --> social expression
```

- Mark `format_emotion_signal()` operator diagnostic; do not let it satisfy an ordinary reply.
- Permit raw state speech only after an explicit operator/debug request or a supported meaning whose proposition is instrumentation itself.
- Preserve diagnostic payloads in metadata even when unspoken.
- Treat code pointers similarly as deliberate information, not automatic fallback.
- Move early-communication mechanism phrases to logs/fragment metadata.
- Prefer clarification, deferral, privacy, or silence over telemetry when meaning is absent.

## Migration

1. **Baseline:** add a versioned benchmark for current Discord/early-comm behaviour and bounded per-response retention of lexical choices/alternatives, selected strategy, intent, and realisations (without private thought payloads).
2. **Pure schema/interpreter:** add validators and a deterministic bounded interpreter; add optional `meaning_references` to Expression Core. Run only on supplied witnesses.
3. **Thought integration:** make `prepare_communication()` accept/derive a meaning set; reuse guided evidence composition; expose through the existing Model Manager façade.
4. **Discord shadow mode:** derive candidates from scene, bounded memory consideration, thoughts/native intent, affect, urge, and audience. Retain comparisons while output remains unchanged.
5. **Realiser routing:** make text/native/voice/gesture/music consume selected meaning. Move emotion/humour to evidence and stance. Keep explicit diagnostics separate; fall back to clarify/defer/silence.
6. **Early-comm migration:** retain token/tone/riff algorithms as realiser resources while replacing social mechanism narration. Preserve urge, trust, cooldown, target, and non-expression gates.
7. **Reviewed promotion:** compare pinned historical behaviour, keep compatibility paths instrumented, and require a written report plus reversible snapshot.

## Required tests and benchmarks

Schema/boundary tests:

- Retain 0..N candidates, alternatives, contradictions, confidence, provenance, privacy, and abstention.
- Canonical meaning needs no English, punctuation, phoneme, avatar, or medium convention.
- Private witness payloads never enter public intent/trace records.
- Meaning, intent, realisation, reaction, and interpretation IDs form an intact bounded chain.

Composition tests:

- Affect alone may change stance/disclosure but cannot invent a proposition or audience claim.
- Identical state with different thought/memory/audience context yields different hypotheses.
- One proposition can survive changed affect while its stance changes.
- Conflicting/tied/weak witnesses retain alternatives or abstain.
- High urge without content is silent; strong content may remain private/deferred.
- Humour contributes only with suitable trace and audience evidence.

Realiser/regression tests:

- One candidate is independently realised through test native, text, voice, gesture, and music realisers without consuming another medium’s output.
- Native works without source English; English retains referent and whole-utterance uncertainty.
- Realiser failure never mutates meaning or silently emits diagnostics.
- Length limits preserve uncertainty/negation or abstain.
- Ordinary unknown/nonconstructive Discord input no longer produces `State signal`; explicit diagnostics still support 6/24 sliders.
- Early communication no longer socially emits the listed mechanism phrases.
- A pinned replay of this trace must not conclude that `meanings` proves a request.
- Lexical `resolved_meanings` remains visibly distinct from communicative meaning.

Add explicit benchmark histories:

- `communicative_meaning` V1/V2;
- `expression_core` V2/V3;
- `thought_processor` V4/V5;
- `discord_expression` V1/V2;
- `early_comm_expression` V1/V2.

Score evidence coverage, unsupported propositions, ambiguity calibration, referent grounding, privacy/abstention, diagnostic leakage, cross-medium equivalence, phrase-family novelty, verbosity, reaction-as-witness compliance, and bounded resources. Use deterministic held-out conversations/source documents and adversarial minimal pairs.

## Failure modes and controls

| Failure | Control |
|---|---|
| Confabulated meanings | Require proposition-slot evidence, negative evidence, freshness, and unsupported-slot rejection reasons. |
| Emotional overclaiming | Affect is a witness/stance modifier; relational or self-state claims need explicit scoped evidence. |
| Parroting training phrases | Separate structural episodes from surfaces; hold out phrase families/sources and measure nearest-training overlap. |
| Personality hard-coding | Keep stance tendencies contextual, provenance-linked, revisable, and subordinate to privacy/current evidence. |
| Excessive verbosity | Use focus/disclosure budgets and measure semantic coverage per token; never drop uncertainty or negation. |
| Permanent diagnostic speech | Operator-only diagnostics; ordinary fallback tests must prefer clarify/defer/silence. Track leakage rate. |
| Human feedback as reward | Preserve observation plus multiple interpretations; prohibit direct scalar reinforcement from reactions. |
| Language as hidden cognition | Require non-linguistic fixtures, native IDs, independent native realisation, and no canonical English field. |
| Early ambiguity collapse | Retain close candidates/contradictions into intent; realisers qualify or abstain. |
| Privacy leakage | Separate think eligibility from disclosure; reference private evidence without copying it; test audience changes. |
| State-to-phrase shortcut | Prohibit single-witness phrase emission; use contrastive same-state/different-context tests. |
| Realiser changes claim | Compare realised semantic/native structure back to candidate and record additions/omissions. |
| Silence treated as failure | Make private/defer/abstain normal reason-coded outcomes and score appropriate non-expression positively. |

## Recommended first implementation slice

After review, the smallest reversible slice is to define/validate the meaning records, add `meaning_references` to the intent boundary, implement a pure bounded interpreter that preserves alternatives and abstention, and run it in Discord shadow mode. Add V1/V2 benchmarks for this example and contrastive cases before routing live output through it. This exercises the missing seam without prematurely encoding a personality, a slider phrasebook, or English as cognition.

## Implemented shadow slice

The greenlit initial slice now provides:

- `communicative_meaning.py` with bounded `CommunicativeMeaningSet`, embedded `MeaningInterpretation`, structural `MeaningCandidate` records, alternatives, contradictions, disclosure state, and reason-coded abstention;
- an evidence-hungry composition rule: witnesses without both an explicit communicative act and proposition cannot create meaning, so affect/urge alone abstains;
- bounded semantic/native conversation examples from recent supplied conversation context, with verbatim surface fields removed by default and reactions explicitly marked as witnesses rather than rewards;
- optional `meaning_references` on `ExpressionIntent` and meaning-aware `ThoughtProcessor.prepare_communication()` plans;
- Discord shadow metadata containing the meaning result and conversation examples while retaining the existing selected output;
- versioned `communicative_meaning` V1/V2, `expression_core` V2/V3, and `thought_processor` V4/V5 comparisons.

This is not yet a learned meaning generator or a production routing change. The
shadow interpreter currently needs structural meaning witnesses from an owning
cognitive subsystem; ordinary affect-only Discord turns intentionally record an
abstention. Existing conversation data is reused in bounded form rather than
copied into a new repository corpus.

### Storage and lookup follow-up

Operation-local measurement found that a six-word lookup against the current
80 MB SQLite projection took 8.24–12.96 seconds on the durable HDD (median
8.76 seconds) and about 0.006 seconds on the configured NVMe across five paired
samples. CPU time on the HDD query was about 0.004 seconds, the adaptive storage
record calculated 0.9996 mean storage attribution, and all five samples exceeded
the latency budget. This agreed with the independent device-health history and
the existing `index -> fast` recommendation.

The implementation now routes the rebuildable text-vocabulary SQLite projection
through `storage_layout.fast_runtime_path()` and supports indexed word/symbol
subsets for latency-sensitive expression paths. Durable JSON and the prior HDD
SQLite source were not moved or deleted. The installed NVMe projection was
copied and SHA-256 verified; both source and projection matched
`2f46bad1128c53d690b67350eef753d93d250c3b86251718ad4492e8d3c729d8`.

This does not make full-map materialisation acceptable: a warm full load still
used roughly 336 MB peak RSS and spent most of its 1.32 seconds in user-space
decoding. Full loads remain available for maintenance, while normal symbolic
reply and realisation paths request only their bounded words or symbols.

### Validation status

Focused native-runner coverage passed for communicative meaning, Expression
Core, Thought Processor, text-vocabulary storage, symbolic dual messages, and
the Discord shadow case. Versioned candidates scored: communicative meaning V2
`4/4`, text-vocabulary lookup V2 `2/2`, Expression Core V3 `7/7`, and Thought
Processor V5 `9/9`, with their retained prior versions still reported.

The complete Discord grounding-routing file was also attempted under a
50-second bound. Its first 20 tests, including the new meaning-shadow test,
passed; the runner then stalled without reporting a failure and was terminated
at the bound. This is incomplete full-file coverage, not a passing claim. The
two tests immediately around the apparent stopping point passed independently.
