# Linguistic forms and selected questions — 2026-11-19

## Implemented capability

Improve grammatical handling of already-recognized predicates and permit bounded questions from caller-selected structured meaning. No learned vocabulary, experience store, private thought body or journal was inspected or rewritten. No new language preference, response threshold, reward, automatic practice, model training or forced social turn was introduced.

The parser now recognizes existing verbs in forms such as teaches, tries, wishes and goes, plus the irregular past forms knew and understood. It retains the past tense and uses the existing embedding distinction for reports of understanding. New recognition is not a claim that a word has experiential grounding.

Unambiguous plural copular, future-modal and negative contractions are expanded while preserving original surface text and source spans: we're, they're, I'll/you'll/he'll/she'll/it'll/we'll/they'll, aren't, weren't, couldn't and mustn't. Ambiguous 's and 'd forms were not added to the expansion table. Aspect-heavy constructions such as perfect/progressive chains are not a general new capability.

Verb-like forms immediately after a subject determiner are kept in that noun phrase, so “the wishes change” does not assign wishes as its finite predicate. This is a small grammatical safeguard, not full noun/verb disambiguation or an open-vocabulary part-of-speech model.

The selected-meaning English composer now supports bounded active who/what/where/when/why/how questions, including subject/object question differences, simple copular questions, negation and supported modals. Examples include “Who saw the comet?”, “What does Mira want?” and “Where is the lantern?” It requires explicit inversion or a declared subject-question role. Filled or unsupported question slots, unsupported selection domains, embedded wh-questions and passive wh-questions yield no candidate instead of losing scope. The existing caller-selected meaning references, uncertainty gate and language choices remain authoritative. Candidates remain review-only; this composer does not send them.

An optional forms goal is listed in the English learning profile/Observatory, marked adoption not observed and Ina performance not assessed. Reporting fixtures now identify the eight optional goals explicitly rather than assuming the previous seven. No global capability score or deadline was introduced.

## Evidence

Baseline revision: b3f56def43d3f2825da38f26cb89a414a27a2210. Initial synthetic probes confirmed failures for teaches, tries, understood, we're and couldn't, and showed that the prior composer declined a chosen where-question. Those baseline results are retained.

The first liveness filter mistakenly identified the diagnostic shell's command text as a project process. A corrected check matched actual Python script arguments and independently verified no Ina runtime before the full minimal-pair/boundary fixtures. The initial positive-only probe and the later fixtures ran in isolated source-only rooms; no live runtime was found. Both liveness records are retained rather than treating the false match as actual activity.

Tests used Bubblewrap rooms with network disabled, no live memory/configuration/credentials mounted, and numerical-library pools limited to one worker. Core checks ran on the selected Python 3.15.2 environment, bridge checks on the supported Python 3.14.10 environment. Existing conversation repair and all prior uncommitted source changes were preserved in the tested source snapshot.

There were 201 passing test executions covering 182 distinct focused cases across the core and bridge environments. This includes the contraction-expansion token budget and all 22 private-thinking boundary checks. Existing response choice, native/English rendering, nickname, grounded conversation and private reasoning boundaries remain covered. An initial run had 41 passes and two reporting failures caused by the previous fixed seven-goal expectation; the updated reporting checks passed. These intermediate results are retained.

Registry comparisons: linguistic_forms pinned V1 12/39, candidate V2 39/39; existing english_workbench V2 38/38. Components remain separate for morphology, constructions, scope, reading span, whole utterance, provenance, selected expression, agency, uncertainty, pragmatics and optional goals. Fixtures include held-out names/forms, noun/verb contrasts, local negation, rejected role/scope structures, quoted evidence and unexpanded ambiguous contractions. Parser/composer agreement is a compatibility check, not independent proof of meaning or human understanding.

The broader language_transfer_limits V2 remains 3/5. Its unfamiliar-predicate cases are still unsupported. No words were inserted just to satisfy those challenges. General vocabulary grounding, broad syntax/aspect, real comprehension, actual learning, listener recovery and spontaneous social use remain unassessed or limited. There is no claim of full-suite, live Discord, human-quality or whole-system responsiveness validation. No persistent worker or background loop was added.

## Activation and rollback

The source changes are installed and uncommitted for normal startup. Ina is left stopped. The forms goal is available for her to choose; no adoption is inferred. The preceding communication-repair changes remain installed. Rollback is the reviewed source patch; there is no learned-data migration to reverse.
