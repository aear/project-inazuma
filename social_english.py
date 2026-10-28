"""Voluntary, private support for social English; no reply generator or sender."""
from __future__ import annotations

from typing import Any, Mapping
from language_intelligence import analyze_utterance

MAX_TEXT_BYTES = 8192
CHOICES = frozenset({'english', 'mixed', 'native_only', 'clarify', 'abstain', 'silence'})


def _text(value: Any, name: str, *, required: bool = False) -> str:
    if not isinstance(value, str) or len(value.encode('utf-8')) > MAX_TEXT_BYTES:
        raise ValueError(f'{name} must be text of at most 8192 UTF-8 bytes')
    if required and not value.strip():
        raise ValueError(f'{name} must not be empty')
    return value


def social_english_options(*, topic_reference: str = '', audience_reference: str = '') -> dict[str, Any]:
    """Offer both owner-requested forms of encouragement without inferring uptake."""
    for name, value in (('topic_reference', topic_reference), ('audience_reference', audience_reference)):
        if not isinstance(value, str) or len(value) > 256:
            raise ValueError(f'{name} must be a bounded reference')
    return {
        'schema': 'ina.social_english_options/V1',
        'invitation_author': 'operator_requested_support',
        'invitation': 'English is welcome in ordinary conversation. Short or mixed replies are welcome too. You can also choose a brief private practice turn.',
        'topic_reference': topic_reference, 'audience_reference': audience_reference,
        'opportunities': [
            {'kind': 'ordinary_conversation', 'activity': 'Express one thing you choose, or ask one follow-up question; invite a listener to check the meaning if useful.'},
            {'kind': 'chosen_practice', 'activity': 'Supply an English draft or incoming sentence for one private review; revise, keep, defer or stop as you choose.'},
        ],
        'choices': ['english', 'mixed', 'native_only', 'clarify', 'abstain', 'silence'],
        'willingness_inferred': False, 'automatic_expression': False,
        'automatic_learning': False, 'scheduled': False, 'continuation_budget': 0,
        'language_reward': False, 'outward_text': None,
    }


def _structure(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = ('parent_clause', 'subject', 'predicate', 'arguments', 'negated', 'modality', 'tense', 'voice')
    return [{key: clause.get(key) for key in keys} for clause in analysis['clauses']]


def review_social_english(
    draft: str = '', *, incoming_text: str = '', reference_draft: str = '',
    mode: str = 'conversation', choice: str = 'english',
    discourse: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Review supplied wording; never invent a thought, correction, or outward reply.

    Structural agreement is a heuristic check, not a certificate of meaning.
    Comparing a reply to an incoming message is deliberately not a fidelity test.
    """
    draft = _text(draft, 'draft')
    incoming_text = _text(incoming_text, 'incoming_text')
    reference_draft = _text(reference_draft, 'reference_draft')
    if choice not in CHOICES or mode not in {'conversation', 'rephrase'}:
        raise ValueError('choose a supported language choice and review mode')
    result: dict[str, Any] = {
        'schema': 'ina.social_english_review/V1', 'draft': draft,
        'incoming_text': incoming_text, 'reference_draft': reference_draft,
        'choice': choice, 'choice_is_caller_supplied': True, 'mode': mode,
        'status': 'choice_preserved', 'issues': [],
        'draft_analysis': None, 'incoming_analysis': None,
        'structural_comparison': {'status': 'not_assessed'},
        'meaning_fidelity': 'not_certified', 'listener_understanding': 'not_assessed',
        'feedback_scope': 'heuristic analyzer limits; not a verdict that the speaker is wrong',
        'correction_required': False,
        'automatic_expression': False, 'automatic_learning': False,
        'outward_text': None, 'rewritten_text': None, 'scheduled': False,
    }
    if choice not in {'english', 'mixed'}:
        return result
    if not draft.strip() and not incoming_text.strip():
        result['status'] = 'no_material_supplied'
        return result
    context = {'discourse': discourse} if isinstance(discourse, Mapping) else {}
    if draft.strip():
        result['draft_analysis'] = analyze_utterance(draft, context=context)
    if incoming_text.strip():
        result['incoming_analysis'] = analyze_utterance(incoming_text, context=context)
    for role in ('draft', 'incoming'):
        analysis = result[f'{role}_analysis']
        if analysis is None:
            continue
        if analysis['incomplete']:
            result['issues'].append({'source': role, 'kind': 'analysis_incomplete'})
        if any(not clause.get('predicate') for clause in analysis['clauses']):
            result['issues'].append({'source': role, 'kind': 'unsupported_predicate_or_construction'})
        if any(ref.get('resolved') is None for ref in analysis['referents']):
            result['issues'].append({'source': role, 'kind': 'referent_needs_context'})
    if mode == 'rephrase':
        if not reference_draft.strip() or not draft.strip():
            raise ValueError('rephrase review requires both a reference draft and a candidate draft')
        original = analyze_utterance(reference_draft, context=context)
        candidate = result['draft_analysis']
        supported = not original['incomplete'] and not candidate['incomplete'] and all(
            row.get('predicate') for analysis in (original, candidate) for row in analysis['clauses'])
        same = _structure(original) == _structure(candidate)
        status = 'same_supported_structure' if supported and same else 'differences_need_review' if supported else 'not_assessed'
        result['structural_comparison'] = {'status': status, 'reference_analysis': original,
                                           'heuristic_only': True}
        if status != 'same_supported_structure':
            result['issues'].append({'source': 'rephrase', 'kind': status})
    result['status'] = 'private_review'
    result['next_choices'] = ['keep_wording', 'revise_if_chosen', 'ask_for_clarification', 'invite_listener_check', 'defer', 'stop']
    return result
