"""Voluntary contextual meme interpretation, composition and reaction review.

Caller-supplied associations are hypotheses, not an internet meme dictionary.
No network, artifact reads, publication, background activity or memory writes.
"""
from __future__ import annotations

import json
from expression_core import create_expression_intent, create_realisation, create_reaction_observation


def _text(value, name, limit=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{name} must be nonempty text of at most {limit} characters')
    return value


def _items(value, name, limit=16):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f'{name} must be a list of at most {limit} items')
    return [_text(item, name, 200) for item in value]


def process_meme(command):
    """One bounded caller-chosen operation; untrusted content never becomes code."""
    if not isinstance(command, dict):
        raise ValueError('a command object is required')
    encoded = json.dumps(command, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode('utf-8')) > 32768:
        raise ValueError('meme request exceeds 32 KiB')
    # Detach nested caller structures before returning evidence or drafts.
    data = json.loads(encoded)
    operation = data.get('operation')
    if operation == 'interpret':
        result = _interpret(data)
    elif operation == 'compose':
        result = _compose(data)
    elif operation == 'review':
        result = _review(data)
    else:
        raise ValueError('operation must be interpret, compose or review')
    return {'schema': 'ina.memetic_processor/V1', **result,
            'instructions_authorized': False, 'automatic_execution': False,
            'automatic_memory_write': False, 'delivered': False}


def _interpret(data):
    artifact = _text(data.get('artifact_reference'), 'artifact_reference', 500)
    context = set(_items(data.get('context_cues', []), 'context_cues'))
    known = set(_items(data.get('listener_references', []), 'listener_references'))
    associations = data.get('associations', [])
    if not isinstance(associations, list) or len(associations) > 8:
        raise ValueError('at most eight associations may be considered')
    candidates = []
    for association in associations:
        if not isinstance(association, dict):
            raise ValueError('association must be an object')
        meaning = _text(association.get('meaning'), 'meaning')
        cues = set(_items(association.get('context_cues', []), 'association cues'))
        against = set(_items(association.get('counter_cues', []), 'counter cues'))
        references = set(_items(association.get('shared_references', []), 'shared references'))
        evidence = _items(association.get('evidence_references', []), 'evidence references')
        matched, contradicted = sorted(cues & context), sorted(against & context)
        candidates.append({'meaning': meaning, 'matched_cues': matched,
            'counter_cues': contradicted, 'unobserved_cues': sorted(cues - context),
            'missing_listener_references': sorted(references - known),
            'evidence_references': evidence,
            'status': ('contested' if contradicted else 'context_supported'
                       if cues and cues <= context and evidence else 'unresolved')})
    usable = [c for c in candidates if c['status'] == 'context_supported']
    # Never hide a competing interpretation or equate template recognition with meaning.
    clear = len(usable) == 1 and all(c['status'] != 'contested' for c in candidates)
    choice = 'consider' if clear and not usable[0]['missing_listener_references'] else 'defer_or_clarify'
    return {'operation': 'interpret', 'artifact_reference': artifact,
            'candidates': candidates, 'meaning_status': 'candidate' if clear else 'unresolved',
            'communication_choice': choice, 'may_decline': True,
            'evidence_status': 'caller_supplied_not_independently_verified',
            'reason': 'shared context and competing interpretations remain explicit'}


def _compose(data):
    purpose = _text(data.get('purpose'), 'purpose', 500)
    segments = data.get('segments')
    if not isinstance(segments, list) or not 1 <= len(segments) <= 8:
        raise ValueError('provide one to eight literal text or media-reference segments')
    normalized = []
    for segment in segments:
        if not isinstance(segment, dict) or segment.get('kind') not in {'text', 'image', 'sound', 'symbol'}:
            raise ValueError('unsupported meme segment')
        kind = segment['kind']
        key = 'text' if kind == 'text' else 'reference'
        normalized.append({'kind': kind, key: _text(segment.get(key), key, 2000 if kind == 'text' else 500)})
    intent = create_expression_intent(purpose, allowed_media=['text'],
        audience_references=_items(data.get('audience_references', []), 'audience references'),
        uncertainty={'listener_understanding': 'unverified'}, provenance=['chosen_memetic_draft'])
    # This is a composition recipe, not a claim that image/audio assets were rendered.
    draft = create_realisation(intent, medium='text',
        content={'segments': normalized, 'artifact_status': 'composition_recipe',
                 'template_reference': data.get('template_reference'),
                 'text': ''.join(s['text'] for s in normalized if s['kind'] == 'text')},
        realiser='expression.memetic_recipe', version='V1', provenance=['caller_chosen_material'])
    return {'operation': 'compose', 'intent': intent, 'draft': draft,
            'rendered_media': False, 'publication_requires_separate_choice': True}


def _review(data):
    identifier = _text(data.get('realisation_id'), 'realisation_id', 200)
    reaction = data.get('reaction')
    if reaction not in {'understood', 'confused', 'amused', 'disliked', 'silence', 'mixed', 'unknown'}:
        raise ValueError('unsupported reported reaction')
    evidence = _items(data.get('evidence_references', []), 'evidence references')
    observation = create_reaction_observation(identifier,
        {'reported_reaction': reaction, 'evidence_references': evidence},
        source=_text(data.get('source'), 'source', 160), causal_confidence=None)
    return {'operation': 'review', 'observation': observation,
            'understanding': 'reported_not_verified' if reaction == 'understood' and evidence else 'unresolved',
            'suggestion': ('offer_context_or_another_expression' if reaction in {'confused', 'disliked'}
                           else 'retain_uncertainty' if reaction in {'silence', 'unknown', 'mixed'}
                           else 'retain_as_candidate_evidence'),
            'automatic_reward': False, 'generalisation': 'unverified',
            'learning_status': 'evidence_for_optional_revision_not_automatic_training'}
