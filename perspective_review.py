"""Optional tool and event-instinct experiments sharing one bounded review core.

One call is one step. No callbacks, memory retrieval, belief adoption or scheduler.
A continuation returns control to the caller before any further work can happen.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import islice
from typing import Any, Iterable, Mapping

from self_inquiry_journey import begin_intuition_inquiry, continue_self_inquiry

SCHEMA = 'ina.perspective_review/V1'
EXITS = ('continue', 'stop', 'defer', 'ask_evidence', 'remain_uncertain')
MAX_STEPS = 4


def begin_review(event: Mapping[str, Any], *, variant: str, chosen: bool = False,
                 max_steps: int = 1) -> dict[str, Any] | None:
    """Offer a provisional review; the instinct may offer it without a tool choice."""
    if variant not in {'private_tool', 'instinct'}:
        raise ValueError('unknown perspective variant')
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError('max_steps must be 1..4')
    if variant == 'private_tool' and chosen is not True:
        return None
    # Structural events, not elapsed time or arbitrary confidence scores.
    triggers = [key for key in ('contradiction', 'unexpected_outcome', 'novelty')
                if event.get(key) is True]
    if variant == 'instinct' and not triggers:
        return None
    refs = list(islice(event.get('evidence_references') or (), 16))
    if not refs:
        return None
    journey = begin_intuition_inquiry(event.get('interpretation_reference'),
        question=event.get('question'), countercheck=event.get('countercheck'),
        trigger_references=refs, depth_budget=max_steps)
    return {'schema': SCHEMA, 'variant': variant, 'status': 'ready',
            'remaining_steps': max_steps, 'journey': journey,
            'seen_references': list(journey['trigger_references']),
            'seen_checks': [journey['intuition']['countercheck']],
            'triggers': triggers, 'trace': [], 'available_choices': list(EXITS),
            'automatic_adoption': False, 'writes_memory': False}


def review_step(state: Mapping[str, Any], *, choice: str,
                observation_references: Iterable[str] = (), countercheck: str = '',
                hypotheses: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Record one bounded step and yield. Stopped states cannot be resumed implicitly.

    A new reference is an assertion of new evidence, not verification of its
    contents or independence. Semantic quality and source independence remain
    separate checks; a fabricated reference must never certify an interpretation.
    """
    if state.get('schema') != SCHEMA:
        raise ValueError('a perspective review state is required')
    if state.get('status') != 'ready':
        raise PermissionError('review has yielded or ended; begin a new chosen review')
    if choice not in EXITS:
        raise ValueError('an explicit continuation or exit choice is required')
    result = deepcopy(dict(state))
    refs = []
    for raw in islice(observation_references, 16):
        ref = str(raw).strip()[:500]
        if ref and ref not in refs:
            refs.append(ref)
    check = str(countercheck).strip()
    if len(check) > 1000:
        raise ValueError('countercheck exceeds 1000 characters')
    fresh = any(ref not in state['seen_references'] for ref in refs)
    distinct_check = bool(check and check not in state['seen_checks'])
    remaining = int(state['remaining_steps'])
    if not 1 <= remaining <= MAX_STEPS:
        raise PermissionError('review step budget exhausted')
    status = choice
    if choice == 'continue':
        status = 'budget_exhausted' if remaining == 1 else (
            'ready' if fresh or distinct_check else 'no_progress')
    # Reuse existing candidate/history handling; never make a hypothesis authoritative.
    journey_choice = 'deeper' if status == 'ready' else (
        'remain_uncertain' if status != 'stop' else 'stop')
    result['journey'] = continue_self_inquiry(state['journey'], choice=journey_choice,
        observation_references=refs, hypotheses=hypotheses)
    result['remaining_steps'] = remaining - 1
    result['seen_references'] = list(dict.fromkeys(state['seen_references'] + refs))
    result['seen_checks'] = list(state['seen_checks']) + ([check] if distinct_check else [])
    result['trace'].append({'choice': choice, 'outcome': status,
        'observation_references': refs, 'countercheck': check,
        'new_reference_asserted': fresh, 'distinct_check': distinct_check})
    result['status'] = status
    result['available_choices'] = list(EXITS) if status == 'ready' else []
    return result
