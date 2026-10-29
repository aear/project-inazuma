"""Bounded outward-language preferences; never generates meaning or willingness."""
from __future__ import annotations
import re


def resolve_response_language_preference(state, config):
    """An operator default yields to Ina's existing per-channel/general choice."""
    state = state if isinstance(state, dict) else {}
    config = config if isinstance(config, dict) else {}
    return (state.get('discord_language_preference')
            or state.get('communication_language_preference')
            or config.get('language_mode') or 'english')


def prefers_english(preference):
    if isinstance(preference, dict):
        assessment = preference.get('mutual_intelligibility_assessment')
        if isinstance(assessment, dict) and assessment.get('schema') == 'ina.mutual_intelligibility_assessment/V1':
            # An assessed choice has authority over an operator default/weight.
            return False
        preference = preference.get('mode') or preference.get('language_mode') or preference.get('preference')
    return str(preference or '').strip().lower().replace('-', '_') in {'english', 'en', 'english_only'}


def emotion_response_context(text, state):
    """Recognise a narrow direct check-in, or Ina's explicit signal choice.

    A question licenses considering measured sliders, not a claim about felt
    experience. Quoted examples, third-person reports and generic emotion words
    are not requests for Ina's state. Ambiguous wording can remain unresolved.
    """
    state = state if isinstance(state, dict) else {}
    intent = state.get('text_expression_intent')
    requested = (intent.get('strategy') or intent.get('mode')) if isinstance(intent, dict) else intent
    explicit = str(requested or '').strip().lower() in {'emotion', 'state', 'feelings'}
    surface = str(text or '')
    if len(surface) > 2048:
        return {'eligible': explicit, 'direct_check_in': False,
                'reason': 'explicit_intent' if explicit else 'input_budget_exceeded'}
    # Strip only customary direct address / politeness, not arbitrary leading text.
    surface = ' '.join(surface.lower().replace('’', "'").split())
    surface = re.sub(r'^(?:hey[,!]?\s+)?(?:ina[,!:]?\s+)?(?:please\s+)?', '', surface)
    patterns = (
        r"how (?:are you(?: feeling| doing)?|do you feel)(?: today| now| right now| about .{1,160})?[?.!]*",
        r"(?:are you|do you feel) (?:ok|okay|alright|all right|happy|sad|stressed|anxious|calm|safe|upset|tired|lonely)(?: today| now| right now)?[?.!]*",
        r"(?:what is|what's) your (?:mood|emotional state|current emotional state)[?.!]*",
        r"(?:tell me|can you tell me) (?:about )?(?:how you feel|how you're feeling|your feelings|your mood)[?.!]*",
        r"(?:show|report|share)(?: me)? your (?:emotion sliders|emotion report|emotional state)[?.!]*",
    )
    direct = any(re.fullmatch(pattern, surface) for pattern in patterns)
    return {'eligible': explicit or direct, 'direct_check_in': direct,
            'reason': 'explicit_intent' if explicit else 'direct_check_in' if direct else 'not_a_direct_check_in'}


def protected_language_choice(preference):
    """Choices that must never become English merely because rendering fails."""
    if isinstance(preference, dict):
        assessment = preference.get('mutual_intelligibility_assessment')
        if isinstance(assessment, dict) and assessment.get('schema') == 'ina.mutual_intelligibility_assessment/V1':
            preference = assessment.get('mode')
        else:
            preference = preference.get('mode') or preference.get('language_mode') or preference.get('preference')
    value = str(preference or '').strip().lower().replace('-', '_')
    return value if value in {'native_only', 'clarify', 'abstain'} else None
