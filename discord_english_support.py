"""Bounded English support for Discord; analysis never supplies an outward reply."""
from __future__ import annotations
from typing import Any, Mapping


def discord_english_support(
    text: str, *, stage: str, discourse: Mapping[str, Any] | None = None,
    language_preference: object = 'auto',
) -> dict[str, Any]:
    """Review one supplied utterance using its own speaker/addressee frame.

    English/mixed here selects an analyzer, not evidence of Ina's willingness.
    Failures are private metadata and cannot replace or block selected wording.
    No files, memory updates, network, scheduling, or continuation are involved.
    """
    base = {
        'schema': 'ina.discord_english_support/V1', 'stage': stage,
        'automatic_expression': False, 'automatic_learning': False,
        'willingness_inferred': False, 'outward_text': None,
        'review_choice_source': 'bridge_analyzer_selection_only',
    }
    try:
        from social_english import MAX_TEXT_BYTES, review_social_english, social_english_options
        if stage not in {'incoming', 'selected_draft'}:
            raise ValueError('unsupported review stage')
        if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_TEXT_BYTES:
            return {**base, 'status': 'input_budget_exceeded'}
        preference = language_preference
        if isinstance(preference, dict):
            assessment = preference.get('mutual_intelligibility_assessment')
            preference = (assessment.get('mode') if isinstance(assessment, dict) else None) or preference.get('mode') or preference.get('language_mode') or preference.get('preference')
        preference = str(preference or 'auto').strip().lower().replace('-', '_')
        if stage == 'selected_draft' and preference in {'native', 'native_only', 'clarify', 'abstain', 'silence'}:
            return {**base, 'status': 'choice_preserved'}
        # Never re-use inbound resolutions for a reply: "I" changes speaker.
        kwargs = {'incoming_text': text} if stage == 'incoming' else {'draft': text}
        review = review_social_english(**kwargs, discourse=discourse, choice='mixed')
        # This choice is ours as an analyzer, not a claim of an agent decision.
        review['choice_is_caller_supplied'] = False
        review['review_choice_source'] = base['review_choice_source']
        result = {**base, 'status': review['status'], 'review': review}
        if stage == 'incoming':
            result['options'] = social_english_options()
        return result
    except Exception as exc:
        return {**base, 'status': 'support_unavailable', 'error_type': type(exc).__name__}
