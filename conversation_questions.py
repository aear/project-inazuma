"""Bounded conversational conventions, not obligations or evidence of knowledge."""
from __future__ import annotations
import re

MAX_CHARACTERS = 4096
_CHOICES = ['answer_if_grounded', 'ask_for_clarification', 'acknowledge_uncertainty',
            'defer', 'native_expression', 'deliberate_mirror', 'silence']
# Keep contractions intact. Mask quotations/code/links before using punctuation.
_QUOTED = re.compile(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]*\x27(?!\w)|‘[^’\n]*’|https?://\S+|(?m:^\s*>[^\n]*)')
_START = re.compile(r'^(?:who|what|when|where|why|how|which|whose)\b', re.I)
_POLAR = re.compile(r'^(?:am|is|are|was|were|do|does|did|have|has|had|can|could|will|would|shall|should|may|might|must)\b\s+\w+', re.I)


def question_context(text: str) -> dict:
    """Describe an answer invitation; repeated '?' never increases its weight.

    Limited English surface heuristics: rhetorical intent, truth, willingness,
    recipient identity and semantic comprehension remain unverified.
    """
    result = {'schema': 'ina.conversation_question/V1', 'kind': 'not_identified',
              'answer_invitation': False, 'has_question_mark': False,
              'complete': True, 'response_bonus': 0.0, 'mirror_adjustment': 0.0,
              'choices': list(_CHOICES), 'answer_required': False,
              'knowledge_inferred': False, 'willingness_inferred': False,
              'automatic_learning': False, 'rhetorical_intent': 'unresolved', 'requested_information': 'unresolved',
              'convention': 'Questions often invite a relevant answer. Clarification, uncertainty, deferral and silence remain available.'}
    if not isinstance(text, str) or len(text) > MAX_CHARACTERS:
        return {**result, 'complete': False, 'kind': 'input_budget_exceeded'}
    visible = _QUOTED.sub(lambda m: ' ' * len(m.group()), text)
    # Unclosed code or quotation makes the remaining punctuation unreliable.
    if any(c in visible for c in ('`', '"', '“', '”', '‘', '’')) or re.search(r"(?<!\w)'", visible):
        # A curly contraction apostrophe is not an opening quotation.
        visible = re.sub(r'(?<=\w)’(?=\w)', "'", visible)
        if any(c in visible for c in ('`', '"', '“', '”', '‘', '’')) or re.search(r"(?<!\w)'", visible):
            return {**result, 'complete': False, 'kind': 'quotation_unresolved'}
    visible = ' '.join(visible.split()).strip()
    result['has_question_mark'] = '?' in visible or '？' in visible
    lower = visible.casefold().replace('’', "'")
    lower = re.sub(r'^(?:hey|hi|hello)[,!]?\s+', '', lower)
    lower = re.sub(r'^please\s+', '', lower)
    if re.match(r"^(?:no (?:need|pressure) to (?:answer|reply)|you (?:don\x27t|do not) (?:need|have) to (?:answer|reply)|(?:do not|don\x27t) (?:answer|reply))\b", lower):
        return {**result, 'kind': 'answer_not_requested'}
    if re.match(r'^(?:(?:she|he|they|i|we|\w+) (?:asked|said|wondered)|(?:the|a) question (?:is|was))\b', lower):
        return {**result, 'kind': 'reported_or_mentioned'}
    # One short comma-marked vocative, e.g. "Ina, do you ...".
    lower = re.sub(r'^\w{1,80},\s+(?=(?:who|what|when|where|why|how|which|whose|am|is|are|was|were|do|does|did|have|has|had|can|could|will|would|shall|should|may|might|must)\b)', '', lower)
    if re.match(r'^(?:what (?:a|an)\b|how (?:lovely|beautiful|nice|wonderful)\b)', lower) and not result['has_question_mark']:
        return {**result, 'kind': 'possible_exclamation'}
    if _START.match(lower) and not result['has_question_mark']:
        # Fronted subordinate clauses are often statements, not direct questions.
        if re.match(r'^(?:what|where|when|how|why|which|whose)\s+(?:i|you|we|they|he|she|it|the|a|an)\b', lower):
            return {**result, 'kind': 'embedded_or_declarative'}
    if _START.match(lower):
        kind = 'information_question'
    elif _POLAR.match(lower):
        kind = 'polar_or_request'
    elif result['has_question_mark'] and re.search(r'\w', lower):
        kind = 'question_mark_fragment'
    else:
        return result
    first = lower.split()[0] if lower.split() else ''
    requested = {'who': 'person', 'whose': 'possessor', 'where': 'place',
                 'when': 'time', 'why': 'reason', 'how': 'method_or_state',
                 'which': 'selection', 'what': 'thing_or_description'}.get(first,
                    'yes_no_or_requested_action' if kind == 'polar_or_request' else 'unresolved')
    return {**result, 'kind': kind, 'requested_information': requested, 'answer_invitation': True,
            'response_bonus': 0.12, 'mirror_adjustment': -0.18}


def review_conversational_reply(incoming_text: str, draft: str) -> dict:
    """Flag a surface echo without declaring any other wording a correct answer."""
    context = question_context(incoming_text)
    result = {'schema': 'ina.conversational_reply_review/V1', 'question': context,
              'status': 'not_assessed', 'answer_correctness': 'not_assessed',
              'automatic_rewrite': False, 'outward_text': None}
    if not context['complete'] or not isinstance(draft, str) or len(draft) > MAX_CHARACTERS:
        return {**result, 'status': 'input_unresolved'}
    if not draft.strip():
        return {**result, 'status': 'no_expression', 'choice_preserved': True}
    words = lambda value: re.findall(r'\w+(?:[\x27’]\w+)*', value.casefold())
    original, candidate = words(incoming_text), words(draft)
    if original and original == candidate:
        return {**result, 'status': 'surface_echo',
                'echo_is_answer_evidence': False,
                'suggested_choices': list(_CHOICES)}
    return {**result, 'status': 'distinct_wording_unverified'}
