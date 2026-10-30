"""Private English candidates from caller-selected structured meaning, not prompts."""
from __future__ import annotations
from collections.abc import Mapping
import re
from expression_core import TextRealiser, text_expression_guidance, INTENT_SCHEMA
from language_intelligence import _VERBS, _MODALS

_IRREGULAR = {'be':('was','been'),'give':('gave','given'),'tell':('told','told'),
 'teach':('taught','taught'),'find':('found','found'),'keep':('kept','kept'),
 'know':('knew','known'),'think':('thought','thought'),'say':('said','said'),
 'go':('went','gone'),'see':('saw','seen'),'write':('wrote','written'),
 'read':('read','read'),'feel':('felt','felt'),'have':('had','had'),
 'choose':('chose','chosen'),'steal':('stole','stolen'),'run':('ran','run'),'prefer':('preferred','preferred'),'understand':('understood','understood')}


class UnsupportedEnglish(ValueError):
    pass


def _phrase(value, *, maximum=192):
    if isinstance(value,Mapping):value=value.get('surface')
    if not isinstance(value,str) or not 1<=len(value)<=maximum:raise UnsupportedEnglish('missing_or_oversized_lexical_surface')
    if re.search(r'[!?;\n\r]|(?<!\d)\.|\.(?!\d)',value):raise UnsupportedEnglish('multi_clause_lexical_surface')
    return ' '.join(value.split())


def _agent(value):
    if isinstance(value,Mapping):
        if value.get('id')=='self' or value.get('is_self'):return 'I'
        value=value.get('name') or value.get('surface')
    text=_phrase(value,maximum=96)
    return 'I' if text.casefold()=='i' else text


def _past(verb, participle=False):
    if verb in _IRREGULAR:return _IRREGULAR[verb][1 if participle else 0]
    if verb.endswith('e'):return verb+'d'
    if verb.endswith('y') and len(verb)>1 and verb[-2] not in 'aeiou':return verb[:-1]+'ied'
    return verb+'ed'


def compose_selected_english(native_intent, expression_intent, *, choice='english', cognition_plan=None):
    """Offer one bounded candidate; references are caller claims, not truth proofs.

    No incoming-message shortcut, thought selection, delivery, learning or reward.
    Unsupported structures return no candidate rather than losing a role/scope.
    """
    if choice not in {'english','mixed','native_only','clarify','abstain','silence'}:raise ValueError('unsupported expression choice')
    result={'schema':'ina.english_composition/V1','status':'choice_preserved','candidate_text':None,
            'choice':choice,'automatic_expression':False,'automatic_learning':False,
            'outward_text':None,'meaning_fidelity':'not_certified','grounding':'caller_supplied_references_not_verified'}
    if choice not in {'english','mixed'}:return result
    if not isinstance(expression_intent,Mapping) or expression_intent.get('schema')!=INTENT_SCHEMA:raise ValueError('a selected expression intent is required')
    if not (expression_intent.get('meaning_references') or expression_intent.get('semantic_references')):raise ValueError('selected meaning references are required')
    if not isinstance(native_intent,Mapping) or native_intent.get('version')!=1:raise ValueError('a supported native intent is required')
    if native_intent.get('incomplete'):return {**result,'status':'unsupported_structure','reason':'incomplete_selected_meaning'}
    events=native_intent.get('events')
    if not isinstance(events,list) or not 1<=len(events)<=4:raise ValueError('select 1..4 events')
    if any(not isinstance(e,Mapping) for e in events):raise ValueError('events must be mappings')
    ids=[e.get('id') for e in events]
    if any(not isinstance(i,str) or not 1<=len(i)<=32 for i in ids) or len(set(ids))!=len(ids):raise ValueError('unique bounded event ids required')
    by_id=dict(zip(ids,events));children={i:[] for i in ids};roots=[]
    for e in events:
        parent=e.get('parent_clause')
        if parent:
            if parent not in by_id:raise ValueError('missing parent event')
            children[parent].append(e['id'])
        else:roots.append(e['id'])
    if not roots:raise ValueError('cyclic event structure')
    guidance=text_expression_guidance(cognition_plan)
    if guidance['status']=='unknown':return {**result,'status':'needs_evidence','missing_evidence':guidance['missing_evidence']}
    if guidance['status']=='uncertain' and any(by_id[i].get('modality') not in {'might','may'} for i in roots):return {**result,'status':'needs_chosen_qualification'}
    seen=set()
    def render(identifier,depth=0):
        if depth>3 or identifier in seen:raise UnsupportedEnglish('cyclic_or_deep_structure')
        seen.add(identifier);e=by_id[identifier]
        if e.get('question_word'):raise UnsupportedEnglish('wh_question_scope_not_supported')
        if e.get('negation_operator') not in {None,'not'}:raise UnsupportedEnglish('negation_operator_not_supported')
        verb=e.get('predicate')
        if not isinstance(verb,str) or verb not in _VERBS:raise UnsupportedEnglish('unsupported_predicate')
        tense=e.get('tense') or 'present';modal=e.get('modality')
        if not isinstance(tense,str) or tense not in {'present','past'} or (modal and (not isinstance(modal,str) or modal not in _MODALS)):raise UnsupportedEnglish('unsupported_tense_or_modality')
        if modal and tense=='past':raise UnsupportedEnglish('past_modal_scope_not_supported')
        negative=e.get('negated',False)
        if not isinstance(negative,bool):raise UnsupportedEnglish('negation_must_be_explicit_boolean')
        if e.get('negation_operator')=='not' and not negative:raise UnsupportedEnglish('contradictory_negation_fields')
        args=e.get('arguments') or {}
        if not isinstance(args,Mapping) or set(args)-{'theme','recipient','addressee'}:raise UnsupportedEnglish('unsupported_argument_roles')
        alternatives=e.get('alternatives') or []
        if alternatives:
            if not isinstance(alternatives,list) or len(alternatives)!=1:raise UnsupportedEnglish('alternative_structure_not_supported')
            options=alternatives[0].get('options') if isinstance(alternatives[0],Mapping) else None
            if not isinstance(options,list) or not 2<=len(options)<=4:raise UnsupportedEnglish('alternative_options_missing')
            phrase=' or '.join(_phrase(option) for option in options)
            if _phrase(args.get('theme')) not in {phrase,'either '+phrase}:raise UnsupportedEnglish('alternative_surface_mismatch')
        if children[identifier] and ('theme' in args or len(children[identifier])!=1):raise UnsupportedEnglish('ambiguous_complement_structure')
        subject=e.get('agent')
        voice=e.get('voice') or 'active'
        if voice=='passive':
            if children[identifier] or 'theme' not in args or modal or e.get('inverted'):raise UnsupportedEnglish('unsupported_passive_structure')
            theme=_phrase(args['theme']);plural=e.get('agreement_number')=='plural' or theme.casefold() in {'we','they','you'}
            auxiliary=('were' if plural else 'was') if tense=='past' else ('are' if plural else 'is')
            parts=[theme,auxiliary,*(['not'] if negative else []),_past(verb,True)]
            if subject:
                agent=_agent(subject);agent={'i':'me','we':'us','he':'him','she':'her','they':'them'}.get(agent.casefold(),agent)
                parts+=['by',agent]
        elif voice=='active':
            subject=_agent(subject)
            plural=e.get('agreement_number')=='plural' or subject.casefold() in {'you','we','they'} or ' and ' in subject.casefold()
            third=not plural and subject.casefold()!='i'
            complement=[]
            if 'recipient' in args or 'addressee' in args:
                if verb not in {'give','tell','teach'}:raise UnsupportedEnglish('recipient_construction_not_supported')
                recipient=_phrase(args.get('recipient',args.get('addressee')))
                complement.append(recipient)
            if 'theme' in args:complement.append(_phrase(args['theme']))
            inverted=bool(e.get('inverted'))
            if modal:auxiliary=modal;lexical=verb
            elif verb=='be':
                auxiliary=('were' if plural else 'was') if tense=='past' else ('am' if subject=='I' else 'are' if plural else 'is');lexical=None
            elif inverted or negative:auxiliary='did' if tense=='past' else 'does' if third else 'do';lexical=verb
            else:auxiliary=None;lexical=_past(verb) if tense=='past' else ('has' if verb=='have' else verb+'es' if verb.endswith(('s','x','ch','sh','o')) else verb[:-1]+'ies' if verb.endswith('y') and verb[-2] not in 'aeiou' else verb+'s') if third else verb
            parts=([auxiliary,subject] if inverted and auxiliary else [subject,*([auxiliary] if auxiliary else [])])
            if negative:parts.append('not')
            if lexical:parts.append(lexical)
            parts+=complement
            if children[identifier]:
                if inverted:raise UnsupportedEnglish('embedded_question_realisation_not_supported')
                child=by_id[children[identifier][0]]
                if child.get('clausal_relation') not in {'reported_content','attitude_content'}:raise UnsupportedEnglish('unknown_complement_relation')
                parts+=['that',render(child['id'],depth+1)]
        else:raise UnsupportedEnglish('unsupported_voice')
        return ' '.join(parts)
    try:
        sentences=[]
        for identifier in roots:
            text=render(identifier)
            text=text[:1].upper()+text[1:]+('?' if by_id[identifier].get('inverted') else '.')
            sentences.append(text)
        if len(seen)!=len(events):raise UnsupportedEnglish('unreachable_or_cyclic_event')
        candidate=' '.join(sentences)
        if len(candidate)>2048:raise UnsupportedEnglish('output_budget')
    except UnsupportedEnglish as exc:return {**result,'status':'unsupported_structure','reason':str(exc)}
    realisation=TextRealiser().realise(expression_intent,content={'text':candidate,
        'response_kind':'qualified_expression' if guidance['status']=='uncertain' else 'expression',
        'source_event_ids':ids},cognition_plan=cognition_plan,provenance=['english_composition:V1'])
    return {**result,'status':'candidate_for_review','candidate_text':candidate,'realisation':realisation,
            'source_event_ids':ids,'listener_understanding':'not_assessed','willingness_inferred':False}
