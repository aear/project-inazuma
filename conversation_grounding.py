"""Recent conversational evidence and optional replies; no durable learner or sender.

Messages are attributed claims, not world facts. One event can offer one bounded
answer/clarification choice; no reward, continuation, file access or model update.
"""
from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re
import unicodedata
from semantic_event import build_semantic_event, build_native_intent
from expression_core import create_expression_intent
from conversation_questions import question_context

MAX_TURNS = 12
MAX_SOURCE_CHARS = 512
MAX_TOTAL_CHARS = 6000
_LOCATION = re.compile(r'^(?:in|on|under|above|below|beside|near|behind|inside|outside|at|next to|in front of)\b')
_PRONOUNS = {'i','you','he','she','it','we','they','me','him','her','us','them','my','your','our','their','his','its','this','that','these','those'}


def _label(value, limit=160):
    return value.strip() if isinstance(value,str) and 0 < len(value.strip()) <= limit else None


def _phrase(value):
    if isinstance(value,Mapping): value=value.get('surface')
    if not isinstance(value,str) or not 1 <= len(value) <= 192:
        return None
    if not re.fullmatch(r"[\w\s'’-]+",value): return None
    return ' '.join(value.casefold().split())


def _entity(value):
    if isinstance(value,Mapping):
        ref=_label(value.get('id'))
        return ('referent',ref) if ref and ref not in {'unknown','ambiguous'} else None
    phrase=_phrase(value)
    if not phrase or set(phrase.split()) & _PRONOUNS: return None
    return ('phrase',re.sub(r'^(?:the|a|an)\s+','',phrase))


def _argument(event):
    args=event.get('arguments') or {}
    if set(args)-{'theme'}: return None
    value=args.get('theme')
    if isinstance(value,Mapping) and value.get('alternatives'): return None
    return _phrase(value)


def _language_projection(text):
    """Keep a terminal heart/emoji trace while parsing its preceding sentence.

    No affection, willingness or semantic equivalence is inferred. Extra lexical
    text is never discarded, and the exact source stays in the observation.
    """
    last=max(text.rfind(mark) for mark in '.!?？')
    suffix=text[last+1:] if last>=0 else ''
    rest=suffix.replace('<3','')
    if suffix.strip() and len(suffix)<=24 and all(c.isspace() or unicodedata.category(c).startswith('S') or c in '\ufe0f\u200d' for c in rest):
        return text[:last+1],suffix
    return text,''


def _single_event(text, discourse):
    record=build_semantic_event(text,discourse)
    events=record.get('events') or []
    if record.get('incomplete') or len(events)!=1: return None,record
    event=events[0]
    if not event.get('predicate') or event.get('parent_clause') or event.get('alternatives') or event.get('voice')!='active': return None,record
    return event,record


def observe_conversation_turn(turn, discourse):
    """Make one source-linked observation; cached in the existing bounded scene."""
    base={'schema':'ina.conversation_observation/V1','status':'not_eligible',
          'truth_certified':False,'confidence_is_calibrated':False}
    if not isinstance(turn,Mapping) or not isinstance(discourse,Mapping): return base
    source=_label(turn.get('source_id'));text=turn.get('text')
    if (not source or not isinstance(text,str) or not 1<=len(text)<=MAX_SOURCE_CHARS
        or turn.get('text_complete') is not True or turn.get('text_single_line') is not True or turn.get('seeded')
        or turn.get('is_self') or turn.get('direction')!='inbound' or turn.get('author_is_bot')):
        return base
    # These sources need richer quotation/condition scope than this first route.
    if re.search(r'["“”`\n]|(?<!\w)[\x27‘]|\b(?:if|unless|whether|suppose|imagine)\b',text,re.I):
        return {**base,'status':'scope_unresolved'}
    repair=bool(re.match(r'^actually,\s+',text,re.I))
    parsed=re.sub(r'^actually,\s+','',text,flags=re.I) if repair else text
    parsed,suffix=_language_projection(parsed)
    if question_context(parsed)['answer_invitation']: return base
    event,record=_single_event(parsed,discourse)
    if event is None or record.get('speech_act',{}).get('act')!='assertion': return base
    agent=_entity(event.get('agent'));theme=_argument(event)
    if agent is None or not theme: return {**base,'status':'role_unresolved'}
    # Relations with unresolved pronoun objects cannot be answered lexically.
    if set(theme.split()) & _PRONOUNS: return {**base,'status':'role_unresolved'}
    speaker=discourse.get('speaker') or {}
    speaker_id=_label(speaker.get('id'));speaker_name=_label(speaker.get('name'),80)
    if not speaker_id or speaker_id=='unknown' or not speaker_name: return base
    payload={'agent_key':list(agent),'predicate':event['predicate'],'theme':theme,
             'tense':event.get('tense'),'modality':event.get('modality'),
             'negated':event.get('negated',False),'negation_operator':event.get('negation_operator')}
    return {**base,'status':'observed_claim','source_id':source,
            'backend_message_id':_label(turn.get('backend_message_id')),
            'speaker_id':speaker_id,'speaker_name':speaker_name,'source_text':text,
            'uninterpreted_suffix':suffix,
            'semantic_event':record,'native_intent':build_native_intent(record),
            'relation':payload,'fingerprint':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            'repair_requested':repair,'repair_target':_label(turn.get('referenced_message_id')) if repair else None}


def _query(text, discourse):
    context=question_context(text)
    if not context['answer_invitation'] or not context['complete']: return None
    if re.search(r'["“”`\n]|(?<!\w)[\x27‘]',text): return None
    # Only remove a supplied identity's comma-marked vocative, not arbitrary names.
    parsed,suffix=_language_projection(text)
    parsed=parsed.replace('？','?')
    for alias in (discourse.get('self') or {}).get('aliases',[])[:8]:
        if isinstance(alias,str) and len(alias)<=80:
            parsed=re.sub(r'^'+re.escape(alias)+r',\s+','',parsed,flags=re.I)
    event,record=_single_event(parsed,discourse)
    if event is None: return None
    if event.get('modality') or event.get('negation_operator') not in {None,'not'}: return None
    kind=event.get('question_word')
    agent=_entity(event.get('agent'));theme=_argument(event)
    if kind=='where' and event.get('negated'): return None
    if kind=='where' and event['predicate']=='be' and event.get('inverted'):
        # The parser supplies the complete subject; preserve its authoritative
        # referent rather than turning a resolved pronoun back into a word.
        subject=event.get('agent_surface')
        if theme or agent is None: return None
        slot='location'
    elif kind=='what' and event.get('inverted') and agent is not None and not (event.get('arguments') or {}): slot='theme'
    elif kind=='who' and theme and event.get('agent_surface')=='who': slot='agent';agent=None
    elif kind is None and event.get('inverted') and agent is not None and theme: slot='polarity'
    else: return None
    return {'slot':slot,'agent_key':list(agent) if agent else None,'predicate':event['predicate'],
            'theme':theme,'tense':event.get('tense'),'negated':event.get('negated',False),
            'subject_surface':subject if slot=='location' else event.get('agent_surface'),
            'source_analysis':record,'source_surface':text,'uninterpreted_suffix':suffix,'heuristic_only':True}


def _realise_attributed_claim(row, recipient_id, intent):
    """Compose a reported claim from native roles, retaining exact quote fallback."""
    from english_composition import compose_selected_english
    record=deepcopy(row['semantic_event'])
    event=record['events'][0]
    event['parent_clause']='report'
    event['clausal_relation']='reported_content'
    # Lexical perspective changes only for an already resolved source entity.
    # It never converts another speaker's claim into Ina's own endorsement.
    if isinstance(event.get('agent'),Mapping) and event['agent'].get('id')==recipient_id:
        event['agent']={**event['agent'],'name':'you'}
    reporter={'id':row['speaker_id'],'name':'you' if row['speaker_id']==recipient_id else row['speaker_name'],'is_self':False}
    root={'id':'report','predicate':'say','agent':reporter,'agent_surface':reporter['name'],
          'arguments':{},'tense':'past','modality':None,'negated':False,
          'negation_operator':None,'parent_clause':None,'clausal_relation':None,
          'embedded_content_is_fact':False,'voice':'active','question_word':None,
          'inverted':False,'alternatives':[]}
    record['events']=[root,event]
    native=build_native_intent(record)
    realised=compose_selected_english(native,intent)
    candidate=realised.get('candidate_text')
    if candidate:
        return candidate,native,{'status':'composed_attributed_claim','meaning_fidelity':'not_certified'}
    return row['speaker_name']+' said: “'+row['source_text']+'”',native,{
        'status':'exact_attributed_quote','composition_status':realised.get('status'),
        'meaning_fidelity':'quotation_not_verified_world_fact'}


def plan_conversation_reply(text, scene, *, discourse=None):
    """Offer source-backed answer/clarification candidates for the existing chooser.

    Supported: simple where/copula, what/object, who/subject and polar questions.
    No lexical definition, sentiment or preference is invented from repetition.
    """
    result={'schema':'ina.conversation_reply_plan/V1','status':'not_assessed',
            'answer_candidate':None,'clarification_candidate':None,'query':None,
            'evidence':[],'repairs':[],'alternatives':[],'choice_required':True,
            'truth_certified':False,'automatic_learning':False,'outward_text':None,
            'choices':['respond','clarify','mirror','native_only','defer','silence'],
            'continuation_budget':0,'participant_transfer':'not_assessed',
            'evidence_scope':'recent_channel_claims_only','independence_inferred':False}
    if not isinstance(text,str) or len(text)>MAX_SOURCE_CHARS or not isinstance(scene,Mapping): return {**result,'status':'input_budget_exceeded'}
    frame=discourse if isinstance(discourse,Mapping) else scene.get('discourse')
    if not isinstance(frame,Mapping): return {**result,'status':'missing_discourse'}
    query=_query(text,frame);result['query']=query
    if query is None: return {**result,'status':'unsupported_question'}
    turns=scene.get('turns') or []
    if not isinstance(turns,(list,tuple)) or len(turns)>MAX_TURNS: return {**result,'status':'scene_budget_exceeded'}
    current=scene.get('current_turn_id');total=0;rows=[];seen={};invalid=set()
    for turn in turns:
        if not isinstance(turn,Mapping): continue
        raw=turn.get('text')
        total+=len(raw) if isinstance(raw,str) else MAX_TOTAL_CHARS+1
        if total>MAX_TOTAL_CHARS:return {**result,'status':'scene_budget_exceeded'}
        if turn.get('source_id')==current:continue
        obs=turn.get('conversation_observation')
        if not isinstance(obs,Mapping) or obs.get('status')!='observed_claim':continue
        # Cache is produced by observe(); copied/private callers are not trusted
        # authorities. Revalidate the complete source before offering a reply.
        prior_frame=turn.get('discourse')
        checked=observe_conversation_turn(turn,prior_frame)
        if checked!=obs:continue
        source=obs['source_id']
        if source in seen:
            if seen[source]['fingerprint']!=obs['fingerprint'] or seen[source]['source_text']!=obs['source_text'] or seen[source]['speaker_id']!=obs['speaker_id']:invalid.add(source)
            continue
        seen[source]=obs;rows.append(obs)
    rows=[o for o in rows if o['source_id'] not in invalid]
    # Only an explicit linked correction by the same speaker for the same
    # entity/predicate/tense supersedes a claim. Originals remain in the scene.
    retired=set()
    for row in rows:
        if not row['repair_requested'] or not row['repair_target']:continue
        for old in rows:
            if old is row:break
            same=all(row['relation'][k]==old['relation'][k] for k in ('agent_key','predicate','tense'))
            if row['repair_target'] in {old['source_id'],old.get('backend_message_id')} and same and row['speaker_id']==old['speaker_id']:
                retired.add(old['source_id']);result['repairs'].append({'original':old['source_id'],'revision':row['source_id'],'truth_certified':False})
    matched=[]
    for row in rows:
        if row['source_id'] in retired:continue
        rel=row['relation']
        if rel['predicate']!=query['predicate'] or rel['tense']!=query['tense']:continue
        if query['agent_key'] is not None and rel['agent_key']!=query['agent_key']:continue
        if query['slot'] in {'agent','polarity'} and rel['theme']!=query['theme']:continue
        if query['slot']=='location' and not _LOCATION.match(rel['theme']):continue
        matched.append(row)
    result['evidence']=[{'source_id':r['source_id'],'speaker_id':r['speaker_id'],'source_text':r['source_text'],
                         'relation':dict(r['relation']),'truth_certified':False} for r in matched[:8]]
    if len(matched)>8:return {**result,'status':'evidence_budget_exceeded'}
    # Distinct positive objects/people may all be valid. A location change or
    # opposing polarity is unresolved unless an explicit repair linked them.
    signs_by_binding={}
    for row in matched:
        rel=row['relation']
        binding=(tuple(rel['agent_key']),rel['predicate'],rel['theme'],rel['tense'])
        signs_by_binding.setdefault(binding,set()).add(rel['negated'])
    opposed=any(len(signs)>1 for signs in signs_by_binding.values())
    locations={r['relation']['theme'] for r in matched if not r['relation']['negated']}
    differing_locations=query['slot']=='location' and len(locations)>1
    conflict=opposed or differing_locations
    usable=matched if query['slot']=='polarity' else [r for r in matched if r['relation']['negated']==query['negated']]
    if conflict:
        result['alternatives']=result['evidence']
        result['status']='conflicting_claims' if opposed else 'unresolved_claims'
    elif usable:
        # Identical semantic payloads are one reading, not extra confidence.
        unique={}
        for row in usable:unique.setdefault(row['fingerprint'],row)
        selected=list(unique.values())[:3]
        if len(unique)>3:return {**result,'status':'answer_budget_exceeded'}
        references=['conversation:'+r['source_id'] for r in selected]
        intent=create_expression_intent('answer_from_attributed_conversation',
            semantic_references=references,provenance=references,allowed_media=['text'],
            uncertainty={'truth':'unverified_speaker_claim','independence':'not_established'})
        recipient=(frame.get('speaker') or {}).get('id')
        realisations=[_realise_attributed_claim(r,recipient,intent) for r in selected]
        candidate='\n'.join(item[0] for item in realisations)
        result.update(status='answer_available',answer_candidate=candidate,expression_intent=intent,
                      selected_native_intents=[item[1] for item in realisations],
                      realisations=[item[2] for item in realisations],
                      selected_source_ids=[r['source_id'] for r in selected])
        return result
    else:result['status']='missing_evidence'
    if query['slot']=='location' and query['agent_key'][0]=='phrase' and _phrase(query['subject_surface']):
        result['clarification_candidate']='Could you clarify the location of “'+query['subject_surface']+'”?'
    elif conflict:
        result['clarification_candidate']='Could you clarify which of those earlier statements applies?'
    return result


def english_candidate_allowed(preference):
    """English, mixed or existing auto mode may offer an attributed quote."""
    if isinstance(preference,Mapping):
        if isinstance(preference.get('mutual_intelligibility_assessment'),Mapping):return False
        preference=preference.get('mode') or preference.get('language_mode') or preference.get('preference')
    return isinstance(preference,str) and preference.strip().casefold().replace('-','_') in {'english','en','english_only','auto','mixed'}
