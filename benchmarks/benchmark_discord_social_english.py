"""Pinned Discord route comparison with pure collaborators, no client or files.

The real process_inbound_message body is extracted from source. Its storage,
learning, adapter and rendering collaborators are synthetic: this measures route
wiring and English support, not Discord delivery or Ina's live comprehension.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from types import SimpleNamespace

BASELINE = '5273e488040a3aabc2695da592a526b806317723'


def route_fixture(source, *, text='I enjoy your song.', draft='I might share my song.',
                  strategy='respond', preference='auto', scene=True, dm=False,
                  support=None, disabled=False, low_urge=False, identity_aliases=None):
    from discourse_context import build_discourse_context
    from discord_english_support import discord_english_support
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_inbound_message')
    captured={}
    incoming=build_discourse_context(text,speaker='Mira',addressee='Ina',self_identity='Ina')
    message=SimpleNamespace(text=text,metadata={'conversation_scene':{'discourse':incoming} if scene else {},'is_dm':dm},
        sender=SimpleNamespace(backend_id='mira',display_name='Mira',internal_id='mira'),
        channel=SimpleNamespace(name='fixture',backend_id='fixture'))
    class NoFile:
        def __truediv__(self, part):return self
        def exists(self):return False
    def symbolic(*args,**kwargs):
        captured['context']=kwargs.get('context')
        if strategy=='mirror':return {'text':text,'symbols':['fixture'],'unknown':[]}
        return None
    adapter=SimpleNamespace(has_constructive_reply=lambda text:True,handle_prompt=lambda *a,**k:draft)
    def encode(text,**kwargs):
        captured['encoding']=kwargs
        return (('∘⊙' if preference=='native_only' else text),{'fixture_encoding':True})
    env={
        'CommsResponse':lambda **kwargs:SimpleNamespace(**kwargs),
        'get_discord_config':lambda:{'allow_replies':not disabled,'language_mode':preference,'identity_aliases':{'Ina':identity_aliases or []}},
        'get_current_child':lambda:'Ina','_format_image_attachment_note':lambda a:'',
        '_collect_vision_context':lambda a:{},'load_root_config':lambda:{'ignore_urge_for_typing':not low_urge},
        'Path':lambda p:NoFile(),'_resolve_adjusted_urge_level':lambda s:0 if low_urge else 1,
        'get_chat_adapter':lambda:adapter,'logger':logging.getLogger('fixture'),
        '_extract_tokens':lambda t:t.split(),'generate_symbolic_reply_from_text':symbolic,
        '_coerce_positive_int':lambda v,d:d,'DEFAULT_MAX_REPLY_SYMBOLS':64,
        'select_symbolic_message_text':lambda s,p:(s['text'] if s else None,'english'),
        'build_conversation_examples':lambda *a,**k:{},
        'interpret_communicative_meaning':lambda *a,**k:{},
        'plan_experience_cognition':lambda *a,**k:{},'build_expression_cognition_event':lambda a:{},
        'format_emotion_signal':lambda s:None,'format_code_pointer_signal':lambda s:None,
        'resolve_song_expression_candidate':lambda *a,**k:{'available':False},
        'choose_text_expression_strategy':lambda *a,**k:{'strategy':strategy},
        'datetime':datetime,'timezone':timezone,'update_inastate':lambda *a,**k:None,
        'encode_selected_text_expression':encode,
        'discord_english_support':support or discord_english_support,
    }
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied Discord route>','exec'),env)
    started=time.perf_counter()
    result=env['process_inbound_message'](message)
    return result,captured,time.perf_counter()-started


def sources(version):
    if version=='V1':
        from historical_source import historical_text
        return historical_text('discord_bridge.py',BASELINE)
    return Path(__file__).resolve().parents[1].joinpath('discord_bridge.py').read_text()


def assess(version):
    source=sources(version);rows=[];times=[]
    def add(name,component,correct):rows.append({'case':name,'component':component,'correct':bool(correct)})
    r,c,t=route_fixture(source);times.append(t)
    support=r.metadata.get('social_english_support') or {}
    review=r.metadata.get('social_english_draft_review') or {}
    opts=support.get('options') or {}
    add('both voluntary opportunities available in symbolic context','availability',
        len(opts.get('opportunities',[]))==2 and c['context'].get('social_english_support')==support)
    add('incoming speaker and listener authoritative','discourse',_resolved(support,'incoming')=={'i':'mira','your':'ina'})
    add('reply uses Ina as speaker','discourse',_resolved(review,'draft')=={'i':'self','my':'self'})
    add('selected draft preserved exactly','preservation',r.text=='I might share my song.')
    add('review not a paraphrase or meaning certificate','uncertainty',
        ((review.get('review') or {}).get('structural_comparison') or {}).get('status')=='not_assessed' and (review.get('review') or {}).get('meaning_fidelity')=='not_certified')
    add('opportunity is not inferred willingness or learning','agency',
        support.get('willingness_inferred') is False and support.get('automatic_learning') is False)
    for strategy in ('silence','mirror'):
        rr,cc,tt=route_fixture(source,strategy=strategy);times.append(tt)
        add(strategy+' output preserved','preservation',rr.text==(None if strategy=='silence' else 'I enjoy your song.'))
        if strategy=='mirror':add('mirror retains original speaker','discourse',_resolved(rr.metadata.get('social_english_draft_review') or {},'draft')=={'i':'mira','your':'ina'})
    rr,cc,tt=route_fixture(source,preference='native_only');times.append(tt)
    add('native preference preserved','agency',rr.text=='∘⊙')
    add('native draft receives no English coaching','agency',(rr.metadata.get('social_english_draft_review') or {}).get('status')=='choice_preserved')
    rr,cc,tt=route_fixture(source,text='x'*8193);times.append(tt)
    add('over-budget analysis deferred without blocking reply','bounds',rr.text=='I might share my song.' and (rr.metadata.get('social_english_support') or {}).get('status')=='input_budget_exceeded')
    rr,cc,tt=route_fixture(source,text='You enjoy this.',scene=False);times.append(tt)
    refs=_resolved(rr.metadata.get('social_english_support') or {},'incoming')
    add('guild addressee not invented','uncertainty','you' in refs and refs['you'] in {None,'unknown'})
    rr,cc,tt=route_fixture(source,text='You enjoy this.',scene=False,dm=True);times.append(tt)
    add('DM listener resolves to Ina','discourse',_resolved(rr.metadata.get('social_english_support') or {},'incoming').get('you')=='self')
    for kwargs,name in (({'disabled':True},'disabled route'),({'low_urge':True},'urge gate')):
        rr,cc,tt=route_fixture(source,**kwargs);times.append(tt)
        add(name+' still silent before support','agency',rr.text is None and not cc)
    result={'cases':rows,'correct':sum(x['correct'] for x in rows),'total':len(rows),
        'measurement_scope':'extracted real route; synthetic collaborators; no Discord client or live files',
        'route_elapsed_seconds':times,'live_delivery':'not_assessed','listener_understanding':'not_assessed'}
    if version=='V1':result['baseline_revision']=BASELINE
    return result


def _resolved(support,role):
    analysis=((support.get('review') or {}).get(role+'_analysis') or {})
    return {r['surface']:r['resolved'] for r in analysis.get('referents',[])}


def v1():return assess('V1')
def v2():return assess('V2')
