"""Synthetic name-reference and selection checks; no identity learning or Discord."""
from __future__ import annotations
import inspect
from pathlib import Path
BASELINE='6d200475d5b444550819f7c23a1e595518240cf6'


def assess(version):
    from identity_language import identity_aliases
    from benchmarks.benchmark_response_language_policy import functions
    from benchmarks.benchmark_discord_social_english import route_fixture
    if version=='V1':
        from historical_source import historical_module,historical_text
        build=historical_module('discourse_context.py',BASELINE).build_discourse_context
        source=historical_text('discord_bridge.py',BASELINE)
    else:
        from discourse_context import build_discourse_context as build
        source=Path(__file__).resolve().parents[1].joinpath('discord_bridge.py').read_text()
    select=functions(source);supports='self_aliases' in inspect.signature(build).parameters
    cases=[]
    def add(name,component,correct):cases.append({'case':name,'component':component,'correct':bool(correct)})
    def scene(text,*,listener=None,mentioned=(),aliases=('Ina','Inazuma')):
        kw={'speaker':{'id':'human','name':'Mira'},'addressee':listener if listener is not None else {'id':'child','name':'Inazuma','is_self':True},
            'self_identity':{'id':'child','name':'Inazuma_Yagami','is_self':True},'mentioned_entities':mentioned}
        if supports:kw['self_aliases']=aliases
        return build(text,**kw)
    def named(ctx,surface):return next((r for r in ctx.get('resolutions',[]) if r.get('surface')==surface.casefold()),{})
    for alias in ('Ina','Inazuma','INA'):
        c=scene(f'Morning, {alias}. <3');row=named(c,alias)
        add('name links to self '+alias,'identity',len(row.get('referents',[]))==1 and row['referents'][0]['id']=='self')
        add('greeting is direct address '+alias,'address',c.get('self_name_context',{}).get('direct_self_address') is True)
    c=scene('Are you okay, Ina?',listener='unknown')
    add('vocative resolves previously unknown listener','address',c['addressee']['id']=='self' and named(c,'you').get('referents',[{}])[0].get('id')=='self')
    c=scene('Hello, Ina.',listener={'id':'other','name':'Other'})
    add('explicit conflicting listener is not overwritten','uncertainty',c['addressee']['id']=='other' and c.get('name_reference_conflict') is True)
    c=scene('Morning, Ina.',mentioned=[{'id':'other-ina','name':'Ina'}]);row=named(c,'ina')
    add('same-name collision remains ambiguous','uncertainty',row.get('ambiguous') is True and len(row.get('referents',[]))==2)
    for text in ('The word "Ina" has three letters.', '"Morning, Ina."', "'Ina' is your name."):
        c=scene(text);row=named(c,'ina')
        add('quoted mention has no direct self attribution '+text,'quotation',not row.get('referents') and not c.get('self_name_context',{}).get('direct_self_address'))
    add('substring Mina does not resolve as Ina','identity',not named(scene('Morning, Mina.'),'ina'))
    add('disabled aliases remain disabled','agency',not named(scene('Morning, Ina.',aliases=()),'ina'))
    for text in ('Ina is your name, silly. Giggles','Your nickname is Ina.'):
        c=scene(text)
        add('naming observation recognised '+text,'naming',c.get('self_name_context',{}).get('self_name_statement') is True)
    c=scene('Your name is Bob.')
    add('utterance does not invent a new alias','agency',not named(c,'bob') and c.get('self',{}).get('name')=='Inazuma_Yagami')
    incoming=scene('Ina is your name.');outgoing=build('I know your name.',speaker={'id':'child','name':'Ina','is_self':True},addressee={'id':'human','name':'Mira'},self_identity={'id':'child','name':'Ina','is_self':True})
    add('incoming your and outgoing your retain different roles','discourse',named(incoming,'your').get('referents',[{}])[0].get('id')=='self' and named(outgoing,'your').get('referents',[{}])[0].get('id')=='human')
    def choose(state=None,adapter=False):
        kw=dict(mapped_count=4,token_count=4,adapter_available=adapter,expression_drive=.8,emotion_available=False,code_pointer_available=False)
        if 'name_context' in inspect.signature(select).parameters:kw['name_context']={'self_name_statement':True}
        return select(state or {},**kw)
    add('name correction does not default to an echo','selection',choose()['strategy']=='silence')
    add('explicit mirroring remains available','agency',choose({'text_expression_intent':{'strategy':'mirror'}})['strategy']=='mirror')
    add('explicit silence remains available','agency',choose({'text_expression_intent':{'strategy':'silence'}},True)['strategy']=='silence')
    add('grounded response still preferred when available','selection',choose(adapter=True)['strategy']=='respond')
    result,captured,_=route_fixture(source,text='Ina is your name.',identity_aliases=['Ina','Inazuma'],draft='Perhaps.')
    supplied=(captured.get('context') or {}).get('discourse') or {}
    add('bridge passes names into symbolic discourse','wiring',named(supplied,'ina').get('referents',[{}])[0].get('id')=='self')
    add('selected response wording is not replaced by a canned name reply','agency',result.text=='Perhaps.')
    add('alias entries bounded','bounds',len(identity_aliases(['N'+str(i) for i in range(20)]))==8)
    add('alias spelling is not inferred from prefixes','bounds',identity_aliases(['Ina','Ina','<invalid>','x'*81])==['Ina'])
    c=scene('"" '*129+'Morning, Ina.')
    add('quote budget leaves interpretation unresolved','bounds',c.get('self_name_context',{}).get('quotes_complete') is False and not c.get('self_name_context',{}).get('direct_self_address'))
    c=scene(' '*8189+'Inactive')
    add('clipped prefix is not a completed nickname','bounds',not named(c,'ina') and c.get('self_name_context',{}).get('source_complete') is False)
    return {'correct':sum(r['correct'] for r in cases),'total':len(cases),'cases':cases,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'synthetic identity-reference, selection and copied route wiring; not learned self-recognition',
            'live_delivery':'not_assessed','name_preference_inferred':False}
