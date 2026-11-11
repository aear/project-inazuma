"""Pinned chooser/source-route checks; no generation, storage or live Discord."""
from pathlib import Path
BASELINE='71b80f8f011373ff759afec79931094d31355df0'


def assess(version):
    from benchmarks.benchmark_response_language_policy import functions
    from benchmarks.benchmark_discord_social_english import route_fixture
    if version=='V1':
        from historical_source import historical_text
        source=historical_text('discord_bridge.py',BASELINE)
    else:source=Path('discord_bridge.py').read_text()
    choose=functions(source);rows=[]
    native={'native_text':'∘⊙ λ','meaning_references':['meaning:fixture']}
    def add(name,component,ok):rows.append({'case':name,'component':component,'correct':bool(ok)})
    def select(state=None,adapter=False,text='Where is it?',preference='english'):
        return choose(state or {},mapped_count=4,token_count=4,adapter_available=adapter,expression_drive=.8,
            emotion_available=True,code_pointer_available=False,incoming_text=text,language_preference=preference)
    p=select({'text_expression_intent':native});trace=p.get('response_passes',{})
    add('independent native expression after unavailable English','native',p['strategy']=='native' and len(trace.get('passes',[]))==2)
    add('English candidate remains first','order',select({'text_expression_intent':native},True)['strategy']=='respond')
    p=select({'text_expression_intent':{**native,'strategy':'silence'}},True)
    add('explicit silence is final','agency',p['strategy']=='silence' and p['reason']=='explicit_intent')
    for strategy in ('mirror','respond','emotion'):
        p=select({'text_expression_intent':{**native,'strategy':strategy}},True)
        add('explicit '+strategy,'agency',p['strategy']==strategy and p['reason']=='explicit_intent')
    p=select({'text_expression_intent':native},True,preference='native_only')
    add('native language choice skips English','agency',p['strategy']=='native')
    for preference in ('abstain','silence'):
        add('protected '+preference,'agency',select({'text_expression_intent':native},True,preference=preference)['strategy']=='silence')
    p=select(text='How are you feeling?')
    add('direct check-in retains measured report','relevance',p['strategy']=='emotion')
    p=select(text='Explain this code.')
    add('failed English does not license unrelated emotional report','relevance',p['scores']['emotion'] is None and p['strategy']!='emotion')
    p=select({'text_expression_intent':{'native_text':'∘⊙'}},text='Explain this code.')
    add('unreferenced native text is not manufactured into a reply','provenance',p['strategy']!='native')
    add('inbound mirror remains separately labelled','provenance',select(text='Hello there.')['strategy']!='native')
    result,_,_=route_fixture(source,text='Where is it?',real_strategy=True,adapter_available=False,state={'text_expression_intent':native})
    add('real processing route returns independent native without re-encoding','wiring',result.text=='∘⊙ λ' and result.metadata.get('adapter')=='selected_native_expression')
    p=select({'text_expression_intent':native})
    trace=p.get('response_passes',{})
    add('no added penalty for silence','bounds',trace.get('silence_penalty_added') is False)
    add('at most three passes','bounds',trace.get('maximum_passes')==3 and len(trace.get('passes',[]))<=3)
    if version=='V1':
        from historical_source import historical_text
        runtime_source=historical_text('runtime_state.py',BASELINE)
        tools_source=historical_text('personal_tool_runtime.py',BASELINE)
    else:
        runtime_source=Path('runtime_state.py').read_text();tools_source=Path('personal_tool_runtime.py').read_text()
    import ast
    from datetime import datetime,timezone
    from typing import Optional,List,Any,Dict
    node=next(n for n in ast.parse(runtime_source).body if isinstance(n,ast.FunctionDef) and n.name=='set_text_expression_intent')
    captured={}
    env={'Optional':Optional,'List':List,'Any':Any,'Dict':Dict,'datetime':datetime,'timezone':timezone,
         'update_inastate':lambda k,v,**kw:captured.update(value=v)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied choice setter>','exec'),env)
    try:
        value=env['set_text_expression_intent']('native',native_text='λ',meaning_references=['meaning:one'])
        valid=value.get('strategy')=='native' and captured.get('value')==value
    except (TypeError,ValueError):valid=False
    add('existing choice API supports referenced native expression','tools',valid)
    add('private voluntary choice tool is discoverable','tools','"text_expression_choice"' in tools_source)
    if version=='V1':
        from historical_source import historical_module
        emotion_context=historical_module('response_language_policy.py',BASELINE).emotion_response_context
    else:
        from response_language_policy import emotion_response_context as emotion_context
    add('direct check-in with heart remains eligible','relevance',emotion_context('How are you? <3',{})['direct_check_in'])
    add('quoted check-in with heart is not a state request','relevance',not emotion_context('"How are you?" <3',{})['eligible'])
    try:
        response,_,_=route_fixture(source,text='Where is it?',real_strategy=True,adapter_available=False,
            symbolic_error=OSError('fixture'),state={'text_expression_intent':native})
        survived=response.text=='∘⊙ λ' and response.metadata.get('symbolic_comprehension',{}).get('status')=='unavailable'
    except OSError:survived=False
    add('failed vocabulary read leaves independent expression available','resilience',survived)
    return {'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'copied chooser and processing route, synthetic own-intent references; not observed Ina preference or live delivery'}
