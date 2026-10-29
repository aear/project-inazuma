"""Pinned policy comparison; pure source extraction, no runtime or delivery."""
from __future__ import annotations
import ast, inspect, math
from pathlib import Path
from typing import Any, Dict

BASELINE = 'b46dc1eca627d0d21d7fcadbe05937bfb5e5aeb1'


def functions(source):
    root = Path(__file__).resolve().parents[1]
    score = next(n for n in ast.parse((root/'language_processing.py').read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'score_text_mirroring')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in {'choose_text_expression_strategy', '_emotion_values_from_state'}]
    env = {'math':math, 'Dict':Dict, 'Any':Any}
    exec(compile(ast.Module(body=[score,*nodes],type_ignores=[]),'<copied policy source>','exec'),env)
    return env['choose_text_expression_strategy']


def assess(version):
    from response_language_policy import resolve_response_language_preference, prefers_english, emotion_response_context
    root = Path(__file__).resolve().parents[1]
    if version == 'V1':
        from historical_source import historical_text
        source = historical_text('discord_bridge.py', BASELINE)
    else:source = (root/'discord_bridge.py').read_text()
    select = functions(source);new = 'incoming_text' in inspect.signature(select).parameters
    cases=[]
    def add(name, component, correct):cases.append({'case':name,'component':component,'correct':bool(correct)})
    def choose(text='', state=None, adapter=False, preference='english'):
        kw=dict(mapped_count=0,token_count=4,adapter_available=adapter,expression_drive=.8,
                emotion_available=True,code_pointer_available=False,
                conversation_scene={'signals':{'reply_expected':True}})
        if new:kw.update(incoming_text=text,language_preference=preference)
        return select(state or {'emotion_snapshot':{'values':{'clarity':-.9}}},**kw)
    for text in ('How are you feeling?', 'How do you feel about that?', 'Ina, are you okay?',
                 'Tell me about your feelings.', "What's your mood?", 'Show me your emotion sliders.'):
        row=choose(text,adapter=True)
        add(text,'check_in',row['strategy']=='emotion')
    for text in ('Can you explain this code?', 'Better?', 'I feel sad.',
                 'How is Mira feeling?', 'What does "how are you feeling" mean?',
                 'Do not tell me your feelings.', '"How are you feeling?"',
                 'Are you happy with the build configuration?', 'Show me your emotion sliders in this example.'):
        row=choose(text)
        add(text,'relevance',row['scores']['emotion'] is None)
    for strategy in ('silence','emotion','mirror','code_pointer','song','respond'):
        state={'text_expression_intent':{'strategy':strategy}}
        kw=dict(mapped_count=4,token_count=4,adapter_available=True,expression_drive=.8,
                emotion_available=True,code_pointer_available=True,song_available=True)
        if new:kw.update(incoming_text='Explain this code.',language_preference='english')
        row=select(state,**kw)
        add('explicit '+strategy,'agency',row['strategy']==strategy and row['reason']=='explicit_intent')
    native=choose('Explain this.',adapter=True,preference='native_only')
    english=choose('Explain this.',adapter=True)
    add('small grounded English response preference','weighting',english['scores']['respond']>native['scores']['respond'])
    add('no response created without constructive adapter','grounding',choose('Explain this.')['scores']['respond'] is None)
    # Compare source-wired defaults rather than crediting V1 with a new helper.
    add('English operator default wired','default',new and 'resolve_response_language_preference(state, cfg)' in source
        and resolve_response_language_preference({}, {})=='english')
    for preference in ('native_only','mixed','auto','clarify','abstain'):
        add('preserves language '+preference,'agency',resolve_response_language_preference(
            {'discord_language_preference':preference},{'language_mode':'english'})==preference)
    add('bounded emotion inquiry','bounds',not emotion_response_context('x'*2049,{})['eligible'])
    add('English renderer puts selected expression first','presentation',
        'if requested_mode == "english":\n            layers.append(f"English expression: {selected_text}")' in source)
    # Exercise the exact encoder, with anonymous mappings and no generation/files.
    from benchmarks.benchmark_selected_expression_resilience import load_rendering_functions
    from tempfile import TemporaryDirectory
    from typing import Optional
    selector_node = next(n for n in ast.parse((root/'language_processing.py').read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name=='select_symbolic_message_text')
    selector_env={'Optional':Optional,'Dict':Dict,'Any':Any}
    exec(compile(ast.Module(body=[selector_node],type_ignores=[]),'<copied selector>','exec'),selector_env)
    with TemporaryDirectory() as temp:
        path=Path(temp)/'bridge.py';path.write_text(source)
        renderer=load_rendering_functions(path)
        renderer.select_symbolic_message_text=selector_env['select_symbolic_message_text']
        renderer._is_emotion_or_sound_symbol=lambda symbol:False
        renderer.generate_symbolic_reply_from_text=lambda *a,**k:{'symbols':['s'],'unknown':[]}
        renderer.build_dual_symbolic_message=lambda *a,**k:{'native_text':'∘⊙','native_tokens':['∘⊙'],
            'gloss_text':'perhaps','gloss_tokens':['perhaps'],'native_sources':{'s':'fixture'},'gloss_sources':{'s':'fixture'}}
        text,meta=renderer.encode_selected_text_expression('perhaps',child='fixture',language_preference='english',max_symbols=8)
        add('complete selected English leads without deleting native evidence','presentation',
            text.startswith('English expression: perhaps') and 'Native: ∘⊙' in text)
        text,meta=renderer.encode_selected_text_expression('perhaps',child='fixture',language_preference='native_only',max_symbols=8)
        add('selected native language remains available','agency',text=='∘⊙')
        renderer.generate_symbolic_reply_from_text=lambda *a,**k:None
        for preference in ('native_only','abstain','clarify'):
            text,meta=renderer.encode_selected_text_expression('perhaps',child='fixture',language_preference=preference,max_symbols=8)
            add('empty encoding preserves '+preference,'agency',text=='')
        # No distinct English gloss must still leave native fallback available.
        text,mode=selector_env['select_symbolic_message_text']({'native_text':'∘⊙'},'english')
        add('no invented English for a native-only signal','grounding',text=='∘⊙' and mode=='unresolved')
    result={'correct':sum(row['correct'] for row in cases),'total':len(cases),'cases':cases,
            'scope':'copied real selection functions and source wiring; synthetic language/state; no live Ina assessment',
            'live_delivery':'not_assessed','comprehension_improvement':'not_assessed'}
    if version=='V1':result['baseline_revision']=BASELINE
    return result
