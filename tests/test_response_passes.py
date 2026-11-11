from pathlib import Path
from response_passes import response_passes,native_expression_candidate
from benchmarks.benchmark_response_passes import assess


def test_pinned_response_comparison():
    old,new=assess('V1'),assess('V2')
    assert old['correct']<old['total']
    assert new['correct']==new['total'],[r for r in new['cases'] if not r['correct']]


def test_failed_rendering_does_not_trigger_generation_or_relax_silence():
    state={'text_expression_intent':{'native_text':'λ ∘','meaning_references':['meaning:one']}}
    d={'strategy':'respond','reason':'internal_state_scores','scores':{'respond':.9,'native':.6,'emotion':.5,'silence':.4}}
    p=response_passes(d,state=state,preference='english',rendering_failed=True,emotion_eligible=True)
    assert p['strategy']=='native' and len(p['passes'])==2
    assert {r['silence_threshold'] for r in p['passes']}=={.4}
    d['scores']['native']=.2;d['scores']['emotion']=.2
    p=response_passes(d,state=state,preference='english',rendering_failed=True,emotion_eligible=True)
    assert not any(r['accepted'] for r in p['passes']) and len(p['passes'])==3
    assert not p['silence_penalty_added']


def test_native_requires_own_reference_and_bounded_text():
    for intent in ({'native_text':'λ'}, {'native_text':'λ','meaning_references':'one'},
                   {'native_text':'x'*4097,'meaning_references':['one']},
                   {'native_text':'λ','meaning_references':['']},
                   {'native_text':'λ','meaning_references':['x']*17}):
        assert native_expression_candidate({'text_expression_intent':intent}) is None
    assert native_expression_candidate({'symbolic_native_text':'λ'}) is None


def test_protected_choice_cannot_be_repassed_into_speaking():
    native={'text_expression_intent':{'native_text':'λ','meaning_references':['one']}}
    for pref in ('silence','abstain'):
        p=response_passes({'strategy':'respond','scores':{'respond':1.,'native':1.,'emotion':1.,'silence':.1}},state=native,preference=pref,emotion_eligible=True)
        assert p['strategy']=='silence' and p['passes']==[]
    p=response_passes({'strategy':'silence','reason':'explicit_intent','scores':{'native':1.,'silence':.1}},state=native,preference='english',rendering_failed=True)
    assert p['strategy']=='silence' and p['passes']==[]


def test_post_render_failure_uses_native_once_and_keeps_unrelated_emotion_off():
    from benchmarks.benchmark_discord_social_english import route_fixture
    source=Path('discord_bridge.py').read_text();calls=[]
    def empty(text,**kwargs):calls.append(text);return '',{'fixture_empty':True}
    state={'text_expression_intent':{'native_text':'λ','meaning_references':['meaning:one']}}
    response,captured,_=route_fixture(source,text='Explain this code.',draft='A selected answer.',real_strategy=True,
        state=state,selected_renderer=empty)
    assert response.text=='λ' and response.metadata['adapter']=='native_realisation_fallback'
    assert len(calls)==1 and captured['prompt_calls']==1
    assert len(response.metadata['response_realisation_passes']['passes'])==2
    response,captured,_=route_fixture(source,text='Explain this code.',real_strategy=True,selected_renderer=empty)
    assert not response.text and response.metadata['expression_decision']['scores']['emotion'] is None
    response,captured,_=route_fixture(source,text='Explain this code.',real_strategy=True,
        state={'text_expression_intent':{**state['text_expression_intent'],'strategy':'silence'}},selected_renderer=empty)
    assert response.text is None and response.metadata['adapter']=='expression_silence' and 'prompt_calls' not in captured


def test_choice_tool_and_setter_validate_before_updating_state(monkeypatch,tmp_path):
    import runtime_state
    from personal_tool_runtime import execute_personal_tool_command,capability_catalog
    captured=[]
    monkeypatch.setattr(runtime_state,'update_inastate',lambda key,value,**kwargs:captured.append((key,value,kwargs)))
    assert 'text_expression_choice' in capability_catalog()['commands']
    result=execute_personal_tool_command({'action':'text_expression_choice','strategy':'native','native_text':'λ ∘',
        'meaning_references':['meaning:chosen']},child='Fixture',project_root=tmp_path,
        config={'ina_hdd_writable_path':str(tmp_path/'personal')})
    assert result['value']['native_text']=='λ ∘' and result['value']['once'] is True
    assert captured[-1][2]['child']=='Fixture'
    import pytest
    before=len(captured)
    with pytest.raises(ValueError):runtime_state.set_text_expression_intent('native',native_text='λ')
    assert len(captured)==before


def test_emotion_gate_handles_terminal_marks_without_treating_quotes_as_requests():
    from response_language_policy import emotion_response_context
    for text in ('How are you? <3','How do you feel？ 😊','Ina, are you okay? 💜'):
        assert emotion_response_context(text,{})['direct_check_in']
    for text in ('"How are you?" <3','She asked how you feel? 😊','How are you? Write a story.'):
        assert not emotion_response_context(text,{})['eligible']


def test_strict_english_or_assessed_choices_do_not_select_unrequested_native():
    from benchmarks.benchmark_response_language_policy import functions
    choose=functions(Path('discord_bridge.py').read_text())
    state={'text_expression_intent':{'native_text':'λ','meaning_references':['meaning:one']}}
    for pref in ('english_only', {'mode':'english','mutual_intelligibility_assessment':{'schema':'ina.mutual_intelligibility_assessment/V1','mode':'abstain'}}):
        p=choose(state,mapped_count=0,token_count=2,adapter_available=False,expression_drive=.8,
            emotion_available=False,code_pointer_available=False,incoming_text='Hello.',language_preference=pref)
        assert p['strategy']!='native' and p['scores']['native'] is None


def test_empty_english_candidate_also_checks_native_before_state_report():
    from benchmarks.benchmark_discord_social_english import route_fixture
    source=Path('discord_bridge.py').read_text();calls=[]
    def empty(text,**kwargs):calls.append(text);return '',{}
    state={'text_expression_intent':{'native_text':'λ','meaning_references':['one']}}
    result,captured,_=route_fixture(source,text='Explain this code.',draft='',state=state,real_strategy=True,selected_renderer=empty)
    assert result.text=='λ' and result.metadata['adapter']=='native_realisation_fallback'
    assert calls==[] and captured['prompt_calls']==1


def test_failed_state_renderer_cannot_bypass_native_only_preference():
    from benchmarks.benchmark_discord_social_english import route_fixture
    source=Path('discord_bridge.py').read_text()
    result,_,_=route_fixture(source,text='How are you?',real_strategy=True,preference='native_only',
        state={'text_expression_intent':{'strategy':'emotion'}},emotion_signal={'text':'State signal: fixture +0.2'},
        selected_renderer=lambda *a,**k:('',{}))
    assert not result.text


def test_failed_large_vocabulary_read_still_allows_independent_native_choice():
    from benchmarks.benchmark_discord_social_english import route_fixture
    source=Path('discord_bridge.py').read_text()
    for error in (OSError('synthetic read failure'),MemoryError('synthetic allocation failure')):
        result,_,_=route_fixture(source,text='Where is it?',real_strategy=True,adapter_available=False,
            symbolic_error=error,state={'text_expression_intent':{'native_text':'λ','meaning_references':['one']}})
        assert result.text=='λ' and result.metadata['adapter']=='selected_native_expression'
        assert result.metadata['symbolic_comprehension']['status']=='unavailable'
