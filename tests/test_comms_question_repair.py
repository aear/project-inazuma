import os
from unittest import SkipTest
if os.environ.get('INA_OFFLINE_COMMS_TESTS')!='1':
    raise SkipTest('full communication boundary fixtures require verified complete shutdown')
from pathlib import Path
from conversation_grounding import plan_conversation_reply
from benchmarks.benchmark_conversation_grounding import scene_for
from benchmarks.benchmark_comms_question_repair import cases
from benchmarks.benchmark_discord_social_english import route_fixture


def test_candidate_question_repair_contract():
    rows=cases(plan_conversation_reply)
    assert rows and all(r['correct'] for r in rows),rows


def route(**kwargs):
    return route_fixture(Path('discord_bridge.py').read_text(),text='Why does the comet shimmer?',
        real_strategy=True,adapter_available=False,preference=kwargs.pop('preference','english'),**kwargs)[0]


def test_actual_processing_body_can_select_optional_repair():
    result=route()
    assert result.text=='Could you rephrase that question?'
    assert result.metadata['expression_decision']['strategy']=='clarify'
    assert result.metadata['conversation_reply_plan']['answer_candidate'] is None


def test_explicit_silence_is_still_final():
    result=route(state={'text_expression_intent':{'strategy':'silence'}})
    assert result.text is None and result.metadata['expression_decision']['reason']=='explicit_intent'


def test_existing_low_urge_gate_is_preserved():
    result=route(low_urge=True)
    assert result.text is None and result.metadata['adapter']=='urge_gate'


def test_native_only_does_not_get_an_english_repair():
    result=route(preference='native_only')
    assert result.text is None


def test_explicit_native_expression_keeps_its_own_candidate():
    value={'native_text':'∘⊙ λ','meaning_references':['meaning:synthetic'],'strategy':'native'}
    result=route(state={'text_expression_intent':value},preference='native_only')
    assert result.text=='∘⊙ λ'


def test_planning_does_not_write_or_modify_scene(tmp_path,monkeypatch):
    from copy import deepcopy
    monkeypatch.chdir(tmp_path)
    q='Why does the comet shimmer?';scene=scene_for([],q);before=deepcopy(scene)
    plan_conversation_reply(q,scene)
    assert scene==before and list(tmp_path.iterdir())==[]


def test_explicit_mirroring_remains_a_choice():
    result=route(strategy='mirror',state={'text_expression_intent':{'strategy':'mirror'}})
    assert result.text and result.metadata['expression_decision']['strategy']=='mirror'
    assert result.metadata['expression_decision']['reason']=='explicit_intent'
