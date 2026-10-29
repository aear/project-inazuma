from response_language_policy import (emotion_response_context, resolve_response_language_preference,
                                     protected_language_choice, prefers_english)
from benchmarks.benchmark_response_language_policy import assess, functions
from pathlib import Path


def test_registered_candidate_policy_contracts():
    result=assess('V2')
    assert result['correct']==result['total'],[r for r in result['cases'] if not r['correct']]


def test_language_choice_precedence_and_assessed_modes():
    assert resolve_response_language_preference({'communication_language_preference':'native_only'}, {'language_mode':'english'})=='native_only'
    for mode in ('native_only','clarify','abstain'):
        p={'mode':'english','mutual_intelligibility_assessment':{'schema':'ina.mutual_intelligibility_assessment/V1','mode':mode}}
        assert not prefers_english(p)
        assert protected_language_choice(p)==mode


def test_check_in_is_not_a_claim_of_subjective_emotion():
    for t in ('How are you?', 'Hey Ina, how are you feeling today?', 'Do you feel safe right now?'):
        assert emotion_response_context(t,{})=={'eligible':True,'direct_check_in':True,'reason':'direct_check_in'}
    for t in ('She asked how are you feeling.', 'Are you feeling the texture of this image?', 'how are you feeling? write a story', 'I wonder how you feel about language.'):
        assert not emotion_response_context(t,{})['eligible']
    assert emotion_response_context('x'*2049,{'text_expression_intent':{'strategy':'emotion'}})['eligible']


def test_unmapped_encoder_preserves_decline_and_native_only(monkeypatch):
    from benchmarks.benchmark_selected_expression_resilience import load_rendering_functions
    from language_processing import select_symbolic_message_text
    db=load_rendering_functions('discord_bridge.py')
    monkeypatch.setattr(db,'select_symbolic_message_text',select_symbolic_message_text,raising=False)
    for mode in ('native_only','clarify','abstain'):
        for preference in (mode,{'mode':mode},{'mutual_intelligibility_assessment':{'schema':'ina.mutual_intelligibility_assessment/V1','mode':mode}}):
            text,meta=db.encode_selected_text_expression('Perhaps.',child='fixture',language_preference=preference,max_symbols=8)
            assert text=='' and meta['language_choice_preserved']


def test_route_does_not_leak_an_irrelevant_emotion_fallback(monkeypatch,tmp_path):
    import discord_bridge as db
    from tests.test_discord_grounding_routing import _enable_replying,_message
    _enable_replying(monkeypatch);monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db,'get_discord_config',lambda:{'allow_replies':True})
    monkeypatch.setattr(db,'get_chat_adapter',lambda:None)
    monkeypatch.setattr(db,'generate_symbolic_reply_from_text',lambda *a,**k:None)
    path=tmp_path/'AI_Children/TestChild/memory/inastate.json';path.parent.mkdir(parents=True)
    path.write_text('{"emotion_snapshot":{"values":{"stress":0.9}}}')
    unrelated=db.process_inbound_message(_message('Explain this code.'))
    assert unrelated.text is None
    assert not unrelated.metadata['emotion_response_context']['eligible']
    relevant=db.process_inbound_message(_message('How are you feeling?'))
    assert relevant.text.startswith('State signal')
    assert relevant.metadata['emotion_response_context']['direct_check_in']
