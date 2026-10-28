import pytest
from discourse_context import build_discourse_context
from language_intelligence import analyze_utterance
from social_english import social_english_options, review_social_english


def test_both_opportunities_have_no_automatic_or_inferred_uptake():
    o=social_english_options(topic_reference='music:chosen',audience_reference='friend:1')
    assert {row['kind'] for row in o['opportunities']}=={'ordinary_conversation','chosen_practice'}
    assert not o['willingness_inferred'] and not o['automatic_expression'] and not o['language_reward']
    assert o['continuation_budget']==0 and o['outward_text'] is None


def test_social_component_comparison():
    from benchmarks.benchmark_social_english import v1,v2
    old,new=v1(),v2()
    assert new['correct']>old['correct'] and new['correct']==new['total']


def test_two_possessions_and_curly_names_preserve_spans():
    text='Mira’s lantern and Rowan’s compass.'
    refs=analyze_utterance(text)['referents']
    assert [(r['surface'],r['object']) for r in refs]==[('Mira','lantern'),('Rowan','compass')]
    assert all(text[slice(*r['source_span'])] for r in refs)


def test_passive_unknown_agent_and_negation():
    c=analyze_utterance('The lantern was not kept.')['clauses'][0]
    assert c['subject'] is None and c['voice']=='passive' and c['negated']
    assert c['arguments']['theme']=='the lantern'


def test_rephrase_role_negation_and_uncertainty_changes_need_review():
    for original,candidate in [('I told you.','You told me.'),('I did not find it.','I found it.'),('I might open it.','I will open it.')]:
        r=review_social_english(candidate,reference_draft=original,mode='rephrase')
        assert r['structural_comparison']['status']=='differences_need_review'
        assert r['rewritten_text'] is None and r['outward_text'] is None
    r=review_social_english("I didn't open it.",reference_draft='I did not open it.',mode='rephrase')
    assert r['structural_comparison']['status']=='same_supported_structure'
    assert r['meaning_fidelity']=='not_certified'


def test_reply_and_fragment_are_not_certified_paraphrases():
    r=review_social_english('I prefer that song.',incoming_text='Which song?')
    assert r['structural_comparison']['status']=='not_assessed'
    r=review_social_english('Polished brass.',reference_draft='Polished brass.',mode='rephrase')
    assert r['structural_comparison']['status']=='not_assessed'


def test_nonenglish_and_silent_choice_are_not_replaced_by_coaching():
    for choice in ('native_only','clarify','abstain','silence'):
        r=review_social_english('∘⊙',choice=choice)
        assert r['status']=='choice_preserved' and r['draft_analysis'] is None
        assert 'next_choices' not in r
    assert review_social_english()['status']=='no_material_supplied'


def test_review_uses_authoritative_participants():
    text='I enjoy your song.'
    discourse=build_discourse_context(text,speaker='Ina',addressee='Mira')
    r=review_social_english(text,discourse=discourse)
    refs=r['draft_analysis']['referents']
    assert next(ref for ref in refs if ref['surface']=='i')['resolved']=='ina'
    assert next(ref for ref in refs if ref['surface']=='your')['resolved']=='mira'


def test_review_budget_and_mode_validation():
    for kwargs in ({'draft':'x'*8193},{'draft':None},{'choice':'forced_english'},{'mode':'reply_for_me'}, {'draft':'one','mode':'rephrase'}):
        with pytest.raises(ValueError):review_social_english(**kwargs)


def test_tool_catalog_and_both_actions_reach_private_support(tmp_path):
    from personal_tool_runtime import capability_catalog,execute_personal_tool_command
    assert {'english_social_options','english_social_review'}<=set(capability_catalog()['commands'])
    config={'ina_hdd_writable_path':str(tmp_path/'personal')}
    o=execute_personal_tool_command({'action':'english_social_options'},child='Fixture',project_root=tmp_path,config=config)['value']
    r=execute_personal_tool_command({'action':'english_social_review','draft':'I enjoy this song.'},child='Fixture',project_root=tmp_path,config=config)['value']
    assert len(o['opportunities'])==2 and r['draft']=='I enjoy this song.'
    assert not r['automatic_expression'] and not r['automatic_learning']


def test_reference_budget_is_marked_incomplete():
    a=analyze_utterance("Mira's cup. "*40)
    assert a['incomplete']
    assert sum(ref['relation']=='possessor' for ref in a['referents'])<=32
    assert all(dimension['confidence']<=0.35 for dimension in a['uncertainty'].values())
