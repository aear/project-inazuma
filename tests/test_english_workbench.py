import pytest
from benchmarks.benchmark_english_workbench import assess,event,selected
from english_composition import compose_selected_english
from english_learning_profile import build_english_learning_profile
from language_intelligence import analyze_utterance
from semantic_event import build_semantic_event,build_native_intent


def test_candidate_versioned_contracts():
    r=assess('V2');assert r['correct']==r['total'],[c for c in r['cases'] if not c['correct']]


def test_nested_complements_keep_local_negation_and_native_structure():
    text='I believe Mira does not think Rowan wants the compass.'
    a=analyze_utterance(text)
    assert [c['predicate'] for c in a['clauses']]==['believe','think','want']
    assert [c['negated'] for c in a['clauses']]==[False,True,False]
    native=build_native_intent(build_semantic_event(text))
    assert sum(g['construction']=='embedded_content' for g in native['grammar'])==2
    assert all(not e['embedded_content_is_fact'] for e in native['events'])
    a=analyze_utterance('I want to open the gate.')
    assert len(a['clauses'])==1


def test_alternatives_are_retained_without_an_automatic_choice():
    native=build_native_intent(build_semantic_event('Do you want the book or the lantern?'))
    g=next(g for g in native['grammar'] if g['construction']=='alternatives')
    assert not g['choice_selected'] and g['options'][0]['options']==['the book','the lantern']


def test_realiser_preserves_supported_held_out_roles_and_tenses():
    cases=[(event(agent={'id':'vale','name':'Vale'},predicate='have',arguments={'theme':'the map'}),'Vale has the map.'),
           (event(predicate='teach',arguments={'recipient':'Rowan','theme':'the song'},tense='past'),'I taught Rowan the song.'),
           (event(predicate='understand',arguments={'theme':'the request'},tense='past'),'I understood the request.'),
           (event(agent={'id':'self','is_self':True},predicate='find',voice='passive',arguments={'theme':'the lantern'},tense='past'),'The lantern was found by me.')]
    for e,expected in cases:
        n,i=selected([e]);r=compose_selected_english(n,i)
        assert r['candidate_text']==expected and r['meaning_fidelity']=='not_certified'


def test_no_selected_meaning_and_no_text_authority_are_rejected():
    n,i=selected([event()]);i['meaning_references']=[]
    with pytest.raises(ValueError):compose_selected_english(n,i)
    n,i=selected([event()]);i['allowed_media']=['native_symbol']
    with pytest.raises(PermissionError):compose_selected_english(n,i)
    for events in ([event(id='c0',parent_clause='c0')],[event(id='c0'),event(id='c0')],[],[event(id=str(x)) for x in range(5)]):
        n,i=selected(events)
        with pytest.raises(ValueError):compose_selected_english(n,i)


def test_unsupported_scope_or_arguments_are_not_silently_dropped():
    for e in (event(modality='might',tense='past'),event(arguments={'instrument':'the key'}),event(arguments={'theme':'word. New claim'})):
        n,i=selected([e]);r=compose_selected_english(n,i)
        assert r['candidate_text'] is None and r['status']=='unsupported_structure'


def test_chosen_private_tools_do_not_adopt_goals_or_send_text(tmp_path):
    from personal_tool_runtime import capability_catalog,execute_personal_tool_command
    assert {'english_compose_selected','english_learning_options'}<=set(capability_catalog()['commands'])
    config={'ina_hdd_writable_path':str(tmp_path/'personal')}
    profile=execute_personal_tool_command({'action':'english_learning_options'},child='Fixture',project_root=tmp_path,config=config)['value']
    assert all(g['adoption']=='not_observed' for g in profile['goals'])
    n,i=selected([event()])
    r=execute_personal_tool_command({'action':'english_compose_selected','native_intent':n,'expression_intent':i},child='Fixture',project_root=tmp_path,config=config)['value']
    assert r['candidate_text']=='I want the lantern.' and r['outward_text'] is None


def test_observatory_rows_distinguish_goal_and_implementation_evidence(tmp_path):
    import ast,json
    from pathlib import Path
    source=Path('monitoring_dashboard.py').read_text();tree=ast.parse(source)
    names={'_developmental_readiness','_safe_json','_modified'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace={'Path':Path,'Any':object,'json':json,'MAX_JSON_BYTES':32*1024*1024,'time':__import__('time'),'datetime':__import__('datetime').datetime,'timezone':__import__('datetime').timezone}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<dashboard collectors>','exec'),namespace)
    path=tmp_path/'current.json';path.write_text(json.dumps({'assessments':{'language_english':{'status':'complete','results':[],'learning_profile':build_english_learning_profile()}}}))
    cards,rows=namespace['_developmental_readiness'](tmp_path/'none',tmp_path/'none2',path)
    goals=[r for r in rows if r[0].startswith('English goal')]
    assert len(goals)==7 and all('optional' in r[2] and 'Ina not assessed' in r[2] for r in goals)


def test_operators_and_partial_meanings_are_not_silently_lost():
    for e in (event(question_word='why'),event(negated=True,negation_operator='never'),event(alternatives=[{'options':['tea','coffee']}] )):
        n,i=selected([e]);assert compose_selected_english(n,i)['candidate_text'] is None
    n,i=selected([event()]);n['incomplete']=True
    assert compose_selected_english(n,i)['candidate_text'] is None


def test_assertive_modals_do_not_bypass_uncertainty_guidance():
    plan={'schema':'ina.experience_cognition/V3','epistemic_state':{'status':'uncertain'}}
    for modal in ('will','must','should','can'):
        n,i=selected([event(modality=modal)])
        assert compose_selected_english(n,i,cognition_plan=plan)['status']=='needs_chosen_qualification'
    n,i=selected([event(negated=False,negation_operator='not')])
    assert compose_selected_english(n,i)['candidate_text'] is None
