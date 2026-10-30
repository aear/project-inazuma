from discourse_context import build_discourse_context, resolution_for
from identity_language import identity_aliases
from benchmarks.benchmark_self_name_reference import assess


def context(text,**kwargs):
    return build_discourse_context(text,speaker={'id':'human','name':'Mira'},
        addressee=kwargs.pop('addressee',{'id':'child','name':'Inazuma','is_self':True}),
        self_identity={'id':'child','name':'Inazuma_Yagami','is_self':True},self_aliases=['Ina','Inazuma'],**kwargs)


def test_versioned_candidate_reference_contract():
    r=assess('V2');assert r['correct']==r['total'],[c for c in r['cases'] if not c['correct']]


def test_name_and_your_bind_to_self_without_learning_a_new_identity():
    from semantic_event import build_semantic_event
    c=context('Ina is your name, silly.')
    event=build_semantic_event('Ina is your name, silly.',c)['events'][0]
    assert event['agent']['id']=='self'
    assert event['arguments']['theme']['referent']['id']=='self'
    assert c['self_name_context']['self_name_statement']
    assert not c['self_name_context']['identity_changed']


def test_vocative_retains_literal_surface_and_retrieval_routes():
    from discourse_context import retrieval_routes
    text='Morning, Ina. <3';c=context(text)
    name=resolution_for(c,'ina')
    assert text[slice(*name['source_span'])]=='Ina'
    assert name['role']=='named_entity'
    assert {'ina','inazuma'}<=set(name['retrieval_terms'])
    assert any(r['role']=='named_entity' and r['status']=='resolved' for r in retrieval_routes(c))


def test_quoted_names_and_other_people_stay_distinct():
    for text in ('"Ina", she said.', 'The word ‘Ina’ is short.', "The name 'Ina' is mentioned."):
        c=context(text);assert not resolution_for(c,'ina')['referents']
        assert not c['self_name_context']['direct_self_address']
    c=context('Morning, Ina.',mentioned_entities=[{'id':'human-ina','name':'Ina'}])
    assert resolution_for(c,'ina')['ambiguous']
    assert not c['self_name_context']['direct_self_address']


def test_scene_label_duplicate_is_not_an_extra_identity():
    c=context('Morning, Inazuma.',mentioned_entities=['Mira','Inazuma'])
    assert resolution_for(c,'inazuma')['referents'][0]['id']=='self'
    assert not resolution_for(c,'inazuma')['ambiguous']


def test_personal_alias_override_can_decline_operator_names(monkeypatch,tmp_path):
    import discord_bridge as db
    from tests.test_discord_grounding_routing import _enable_replying,_message
    _enable_replying(monkeypatch);monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db,'get_discord_config',lambda:{'allow_replies':True,'identity_aliases':{'TestChild':['Ina','Inazuma']}})
    monkeypatch.setattr(db,'get_chat_adapter',lambda:None)
    monkeypatch.setattr(db,'generate_symbolic_reply_from_text',lambda *a,**k:None)
    path=tmp_path/'AI_Children/TestChild/memory/inastate.json';path.parent.mkdir(parents=True)
    path.write_text('{"communication_identity_aliases":[]}')
    r=db.process_inbound_message(_message('Morning, Ina.'))
    assert not r.metadata['self_name_context']


def test_aliases_are_scoped_to_the_selected_child(monkeypatch,tmp_path):
    import discord_bridge as db
    from tests.test_discord_grounding_routing import _enable_replying,_message
    _enable_replying(monkeypatch);monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db,'get_discord_config',lambda:{'allow_replies':True,'identity_aliases':{'AnotherChild':['Ina']}})
    monkeypatch.setattr(db,'get_chat_adapter',lambda:None)
    monkeypatch.setattr(db,'generate_symbolic_reply_from_text',lambda *a,**k:None)
    r=db.process_inbound_message(_message('Morning, Ina.'))
    assert not r.metadata['self_name_context']


def test_quote_and_source_limits_leave_names_unresolved():
    c=context(('"" '*129)+'Morning, Ina.')
    assert not c['self_name_context']['quotes_complete']
    assert not c['self_name_context']['direct_self_address']
    c=context(' '*8189+'Inactive')
    assert not resolution_for(c,'ina')
    assert not c['self_name_context']['source_complete']


def test_real_route_distinguishes_automatic_echo_from_deliberate_mirroring(monkeypatch,tmp_path):
    import discord_bridge as db
    from tests.test_discord_grounding_routing import _enable_replying,_message
    _enable_replying(monkeypatch);monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db,'get_discord_config',lambda:{'allow_replies':True,'identity_aliases':{'TestChild':['Ina','Inazuma']}})
    monkeypatch.setattr(db,'get_chat_adapter',lambda:None)
    monkeypatch.setattr(db,'generate_symbolic_reply_from_text',lambda *a,**k:{'text':'morning ina','native_text':'∘⊙','gloss_text':'morning ina','symbols':['s1','s2','s3'],'unknown':[]})
    message=_message('Morning, Ina. <3')
    r=db.process_inbound_message(message)
    assert r.text is None and r.metadata['expression_decision']['automatic_mirror_excluded']
    path=tmp_path/'AI_Children/TestChild/memory/inastate.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('{"text_expression_intent":{"strategy":"mirror"}}')
    explicit=db.process_inbound_message(message)
    assert explicit.text=='morning ina'
    assert explicit.metadata['expression_decision']['reason']=='explicit_intent'
