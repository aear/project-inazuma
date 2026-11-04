from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as N
from conversation_grounding import plan_conversation_reply,observe_conversation_turn,english_candidate_allowed
from conversation_scene import ConversationSceneBuffer
from benchmarks.benchmark_conversation_grounding import scene_for,message,assess


def test_pinned_conversation_route_comparison():
    old,new=assess('V1'),assess('V2')
    assert old['correct']<old['total']
    assert new['correct']==new['total'],[r for r in new['cases'] if not r['correct']]


def test_held_out_entities_and_question_forms_need_no_answer_table():
    for statement,question in (
        ('The silver compass is behind the curtain.','Where is the silver compass?'),
        ('The kettle was near the window.','Where was the kettle?'),
        ('I am inside the house.','Where am I?'),
        ('I want the blue ribbon.','What do I want?'),
        ('I see a moth.','Who sees a moth?'),
        ('I do not want rain.','Do I want rain?'),
    ):
        scene=scene_for([statement],question)
        p=plan_conversation_reply(question,scene)
        assert p['status']=='answer_available',(statement,question,p)
        assert p['evidence'][0]['source_text']==statement
        assert p['answer_candidate'].startswith('You said that ')
        assert not p['truth_certified'] and p['outward_text'] is None
        assert p['expression_intent']['semantic_references']==['conversation:s0']
        assert p['selected_native_intents']


def test_duplicate_delivery_does_not_create_support_or_silently_resolve_conflicts():
    scene=scene_for(['The lantern is behind the curtain.'],'Where is the lantern?')
    source=deepcopy(scene['turns'][0]);scene['turns'].insert(0,deepcopy(source))
    p=plan_conversation_reply('Where is the lantern?',scene)
    assert p['selected_source_ids']==['s0'] and len(p['evidence'])==1
    assert not p['independence_inferred']
    conflict=deepcopy(source);conflict['text']='The lantern is under the table.'
    conflict['conversation_observation']=observe_conversation_turn(conflict,conflict['discourse'])
    scene['turns'].insert(0,conflict)
    p=plan_conversation_reply('Where is the lantern?',scene)
    assert p['answer_candidate'] is None and p['status']=='missing_evidence'


def test_modal_and_negative_claims_keep_their_original_scope():
    for source,question in (
        ('The kettle might be under the table.','Where is the kettle?'),
        ('I do not like noise.','What do I not like?'),
        ('I never like noise.','Do I like noise?'),
    ):
        p=plan_conversation_reply(question,scene_for([source],question))
        assert p['status']=='answer_available',(source,p)
        assert p['evidence'][0]['source_text']==source
        if 'might' in source:assert 'might' in p['answer_candidate']
        if 'not' in source:assert 'not' in p['answer_candidate']
        if 'never' in source:assert source in p['answer_candidate']
    for source,question in (
        ('I like music.','What do I not like?'),
        ('I do not like music.','What do I like?'),
        ('The kettle is under the table.','Where is the kettle not?'),
    ):
        assert plan_conversation_reply(question,scene_for([source],question))['answer_candidate'] is None


def test_repair_requires_same_author_target_and_relation():
    for replacement in (
        {'text':'Actually, the kettle is beside the lamp.','reply':'backend-s0','speaker':'Rowan'},
        {'text':'Actually, the kettle is beside the lamp.','reply':'missing'},
        {'text':'Actually, the lantern is beside the lamp.','reply':'backend-s0'},
    ):
        scene=scene_for([{'text':'The kettle is under the table.'},replacement],'Where is the kettle?')
        p=plan_conversation_reply('Where is the kettle?',scene)
        assert not p['repairs']
    scene=scene_for([{'text':'The kettle is under the table.'},
                     {'text':'Actually, the kettle is beside the lamp.','reply':'backend-s0'}], 'Where is the kettle?')
    p=plan_conversation_reply('Where is the kettle?',scene)
    assert len(p['repairs'])==1 and 'beside the lamp' in p['answer_candidate']
    assert scene['turns'][0]['text']=='The kettle is under the table.'


def test_other_channels_and_outbound_quotes_are_not_inbound_evidence():
    b=ConversationSceneBuffer();b.observe(message('1','The kettle is under the table.',channel='private'))
    scene=b.observe(message('2','Where is the kettle?',channel='elsewhere'))
    assert plan_conversation_reply('Where is the kettle?',scene)['answer_candidate'] is None
    scene=scene_for([{'text':'The kettle is under the table.','is_self':True}], 'Where is the kettle?')
    assert plan_conversation_reply('Where is the kettle?',scene)['answer_candidate'] is None


def test_same_display_name_does_not_merge_different_speakers():
    b=ConversationSceneBuffer()
    first=message('1','I like music.');first.sender.backend_id='person-a';b.observe(first)
    second=message('2','What do I like?');second.sender.backend_id='person-b'
    scene=b.observe(second)
    assert plan_conversation_reply(second.text,scene)['answer_candidate'] is None


def test_incomplete_history_unresolved_roles_and_scope_do_not_become_answers():
    for text in ('If the lantern is under the table, I see it.', 'I think the lantern is under the table.',
                 'Imagine the lantern is under the table.', 'The lantern is under it.', 'The lantern is under the table\nI like music',
                 '"The lantern is under the table."', 'The lantern is under the table.'+' '*10+'x'*800):
        question='Where is the lantern?'
        p=plan_conversation_reply(question,scene_for([text],question))
        assert p['answer_candidate'] is None,(text,p)
    b=ConversationSceneBuffer()
    b.ingest_context(message('0',''),[{'id':'old','content':'The lantern is under the table.','author_name':'Mira'}])
    scene=b.observe(message('new','Where is the lantern?'))
    assert plan_conversation_reply('Where is the lantern?',scene)['answer_candidate'] is None


def test_scene_and_question_budgets_are_hard_limits():
    scene=scene_for(['The book is under the table.'],'Where is the book?')
    assert plan_conversation_reply('x'*513,scene)['status']=='input_budget_exceeded'
    large=deepcopy(scene);large['turns']*=7
    assert plan_conversation_reply('Where is the book?',large)['status']=='scene_budget_exceeded'
    modified=deepcopy(scene);modified['turns'][0]['conversation_observation']['source_text']='changed'
    assert plan_conversation_reply('Where is the book?',modified)['answer_candidate'] is None


def test_private_evidence_tool_does_not_deliver_or_adopt_a_goal(tmp_path):
    from personal_tool_runtime import capability_catalog,execute_personal_tool_command
    assert 'conversation_evidence_review' in capability_catalog()['commands']
    scene=scene_for(['I like music.'],'What do I like?')
    r=execute_personal_tool_command({'action':'conversation_evidence_review','question':'What do I like?','scene':scene},
        child='Fixture',project_root=tmp_path,config={'ina_hdd_writable_path':str(tmp_path/'personal')})['value']
    assert r['status']=='answer_available' and r['outward_text'] is None
    assert not r['automatic_learning'] and r['continuation_budget']==0


def test_language_choices_preserve_native_and_assessed_authority():
    for pref in ('native_only','clarify','abstain','silence',{'mode':'native_only'},
                 {'mode':'english','mutual_intelligibility_assessment':{'mode':'native_only'}}):
        assert not english_candidate_allowed(pref)
    for pref in ('english','mixed','auto',{'mode':'english'}):assert english_candidate_allowed(pref)


def test_parser_keeps_multiword_location_subject_and_never_scope():
    from language_intelligence import analyze_utterance
    for text,subject in [('Where is the old silver compass?','the old silver compass'),('Where was the kettle?','the kettle')]:
        clause=analyze_utterance(text)['clauses'][0]
        assert clause['subject']==subject and clause['arguments']=={}
    c=analyze_utterance('Rowan never wants noise.')['clauses'][0]
    assert c['subject']=='rowan' and c['negated'] and c['negation_operator']=='never'


def test_bot_source_and_observer_failure_do_not_interrupt_communication(monkeypatch):
    b=ConversationSceneBuffer();m=message('bot','The kettle is under the table.')
    m.metadata['author_is_bot']=True;b.observe(m)
    scene=b.observe(message('q','Where is the kettle?'))
    assert plan_conversation_reply('Where is the kettle?',scene)['answer_candidate'] is None
    import conversation_grounding
    def fail(*a,**k):raise RuntimeError('synthetic optional observer unavailable')
    monkeypatch.setattr(conversation_grounding,'observe_conversation_turn',fail)
    scene=b.observe(message('next','Where is the lantern?'))
    assert scene['signals']['current_is_question']
    assert scene['turns'][-1]['conversation_observation']['status']=='observation_unavailable'


def test_generated_reply_reframes_only_resolved_speaker_roles():
    p=plan_conversation_reply('What do I like?',scene_for(['I like music.'],'What do I like?'))
    assert p['answer_candidate']=='You said that you like music.'
    assert p['realisations'][0]['status']=='composed_attributed_claim'
    assert p['realisations'][0]['meaning_fidelity']=='not_certified'
    child=p['selected_native_intents'][0]['events'][1]
    assert child['agent']['id']=='mira' and not child['embedded_content_is_fact']
    # A claim about Ina stays attributed; it never becomes her own preference.
    p=plan_conversation_reply('What do you like?',scene_for(['You like music.'],'What do you like?'))
    assert p['answer_candidate']=='You said that I like music.'
    assert not p['truth_certified']


def test_existing_english_renderer_preserves_attributed_selected_meaning(monkeypatch):
    from benchmarks.benchmark_selected_expression_resilience import load_rendering_functions
    from language_processing import select_symbolic_message_text
    renderer=load_rendering_functions('discord_bridge.py')
    monkeypatch.setattr(renderer,'select_symbolic_message_text',select_symbolic_message_text,raising=False)
    monkeypatch.setattr(renderer,'_is_emotion_or_sound_symbol',lambda s:False,raising=False)
    monkeypatch.setattr(renderer,'generate_symbolic_reply_from_text',lambda *a,**k:{'symbols':['s'],'unknown':[]},raising=False)
    monkeypatch.setattr(renderer,'build_dual_symbolic_message',lambda *a,**k:{'native_text':'∘⊙','native_tokens':['∘⊙'],
        'gloss_text':'you might like music','gloss_tokens':['you','might','like','music'],
        'native_sources':{'s':'fixture'},'gloss_sources':{'s':'fixture'}},raising=False)
    selected='You said that you might like music.'
    output,meta=renderer.encode_selected_text_expression(selected,child='Fixture',language_preference='english',max_symbols=12)
    assert selected in output and 'Native: ∘⊙' in output
    monkeypatch.setattr(renderer,'generate_symbolic_reply_from_text',lambda *a,**k:None)
    selected='Mira said: “I never like noise.”'
    output,meta=renderer.encode_selected_text_expression(selected,child='Fixture',language_preference='english',max_symbols=12)
    assert output==selected


def test_optional_planning_failure_leaves_existing_response_path_available(monkeypatch):
    import conversation_grounding
    from benchmarks.benchmark_discord_social_english import route_fixture
    def fail(*a,**k):raise RuntimeError('synthetic optional planner unavailable')
    monkeypatch.setattr(conversation_grounding,'plan_conversation_reply',fail)
    source=Path('discord_bridge.py').read_text()
    result,_,_=route_fixture(source,text='What do I like?',draft='A chosen existing reply.')
    assert result.text=='A chosen existing reply.'
    assert result.metadata['conversation_reply_plan']['status']=='planning_unavailable'


def test_negative_and_positive_claims_about_different_things_can_coexist():
    scene=scene_for(['I like music.','I do not like noise.'],'What do I like?')
    p=plan_conversation_reply('What do I like?',scene)
    assert p['answer_candidate']=='You said that you like music.' and p['status']=='answer_available'
    p=plan_conversation_reply('What do I not like?',scene)
    assert p['answer_candidate']=='You said that you do not like noise.'
    scene=scene_for(['The book is beside the lamp.','The book is near the window.'],'Where is the book?')
    p=plan_conversation_reply('Where is the book?',scene)
    assert p['status']=='unresolved_claims' and p['answer_candidate'] is None


def test_terminal_heart_or_emoji_does_not_destroy_the_question_or_claim():
    source='I like music. <3'
    scene=scene_for([source],'What do I like? <3')
    p=plan_conversation_reply('What do I like? <3',scene)
    assert p['answer_candidate']=='You said that you like music.'
    assert p['evidence'][0]['source_text']==source
    assert scene['turns'][0]['conversation_observation']['uninterpreted_suffix']==' <3'
    assert p['query']['source_surface']=='What do I like? <3'
    assert p['query']['uninterpreted_suffix']==' <3'
    source='The lantern is under the table. 😊'
    scene=scene_for([source],'Where is the lantern？ 😊')
    p=plan_conversation_reply('Where is the lantern？ 😊',scene)
    assert p['answer_candidate']=='You said that the lantern is under the table.'
    # Additional words after punctuation still require their own interpretation.
    scene=scene_for(['I like music. Except this song.'],'What do I like?')
    assert plan_conversation_reply('What do I like?',scene)['answer_candidate'] is None


def test_discord_reference_is_not_confused_with_outgoing_reply_target():
    from benchmarks.benchmark_conversation_grounding import reference_route_fixture
    capture=reference_route_fixture(Path('discord_bridge.py').read_text())
    assert capture['reply_to_backend_id']=='new-message'
    assert capture['metadata']['referenced_backend_message_id']=='original-message'
    b=ConversationSceneBuffer()
    first=message('s0','The book is beside the lamp.')
    first.metadata['backend_message_id']='original-message';b.observe(first)
    correction=message('repair','Actually, the book is under the table.')
    correction.metadata=capture['metadata'];correction.metadata['backend_message_id']='new-message'
    correction.reply_to_id='new-message'
    b.observe(correction)
    scene=b.observe(message('q','Where is the book?'))
    p=plan_conversation_reply('Where is the book?',scene)
    assert p['answer_candidate']=='You said that the book is under the table.'
    assert p['repairs']==[{'original':'s0','revision':'repair','truth_certified':False}]
