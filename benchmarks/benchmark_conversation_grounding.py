"""One-speaker dialogue transfer fixtures; separate from live Ina performance."""
from pathlib import Path
from types import SimpleNamespace as N
BASELINE='546f3fead068f31feaf721c297bab8f31c7fcb98'


def message(index,text,*,speaker='Mira',channel='fixture',reply=None,is_self=False,complete=True):
    return N(id=str(index),text=text,backend='discord',direction='outbound' if is_self else 'inbound',
             reply_to_id=reply,metadata={'backend_message_id':'backend-'+str(index),'referenced_backend_message_id':reply},
             sender=N(display_name=speaker,backend_id=speaker.casefold(),is_self=is_self),
             channel=N(name=channel))


def scene_for(statements,question,*,buffer=None):
    if buffer is None:
        from conversation_scene import ConversationSceneBuffer
        buffer=ConversationSceneBuffer()
    for i,item in enumerate(statements):
        if isinstance(item,str):m=message('s'+str(i),item)
        else:m=message('s'+str(i),**item)
        buffer.observe(m)
    return buffer.observe(message('question',question))


def reference_route_fixture(source):
    """Real Discord ingress method, anonymous collaborators; no client/network."""
    import ast
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_route_to_comms')
    capture={}
    env={'make_sender_info_from_discord':lambda *a,**k:None,'make_channel_info_from_discord':lambda *a,**k:None,
         'BACKEND_NAME':'discord','discord_space_identity':lambda *a,**k:{'guild_name':'fixture','identity':'fixture','label':'fixture','conversation_mode':'ordinary'}}
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),'<copied ingress>','exec'),env)
    client=N(user=N(id='ina'),comms=N(receive_inbound=lambda **kw:capture.update(kw)))
    msg=N(id='new-message',author=N(id='mira',bot=False),channel=N(id='room'),guild=None,
          content='Actually, the book is under the table.',reference=N(message_id='original-message'))
    env['_route_to_comms'](client,msg,is_dm=True)
    return capture


def assess(version):
    from benchmarks.benchmark_discord_social_english import route_fixture
    from conversation_grounding import plan_conversation_reply,english_candidate_allowed
    if version=='V1':
        from historical_source import historical_text,historical_module
        source=historical_text('discord_bridge.py',BASELINE)
        Buffer=historical_module('conversation_scene.py',BASELINE).ConversationSceneBuffer
        profile_builder=historical_module('english_learning_profile.py',BASELINE).build_english_learning_profile
        tool_source=historical_text('personal_tool_runtime.py',BASELINE)
        parse=historical_module('language_intelligence.py',BASELINE).analyze_utterance
    else:
        source=Path('discord_bridge.py').read_text()
        from conversation_scene import ConversationSceneBuffer as Buffer
        from english_learning_profile import build_english_learning_profile as profile_builder
        tool_source=Path('personal_tool_runtime.py').read_text()
        from language_intelligence import analyze_utterance as parse
    rows=[]
    def add(name,component,ok):rows.append({'case':name,'component':component,'correct':bool(ok)})
    def trial(statements,question,**kwargs):
        if (kwargs.get('state') or {}).get('text_expression_intent')=='mirror':kwargs.setdefault('strategy','mirror')
        scene=scene_for(statements,question,buffer=Buffer())
        response,_,_=route_fixture(source,text=question,scene_data=scene,real_strategy=True,adapter_available=False,**kwargs)
        return response,response.metadata.get('conversation_reply_plan',{}),scene
    for source_text,question,expected in [
        ('The book is beside the lamp.','Where is the book?','You said that the book is beside the lamp.'),
        ('I am inside the house.','Where am I?','You said that you are inside the house.'),
        ('The lantern is under the table.','Where is the lantern?','You said that the lantern is under the table.'),
        ('I like music.','What do I like?','You said that you like music.'),
        ('I like music. <3','What do I like? <3','You said that you like music.'),
        ('The book is beside the lamp.','Where is the book？ 😊','You said that the book is beside the lamp.'),
        ('I read the book.','What do I read?','You said that you read the book.'),
        ('I like music.','Who likes music?','You said that you like music.'),
        ('I like music.','Do I like music?','You said that you like music.'),
        ('I do not like noise.','What do I not like?','You said that you do not like noise.'),
        ('I do not like noise.','Do I like noise?','You said that you do not like noise.'),
        ('I never like noise.','Do I like noise?','Mira said: “I never like noise.”'),
        ('The book might be beside the lamp.','Where is the book?','You said that the book might be beside the lamp.'),
    ]:
        r,p,_=trial([source_text],question)
        add(question+' from '+source_text,'source_answer',r.text==expected and p.get('truth_certified') is False and p.get('evidence',[{}])[0].get('source_text')==source_text)
    r,p,_=trial(['I like music.'],'What do I like?')
    native=p.get('selected_native_intents') or []
    add('composed report preserves speaker and uncertain reported content','composition',bool(native) and native[0]['events'][0]['predicate']=='say' and native[0]['events'][1]['agent']['id']=='mira' and native[0]['events'][1]['embedded_content_is_fact'] is False and r.text!='I like music.')
    for statement,question in [
        ('I like music.','What do you like?'),
        ('The book was beside the lamp.','Where is the book?'),
        ('The book is red.','Where is the book?'),
        ('If the book is beside the lamp, I can see it.','Where is the book?'),
        ('“The book is beside the lamp.”','Where is the book?'),
        ('I think the book is beside the lamp.','Where is the book?'),
        ('The book is not beside the lamp.','Where is the book?'),
    ]:
        r,p,_=trial([statement],question)
        add('does not invent '+statement+' / '+question,'grounding',p.get('answer_candidate') is None)
    r,p,scene=trial(['The book is beside the lamp.','The book is under the table.'],'Where is the book?')
    add('conflicting locations invite clarification','repair',p.get('status')=='unresolved_claims' and r.metadata.get('adapter')=='conversation_clarification' and len(p.get('alternatives',[]))==2)
    repaired=[{'text':'The book is beside the lamp.'},{'text':'Actually, the book is under the table.','reply':'backend-s0'}]
    r,p,scene=trial(repaired,'Where is the book?')
    add('linked correction revises reading','repair',p.get('status')=='answer_available' and 'under the table' in (r.text or '') and 'beside the lamp' not in (r.text or '') and len(p.get('repairs',[]))==1)
    add('original observation survives correction','provenance',any(t['text']=='The book is beside the lamp.' for t in scene['turns']) and p.get('repairs',[{}])[0].get('original')=='s0')
    r,p,_=trial([{'text':'The book is beside the lamp.'},{'text':'Actually, the book is under the table.','reply':'missing'}],'Where is the book?')
    add('unlinked correction retains disagreement','repair',p.get('status')=='unresolved_claims' and not p.get('repairs'))
    r,p,_=trial([{'text':'The book is beside the lamp.'},{'text':'Actually, the book is under the table.','reply':'backend-s0','speaker':'Rowan'}],'Where is the book?')
    add('other author cannot overwrite source claim','provenance',p.get('status')=='unresolved_claims' and not p.get('repairs'))
    for strategy in ('silence','mirror','respond'):
        r,p,_=trial(['The book is beside the lamp.'],'Where is the book?',state={'text_expression_intent':strategy})
        add('explicit '+strategy,'agency',r.metadata['expression_decision']['strategy']==strategy)
    for preference in ('native_only','clarify','abstain','silence'):
        r,p,_=trial(['The book is beside the lamp.'],'Where is the book?',preference=preference)
        add('English quote not selected for '+preference,'agency',r.metadata.get('adapter')!='conversation_evidence' and r.metadata.get('adapter')!='conversation_clarification')
    r,p,_=trial(['The book is beside the lamp.'],'Where is the book?',preference='mixed')
    add('mixed choice uses existing renderer','agency',r.metadata.get('adapter')=='conversation_evidence' and 'beside the lamp' in (r.text or ''))
    r,p,_=trial(['The book is beside the lamp.'],'Where is the book?',low_urge=True)
    add('urge gate retains silence','agency',r.text is None and r.metadata.get('adapter')=='urge_gate')
    r,p,_=trial(['I like music.','I like art.'],'What do I like?')
    add('several objects are not automatically contradictions','composition',p.get('status')=='answer_available' and 'music' in (r.text or '') and 'art' in (r.text or ''))
    r,p,_=trial(['I like music.','I do not like noise.'],'What do I like?')
    add('different objects can have different polarities','composition',p.get('status')=='answer_available' and r.text=='You said that you like music.')
    r,p,_=trial(['I like music.','I do not like music.'],'Do I like music?')
    add('opposing claims remain uncertain','repair',p.get('status')=='conflicting_claims' and p.get('answer_candidate') is None)
    r,p,_=trial(['The book is beside the lamp.'],'Why is the book there?')
    add('unsupported reasoning does not invent a cause','grounding',p.get('answer_candidate') is None)
    r,p,_=trial([],'Where is the book?')
    add('specific information gap offers clarification','clarification',p.get('status')=='missing_evidence' and 'the book' in (p.get('clarification_candidate') or '') and r.metadata.get('adapter')=='conversation_clarification')
    # New helper contracts receive credit only if this route actually uses them.
    wired='plan_conversation_reply(user_text, conversation_scene' in source
    add('assessed language choice retained','agency',wired and not english_candidate_allowed({'mode':'english','mutual_intelligibility_assessment':{'schema':'ina.mutual_intelligibility_assessment/V1','mode':'native_only'}}))
    add('private evidence review tool available','tools','"conversation_evidence_review"' in tool_source)
    profile=profile_builder([{'module':'conversation_grounding','version':'V2','correct':1,'total':2}])
    goal=next(g for g in profile['goals'] if g['id']=='listener')
    add('optional goal retains limitations and unknown live performance','reporting',goal['implementation_status']=='limitations_retained' and goal['ina_performance']=='not_assessed' and goal['adoption']=='not_observed')
    for phrase,subject in [('Where is the red book?','the red book'),('Where was the old lantern?','the old lantern'),('I never like noise.','i'),('Mira never wants tea.','mira')]:
        clause=parse(phrase)['clauses'][0]
        add('preserves question/negation subject '+phrase,'role_parsing',clause['subject']==subject and (clause['negation_operator']=='never' if 'never' in phrase else not clause['arguments']))
    ingress=reference_route_fixture(source)
    add('incoming reference is separate from outbound reply routing','provenance',ingress['metadata'].get('referenced_backend_message_id')=='original-message' and ingress['reply_to_backend_id']=='new-message')
    result={'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'scope':'mostly one-speaker synthetic scene/real extracted route; role test uses a second fixture author, not a live participant',
            'live_conversation_quality':'not_assessed','participant_transfer':'not_assessed',
            'durable_learning_gain':'not_assessed','automatic_training':False}
    if version=='V1':result['baseline_revision']=BASELINE
    return result
