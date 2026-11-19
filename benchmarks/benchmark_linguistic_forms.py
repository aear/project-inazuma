"""Offline grammatical minimal pairs; no live learning, thoughts or broadcasts."""
import os
BASELINE='b3f56def43d3f2825da38f26cb89a414a27a2210'


def selected_event(**changes):
    event={'id':'c0','agent':'Mira','predicate':'want','arguments':{'theme':'the compass'},
           'tense':'present','negated':False,'voice':'active','inverted':False}
    return {**event,**changes}


def assess(version):
    if os.environ.get('INA_OFFLINE_LINGUISTIC_TESTS')!='1':
        return {'correct':0,'total':0,'cases':[],'status':'blocked','scope':'requires verified complete shutdown and isolated fixtures','baseline_revision':BASELINE if version=='V1' else None}
    if version=='V1':
        from historical_source import historical_module
        language=historical_module('language_intelligence.py',BASELINE)
        compose=historical_module('english_composition.py',BASELINE).compose_selected_english
    else:
        import language_intelligence as language
        from english_composition import compose_selected_english as compose
    from expression_core import create_expression_intent
    rows=[]
    def add(name,component,ok):rows.append({'case':name,'component':component,'correct':bool(ok)})
    forms=[
      ('Mira teaches Rowan.','mira','teach','present',False,None),
      ('Rowan tries again.','rowan','try','present',False,None),
      ('Mira wishes for rain.','mira','wish','present',False,None),
      ('Rowan goes home.','rowan','go','present',False,None),
      ('Mira changes the plan.','mira','change','present',False,None),
      ('She understood the message.','she','understand','past',False,None),
      ('They knew the route.','they','know','past',False,None),
      ("We're here.",'we','be','present',False,None),
      ("They aren't here.",'they','be','present',True,None),
      ("We weren't there.",'we','be','past',True,None),
      ("She couldn't open the gate.",'she','open','present',True,'could'),
      ("They mustn't open the gate.",'they','open','present',True,'must'),
      ("We'll learn the route.",'we','learn','present',False,'will'),
      ("They’ll find the compass.",'they','find','present',False,'will'),
    ]
    for text,agent,predicate,tense,negative,modal in forms:
        c=language.analyze_utterance(text)['clauses'][0]
        add(text,'morphology',tuple(c.get(k) for k in ('subject','predicate','tense','negated','modality'))==(agent,predicate,tense,negative,modal))
    for text in ('The wishes change.','The changes help.'):
        c=language.analyze_utterance(text)['clauses'][0]
        add('nominal role '+text,'constructions',c.get('subject')==' '.join(text.lower().split()[:2]) and c.get('predicate')==('change' if 'wishes' in text else 'help'))
    a=language.analyze_utterance("I don't think she understood the message.")['clauses']
    add('matrix negation stays out of embedded content','scope',len(a)==2 and [c['negated'] for c in a]==[True,False] and a[1]['predicate']=='understand' and a[1].get('embedded_content_is_fact') is False)
    a=language.analyze_utterance('Mira understood that Rowan wants the compass.')['clauses']
    add('understanding report retains embedding','scope',len(a)==2 and a[0]['predicate']=='understand' and a[1].get('embedded_content_is_fact') is False)
    a=language.analyze_utterance('Mira teaches Rowan. Rowan tries again.')['clauses']
    add('roles stay distinct across a short reading span','reading_span',len(a)==2 and [c['subject'] for c in a]==['mira','rowan'] and [c['predicate'] for c in a]==['teach','try'])
    a=language.analyze_utterance('Rowan tries tea or coffee.')['clauses'][0]
    add('alternatives are retained without choosing','whole_utterance',a['predicate']=='try' and len(a.get('alternatives') or [])==1 and a['alternatives'][0]['exclusivity']=='unresolved')
    tokens=language.morphology('She couldn’t open it.')
    parts=[t for t in tokens if t['surface']=='couldn’t']
    add('contracted spans preserve their original characters','provenance',len(parts)==2 and [p['normalized'] for p in parts]==['could','not'] and parts[0]['source_span']==parts[1]['source_span']==[4,12])
    intent=create_expression_intent('synthetic chosen question',meaning_references=['meaning:fixture'],allowed_media=['text'])
    fixtures=[
      (selected_event(agent='the lantern',predicate='be',arguments={},question_word='where',inverted=True),'Where is the lantern?'),
      (selected_event(arguments={},question_word='what',inverted=True),'What does Mira want?'),
      (selected_event(agent='who',predicate='see',arguments={'theme':'the comet'},tense='past',question_word='who'),'Who saw the comet?'),
      (selected_event(agent='she',predicate='see',arguments={},tense='past',question_word='who',inverted=True),'Who did she see?'),
      (selected_event(agent='we',predicate='go',arguments={},question_word='when',inverted=True),'When do we go?'),
      (selected_event(agent='you',predicate='open',arguments={'theme':'the gate'},question_word='how',inverted=True),'How do you open the gate?'),
      (selected_event(agent='she',predicate='open',arguments={'theme':'the gate'},negated=True,question_word='why',inverted=True),'Why does she not open the gate?'),
    ]
    for event,expected in fixtures:
        value=compose({'version':1,'events':[event]},intent)
        add('chosen expression '+expected,'expression',value.get('candidate_text')==expected and value.get('outward_text') is None and value.get('automatic_expression') is False)
    for choice in ('native_only','silence','abstain','clarify'):
        r=compose({'version':1,'events':[fixtures[0][0]]},intent,choice=choice)
        add('preserve '+choice,'agency',r['status']=='choice_preserved' and r['candidate_text'] is None)
    bad=selected_event(question_word='what',inverted=True) # filled theme, unresolved question slot
    r=compose({'version':1,'events':[bad]},intent)
    add('unresolved question role is not dropped','uncertainty',r['status']=='unsupported_structure' and r['candidate_text'] is None)
    r=compose({'version':1,'events':[selected_event(question_word='which',inverted=True)]},intent)
    add('unsupported selection domain remains unresolved','uncertainty',r['status']=='unsupported_structure' and r['candidate_text'] is None)
    for text in ("Mira's compass.","I'd go.","She's known."):
        tokens=language.morphology(text)
        add('ambiguous contraction remains unexpanded '+text,'uncertainty',all(not t['features'].get('contraction') for t in tokens))
    from benchmarks.benchmark_conversation_grounding import scene_for
    from conversation_grounding import plan_conversation_reply
    q='Who teaches Rowan?'
    quoted=plan_conversation_reply(q,scene_for(['"Mira teaches Rowan."'],q))
    add('quoted wording is not asserted conversational evidence','pragmatics',quoted['answer_candidate'] is None)
    from english_learning_profile import build_english_learning_profile
    profile=(historical_module('english_learning_profile.py',BASELINE).build_english_learning_profile() if version=='V1' else build_english_learning_profile())
    forms=[g for g in profile['goals'] if g['id']=='forms']
    add('optional forms goal has no attributed adoption or performance','goals',bool(forms) and forms[0]['optional'] and forms[0]['adoption']=='not_observed' and forms[0]['ina_performance']=='not_assessed' and profile['global_score'] is None)
    return {'correct':sum(bool(r['correct']) for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'implementation grammar and selected-meaning fixtures; no live comprehension, social use, learning or global capability score'}
