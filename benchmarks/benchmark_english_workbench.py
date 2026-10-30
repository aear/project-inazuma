"""Finite composition/scope checks; neither a learned-state nor leaderboard score."""
from __future__ import annotations
BASELINE='6d200475d5b444550819f7c23a1e595518240cf6'


def selected(events):
    from expression_core import create_expression_intent
    return {'version':1,'events':events},create_expression_intent('chosen expression',
        meaning_references=['meaning:fixture'],allowed_media=['text'],provenance=['fixture:selected_meaning'])


def event(**kw):
    return {'id':'c0','agent':{'id':'self','name':'Ina','is_self':True},'predicate':'want',
            'arguments':{'theme':'the lantern'},'tense':'present','modality':None,
            'negated':False,'voice':'active',**kw}


def assess(version):
    if version=='V1':
        from historical_source import historical_module
        analyze=historical_module('language_intelligence.py',BASELINE).analyze_utterance
        compose=None
    else:
        from language_intelligence import analyze_utterance as analyze
        from english_composition import compose_selected_english as compose
    rows=[]
    def add(name,component,correct):rows.append({'case':name,'component':component,'correct':bool(correct)})
    for text,expected in [("I don't think she wants the lantern.",[True,False]),
                          ("I think she doesn't want the lantern.",[False,True]),
                          ('Mira believes that Rowan knows she opened it.',[False,False,False]),
                          ('I told you she did not keep it.',[False,True])]:
        clauses=analyze(text)['clauses']
        add(text,'comprehension_scope',[c['negated'] for c in clauses]==expected and len(clauses)==len(expected))
    c=analyze('Mira believes that Rowan opened it.')['clauses']
    add('belief content is not promoted to observed fact','comprehension_scope',len(c)==2 and c[1].get('clausal_relation')=='attitude_content' and c[1].get('embedded_content_is_fact') is False)
    for text in ('Do you want tea or coffee?','I might choose the lantern or the compass.'):
        c=analyze(text)['clauses'][0];options=c.get('alternatives') or []
        add(text,'alternatives',len(options)==1 and len(options[0]['options'])==2 and options[0]['exclusivity']=='unresolved')
    c=analyze('I do not want tea or coffee.')['clauses'][0]
    add('negative alternative scope remains uncertain','alternatives',bool(c.get('alternatives')) and c['alternatives'][0]['negation_scope']=='unresolved')
    fixtures=[
      ([event()], 'I want the lantern.'),
      ([event(agent={'id':'mira','name':'Mira'},negated=True)], 'Mira does not want the lantern.'),
      ([event(predicate='give',arguments={'recipient':'you','theme':'the compass'},tense='past',negated=True)],'I did not give you the compass.'),
      ([event(predicate='open',modality='might',negated=True,arguments={'theme':'the gate'})],'I might not open the gate.'),
      ([event(agent='you',inverted=True,arguments={'theme':'tea or coffee'})],'Do you want tea or coffee?'),
      ([event(predicate='be',arguments={'theme':'here'})],'I am here.'),
      ([event(agent=None,predicate='find',voice='passive',tense='past',arguments={'theme':'the compass'})],'The compass was found.'),
      ([event(predicate='think',arguments={},negated=True),event(id='c1',agent='she',parent_clause='c0',clausal_relation='attitude_content')],'I do not think that she wants the lantern.'),
    ]
    for events,expected in fixtures:
        native,intent=selected(events)
        result=compose(native,intent) if compose else {}
        add('compose '+expected,'expression_composition',result.get('candidate_text')==expected)
        add('candidate stays private '+expected,'expression_choice',bool(result) and result.get('outward_text') is None and result.get('automatic_expression') is False)
    native,intent=selected([event()])
    for choice in ('native_only','silence','clarify','abstain'):
        result=compose(native,intent,choice=choice) if compose else {}
        add('preserve '+choice,'expression_choice',result.get('status')=='choice_preserved' and result.get('candidate_text') is None)
    for status,modal,expected in [('unknown',None,'needs_evidence'),('uncertain',None,'needs_chosen_qualification'),('uncertain','might','candidate_for_review'),('uncertain','will','needs_chosen_qualification'),('uncertain','must','needs_chosen_qualification')]:
        plan={'schema':'ina.experience_cognition/V3','epistemic_state':{'status':status}}
        n,i=selected([event(modality=modal)]);r=compose(n,i,cognition_plan=plan) if compose else {}
        add('preserve epistemic '+status+' '+str(modal),'calibration',r.get('status')==expected)
    n,i=selected([event(predicate='unprovided_verb')]);r=compose(n,i) if compose else {}
    add('unsupported meaning does not yield invented wording','expression_bounds',r.get('status')=='unsupported_structure' and r.get('candidate_text') is None)
    from english_learning_profile import build_english_learning_profile
    profile=build_english_learning_profile([{'module':'language_transfer_limits','version':'V2','correct':1,'total':5}])
    add('goal profile retains measured limitations','goals',version=='V2' and next(g for g in profile['goals'] if g['id']=='transfer')['implementation_status']=='limitations_retained')
    add('goals are optional and not attributed to Ina','goals',version=='V2' and all(g['optional'] and g['adoption']=='not_observed' for g in profile['goals']) and profile['global_score'] is None and profile['human_quality']=='not_assessed')
    import ast,json,tempfile
    from pathlib import Path
    from datetime import datetime,timezone
    root=Path(__file__).resolve().parents[1]
    if version=='V1':
        from historical_source import historical_text
        source=historical_text('monitoring_dashboard.py',BASELINE)
        tool_source=historical_text('personal_tool_runtime.py',BASELINE)
    else:
        source=(root/'monitoring_dashboard.py').read_text()
        tool_source=(root/'personal_tool_runtime.py').read_text()
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in {'_developmental_readiness','_safe_json','_modified'}]
    env={'Path':Path,'Any':object,'json':json,'MAX_JSON_BYTES':32*1024*1024,'datetime':datetime,'timezone':timezone}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<copied Observatory>','exec'),env)
    with tempfile.TemporaryDirectory() as directory:
        target=Path(directory)/'assessment.json'
        target.write_text(json.dumps({'assessments':{'language_english':{'status':'complete','learning_profile':profile}}}))
        _,display=env['_developmental_readiness'](Path(directory)/'none',Path(directory)/'none2',target)
        goal_rows=[r for r in display if r[0].startswith('English goal')]
        add('Observatory shows optional goals without claiming Ina performance','reporting',len(goal_rows)==7 and all('Ina not assessed' in r[2] for r in goal_rows))
    add('private selected-meaning and optional-goal tools available','availability','english_compose_selected' in tool_source and 'english_learning_options' in tool_source)
    return {'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'baseline_scope':'existing grammar; selected-meaning composition API absent in V1',
            'scope':'synthetic selected-meaning realisation and grammar; not Ina learning or live English performance'}
