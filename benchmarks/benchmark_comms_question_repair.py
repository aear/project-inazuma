"""Synthetic, offline-only repair offers; no actual chat or thought contents."""
import os
from pathlib import Path
BASELINE='b3f56def43d3f2825da38f26cb89a414a27a2210'


def cases(plan):
    from benchmarks.benchmark_conversation_grounding import scene_for
    rows=[]
    def add(case,ok,component):rows.append({'case':case,'correct':bool(ok),'component':component})
    for text in ('Could you explain what you mean?', 'Why does the comet shimmer?', 'Can you hear this message?'):
        p=plan(text,scene_for([],text))
        add('unsupported direct question offers a repair: '+text,p['answer_candidate'] is None and bool(p['clarification_candidate']),'repair')
    p=plan('What do I like?',scene_for([],'What do I like?'))
    add('missing prior evidence permits a context request',p['answer_candidate'] is None and bool(p['clarification_candidate']),'repair')
    p=plan('Why does the comet shimmer?',{})
    add('missing discourse can offer rephrasing without invented referents',p['answer_candidate'] is None and bool(p['clarification_candidate']),'repair')
    for text in ('Morning, Ina. <3','She asked why the comet shimmers.', '"Why does the comet shimmer?"', 'No need to answer.', "Don't answer this question."):
        p=plan(text,scene_for([],text))
        add('ordinary/quoted/withheld input does not trigger repair: '+text,p['answer_candidate'] is None and p['clarification_candidate'] is None,'scope')
    for text,scene in [('x'*513,{}),('Why?',{'turns':[],'discourse':{}})]:
        p=plan(text,scene)
        add('input remains bounded or has only a repair',p['answer_candidate'] is None,'bounds')
    q='Where is the compass?';p=plan(q,scene_for(['The compass is under the desk.'],q))
    add('source-backed answer retains priority',p['answer_candidate']=='You said that the compass is under the desk.' and p['clarification_candidate'] is None,'answers')
    return rows


def assess(version):
    if os.environ.get('INA_OFFLINE_COMMS_TESTS')!='1':
        return {'correct':0,'total':0,'cases':[],'status':'blocked',
                'scope':'requires complete host shutdown and isolated synthetic fixtures',
                'baseline_revision':BASELINE if version=='V1' else None}
    if version=='V1':
        from historical_source import historical_module
        plan=historical_module('conversation_grounding.py',BASELINE).plan_conversation_reply
    else:
        from conversation_grounding import plan_conversation_reply as plan
    rows=cases(plan)
    return {'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'planner candidates on synthetic inputs; no claim of live reply, willingness or actual comprehension'}
