"""Optional English practice targets, separated from implementation evidence."""
from __future__ import annotations
from collections.abc import Mapping

GOALS=(
 ('scope','Keep who believes or says something separate from whether it happened.', 'comprehension', ['english_workbench']),
 ('roles','Keep speaker, listener, nickname, possession and negation consistent.', 'comprehension', ['language_components','discourse','self_name_reference']),
 ('surface','Retain punctuation and spelling alternatives without silently changing meaning.', 'comprehension', ['character_language']),
 ('composition','Choose one grounded meaning, review an English candidate, then keep, revise or decline it.', 'expression', ['english_workbench','expression_core']),
 ('transfer','Try unfamiliar wording and predicates using held-out material.', 'transfer', ['language_transfer_limits']),
 ('listener','Ask a willing listener what meaning they recovered from one chosen expression.', 'human_quality', []),
 ('external_tasks','Choose a small held-out reading or instruction-following trial.', 'external_assessment', []),
)


def build_english_learning_profile(results=()):
    if not isinstance(results,(list,tuple)):raise ValueError('bounded implementation result list required')
    if len(results)>32:raise ValueError('result budget')
    index={}
    for row in results:
        if isinstance(row,Mapping) and isinstance(row.get('module'),str):index[row['module']]=row
    goals=[]
    for key,description,dimension,modules in GOALS:
        evidence=[]
        for name in modules:
            row=index.get(name)
            if row is not None:evidence.append({'module':name,'version':row.get('version'),
                'correct':row.get('correct'),'total':row.get('total'),'scope':'implementation_fixtures',
                'components':dict(row.get('components') or {})})
        measured=bool(evidence) and len(evidence)==len(modules)
        limitation=any(e['correct']!=e['total'] for e in evidence)
        goals.append({'id':key,'description':description,'dimension':dimension,
            'implementation_status':'limitations_retained' if limitation else 'fixtures_available' if measured else 'not_assessed',
            'implementation_evidence':evidence,'ina_performance':'not_assessed',
            'optional':True,'adoption':'not_observed','continuation_budget':0})
    return {'schema':'ina.english_learning_profile/V1','subject':'current_implementation',
        'invitation_author':'operator_requested_support','goals':goals,
        'choices':['try_one','revise','keep','defer','decline','native_only','clarify','silence'],
        'automatic_practice':False,'automatic_learning':False,'language_reward':False,
        'global_score':None,'human_quality':'not_assessed','live_social_expression':'not_assessed',
        'external_tasks':'not_assessed','promotion_authorized':False}
