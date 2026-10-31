"""Pinned numeric, replay-learning and cognition comparisons; no live state."""
from __future__ import annotations
import math,inspect
BASELINE='aefcc6e7285872ffa239602309147ff56e550f1d'


def assess(version):
    if version=='V1':
        from historical_source import historical_module
        kernels=historical_module('ina_ml/kernels.py',BASELINE)
        decoder=historical_module('transformers/QTransformer.py',BASELINE)
        cognition=historical_module('experience_cognition.py',BASELINE)
        thought=historical_module('thought_processor.py',BASELINE)
    else:
        import ina_ml.kernels as kernels
        import importlib
        decoder=importlib.import_module('transformers.QTransformer')
        import experience_cognition as cognition
        import thought_processor as thought
    rows=[]
    def add(name,component,correct):rows.append({'case':name,'component':component,'correct':bool(correct)})
    for values in ([1e308,1e308],[1e-300,1e-300],[5e-324,5e-324]):
        norm=kernels.normalize_vector(values)
        add('unit direction across magnitude '+str(values[0]),'numerics',all(math.isfinite(v) for v in norm) and math.isclose(math.hypot(*norm),1,rel_tol=1e-12))
    for left,right,want in [([1e308,1e308],[1e308,1e308],1.),([1e-300],[1e-300],1.),([1e308,1e308],[1e308,-1e308],0.),([1.,0.],[-1.,0.],-1.)]:
        try:actual=kernels.cosine_similarity(left,right,epsilon=0.)
        except (ValueError,OverflowError,ZeroDivisionError):actual=math.nan
        add('stable cosine '+str(left)+str(right),'numerics',math.isfinite(actual) and math.isclose(actual,want,abs_tol=1e-12))
    add('ordinary normalization preserved','compatibility',kernels.normalize_vector([3,4])==[.6,.8])
    add('unequal length norms preserved','compatibility',math.isclose(kernels.cosine_similarity([1.],[1.,1.]),1/(math.sqrt(2)+1e-8)))
    add('overlap semantics preserved','compatibility',math.isclose(kernels.cosine_similarity([1.],[1.,1.],overlap_norms=True),1/(1+1e-8)))
    add('absolute epsilon damping preserved','compatibility',math.isclose(kernels.cosine_similarity([1e-100],[1e-100]),1e-192,rel_tol=1e-10,abs_tol=0.))
    for invalid in (math.nan,math.inf):
        score=kernels.cosine_similarity([invalid],[1.])
        add('nonfinite vector cannot establish similarity '+str(invalid),'numerics',score==0.)
    model=decoder.QTransformer();supports='source_event' in inspect.signature(model.learn_mapping).parameters
    def learn(model,label,event_id):
        kw={'poetic_word':label,'persist':False}
        if supports:kw['source_event']=event_id
        model.learn_mapping('000000000',**kw)
    learn(model,'lantern','event:1');learn(model,'lantern','event:1')
    add('identified event only learned once','learning_provenance',model.decoder_stats['words']['000']['lantern']==1.)
    learn(model,'compass','event:1')
    add('conflicting replay cannot rewrite the first event','learning_provenance','compass' not in model.decoder_stats['words']['000'])
    tied=decoder.QTransformer();learn(tied,'alpha','a');learn(tied,'omega','b')
    add('equal observed mappings remain unresolved','ambiguity',tied._learned('words','000') is None)
    output=tied.collapse_to_meaning('000000000')
    add('ambiguous mapping is not labelled learned','ambiguity',not output['learned_fields']['poetic_word'] and output.get('mapping_assessments',{}).get('words',{}).get('status')=='ambiguous')
    add('conflict does not become an arbitrary scaffold word','ambiguity',output['poetic_word']=='???')
    single=decoder.QTransformer();learn(single,'harbour','single')
    add('single learned mapping remains available','compatibility',single._learned('words','000')=='harbour')
    # Fresh evaluation of a bucket after repeated delivery of one wrong example.
    trained=decoder.QTransformer()
    for _ in range(20):learn(trained,'wrong-route','repeat')
    for i in range(4):learn(trained,'observed-route','independent:'+str(i))
    add('replay-resistant bucket prediction on a held-out query','learning_trial',trained._learned('words','000')=='observed-route')
    p=thought.ThoughtProcessor();t=p.process_non_linguistic({'cue':'one'},confidence=1.,relevance=1.)
    data=[{'option':'act','thought_id':t.thought_id,'weight':1.}]*20+[{'option':'wait','thought_id':t.thought_id,'weight':1.}]
    d=p.decide(['act','wait'],[t],evidence=data)
    add('repetition is not independent decision support','decision',d.selected is None and d.status=='undecided')
    d=p.decide(['act','wait'],[t],evidence=[{'option':'act','thought_id':t.thought_id,'weight':1.},{'option':'act','thought_id':t.thought_id,'weight':-1.}])
    add('opposing readings remain visible','decision',len(d.evidence)==2 and d.scores['act']==0.)
    event={'candidate_answer':'candidate','evidence':{'causal':['a'],'sensory':['b']},'evidence_origins':{'a':'same','b':' same '}}
    add('whitespace variants do not create another origin','provenance',cognition.assess_uncertainty(event)['status']=='uncertain')
    event['evidence_origins']={'a':'first','b':'second'}
    add('distinct supplied origins still support the existing state','compatibility',cognition.assess_uncertainty(event)['status']=='known')
    candidate={'state_id':'derived','relevance':1,'confidence':1,'salience':1,'novelty':1,'independent_witnesses':2,'source_references':[None,{}]}
    gate=cognition.gate_transient_state([candidate])['candidates'][0]
    add('malformed references cannot propagate derived state','provenance',gate['action']=='hold' and not gate['source_references'])
    pred=cognition.build_multi_horizon_predictions([{'horizon':'near','prediction':'candidate','source_references':[None], 'disconfirming_observations':['different outcome']}])
    add('predictions require actual reference strings','provenance',not pred['horizons']['near'])
    lenses=cognition.inspect_attention_lenses({'evidence':{'causal':[None,{},'  ']}})
    add('attention evidence retains only reference strings','provenance',next(r for r in lenses if r['lens']=='causal')['evidence_references']==[])
    try:signals=cognition.bounded_signals({'signals':{'novelty':10**400}})
    except OverflowError:signals={}
    add('oversized numeric signal stays bounded','bounds',signals.get('novelty')==0.)
    d=p.decide(['act','wait'],[t],evidence=[{'option':'act','thought_id':t.thought_id,'weight':1.,'reason':r} for r in ('first description','second description')])
    add('descriptions survive without multiplying a shared thought','decision',d.scores['act']==1. and len(d.evidence[0].get('reason_observations',[]))==2)
    return {'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'finite numerical and synthetic decoder/decision checks; no Ina learned state or live calibration',
            'replay_scope':'optional caller event ids; 512-event per-object window, not persisted',
            'learning_trial':'one synthetic bit bucket and caller-declared examples; not broad ML transfer evidence'}
