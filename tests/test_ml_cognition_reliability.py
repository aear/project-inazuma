import math,copy,random
import pytest
from benchmarks.benchmark_ml_cognition_reliability import assess
from ina_ml import cosine_similarity,normalize_vector
from transformers.QTransformer import QTransformer
from thought_processor import ThoughtProcessor


def test_candidate_versioned_contract():
    r=assess('V2');assert r['correct']==r['total'],[c for c in r['cases'] if not c['correct']]


def test_shared_kernels_against_decimal_and_ordinary_reference():
    from decimal import Decimal,localcontext
    for left,right in [([1e300,2e300],[2e300,1e300]),([1e-300,2e-300],[2e-300,1e-300])]:
        with localcontext() as context:
            context.prec=80
            a,b=[Decimal(str(x)) for x in left],[Decimal(str(x)) for x in right]
            expected=float(sum(x*y for x,y in zip(a,b))/(sum(x*x for x in a).sqrt()*sum(y*y for y in b).sqrt()))
        assert math.isclose(cosine_similarity(left,right,epsilon=0.),expected,rel_tol=1e-12)
    rng=random.Random(101)
    for _ in range(40):
        a=[rng.uniform(-2,2) for _ in range(16)];b=[rng.uniform(-2,2) for _ in range(16)]
        old=sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b))+1e-8)
        assert math.isclose(cosine_similarity(a,b),old,abs_tol=1e-12)
    assert normalize_vector([0,0])==[0.,0.]
    for epsilon in (-1.,math.nan,math.inf):
        with pytest.raises(ValueError):cosine_similarity([1.],[1.],epsilon=epsilon)


def test_learning_replay_conflict_and_weight_rejection_are_transactional():
    q=QTransformer();q.learn_mapping('000000000',poetic_word='lantern',source_event='one',persist=False)
    before=copy.deepcopy(q.decoder_stats)
    q.learn_mapping('000000000',poetic_word='compass',source_event='one',persist=False)
    assert q.decoder_stats==before and q.last_learning_update['status']=='conflicting_replay'
    for weight in (math.nan,math.inf,-1.):
        with pytest.raises(ValueError):q.learn_mapping('000000000',poetic_word='bad',weight=weight,persist=False)
        assert q.decoder_stats==before
    with pytest.raises(ValueError):q.learn_mapping('invalid',poetic_word='bad',persist=False)
    with pytest.raises(ValueError):q.learn_mapping('000000000',tags='letters',persist=False)


def test_bounded_replay_window_and_legacy_statistics_reload(tmp_path):
    q=QTransformer(decoder_path=tmp_path/'decoder.json')
    q.learn_mapping('000000000',poetic_word='harbour',source_event='first')
    restored=QTransformer(decoder_path=q.decoder_path)
    assert restored.decoder_stats==q.decoder_stats and restored._learned('words','000')=='harbour'
    assert not restored._learning_events  # Scope is deliberately process-local.
    for i in range(513):q.learn_mapping('000000000',poetic_word='harbour',source_event='event:'+str(i),persist=False)
    assert len(q._learning_events)==512 and 'first' not in q._learning_events


def test_learning_failure_rolls_back_without_consuming_event(monkeypatch):
    q=QTransformer();before=copy.deepcopy(q.decoder_stats)
    def fail():raise OSError('fixture write unavailable')
    monkeypatch.setattr(q,'_save_decoder_stats',fail)
    with pytest.raises(OSError):q.learn_mapping('000000000',poetic_word='harbour',source_event='one')
    assert q.decoder_stats==before and 'one' not in q._learning_events
    q.learn_mapping('000000000',poetic_word='harbour',source_event='one',persist=False)
    assert q.last_learning_update['updated']


def test_concurrent_same_event_only_admitted_once():
    from concurrent.futures import ThreadPoolExecutor
    q=QTransformer()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(q.learn_mapping,'000000000',poetic_word='harbour',source_event='one',persist=False) for _ in range(12)]
        for job in jobs:job.result()
    assert q.decoder_stats['words']['000']['harbour']==1.


def test_decision_iterators_are_bounded_and_signed_readings_not_multiplied():
    p=ThoughtProcessor();t=p.process_non_linguistic({'fixture':1},confidence=1.)
    seen=[]
    def evidence():
        for i in range(1000):seen.append(i);yield {'option':'wait','thought_id':t.thought_id,'weight':1.}
    result=p.decide(['wait','act'],[t],evidence=evidence())
    assert len(seen)==129 and result.evidence_summary['input_truncated']
    assert result.scores['wait']==1. and result.evidence_summary['duplicate_rows_ignored']==127
    result=p.decide(['wait','act'],[t],evidence=[{'option':'wait','thought_id':t.thought_id,'weight':w} for w in (1.,.5,.5,.5)])
    assert result.scores['wait']==.75


def test_ambiguity_and_capacity_do_not_invent_a_learned_choice():
    q=QTransformer(decoder_stats={'words':{'000':{'a':1.,'b':1.}}})
    assert q._learned('words','000') is None
    assert q.collapse_to_meaning('000000000')['poetic_word']=='???'
    q=QTransformer(decoder_stats={'words':{'000':{str(i):1. for i in range(65)}}})
    assert q._mapping_assessment('words','000')['status']=='capacity_unresolved'
    q.learn_mapping('000000000',poetic_word='new',persist=False)
    assert q.last_learning_update['status']=='capacity_reached'


def test_distinct_reasons_survive_even_when_the_same_thought_is_not_counted_twice():
    p=ThoughtProcessor();t=p.process_non_linguistic({'fixture':1},confidence=1.)
    result=p.decide(['wait','act'],[t],evidence=[{'option':'wait','thought_id':t.thought_id,'weight':1.,'reason':reason} for reason in ('blocked door','same observation in another modality')])
    assert result.scores['wait']==1.
    assert result.evidence[0]['reason_observations']==['blocked door','same observation in another modality']
