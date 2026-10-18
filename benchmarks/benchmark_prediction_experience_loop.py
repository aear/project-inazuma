"""Explicit disposable feasibility trial, not a live Ina learner or promotion test.

Reuses ExperienceCycleEngine and the pure claim evaluator extracted from pinned
Hindsight source, avoiding runtime imports and all live memory access.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import tracemalloc
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_source import historical_module, historical_text
from io_utils import atomic_write_json

BASELINE = '5576e4ed3ff7374c5f7edf25f95ffb96ef996de0'


def evaluator(*, current: bool = False, lifecycle: bool = False):
    source = ((Path(__file__).resolve().parents[1]/'transformers/hindsight_transformer.py').read_text()
              if current else historical_text('transformers/hindsight_transformer.py', BASELINE))
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HindsightTransformer')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ({'evaluate_claims','_numeric_map','compute_hindsight','run','save_prediction_updates'} if lifecycle else {'evaluate_claims','_numeric_map'})]
    shell = ast.ClassDef(name='Evaluator', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    namespace = {"datetime":datetime, "log_to_statusbox":lambda *a:None, "make_origin":lambda *a,**kw:{}, "atomic_write_json":atomic_write_json}
    exec(compile(module, 'pinned-hindsight-pure-evaluator', 'exec'), namespace)
    return namespace['Evaluator']()


def run_trial(root: Path, version: str, *, count: int = 12) -> dict:
    if type(count) is not int or not 3 <= count <= 12:
        raise ValueError('count must be 3..12')
    if version not in {'V1', 'V2'}:
        raise ValueError('choose V1 fixed fixture or V2 residual fixture')
    # Historical source is real; the forecast/observation adapter is this trial.
    root.mkdir(parents=True, exist_ok=True)
    Engine = historical_module('experience_engine.py', BASELINE).ExperienceCycleEngine
    engine = Engine(root_path=root/'cycles', enable_hot=False,
                   config={'current_child':'Fixture','storage_layout':{'fast_runtime_enabled':False}})
    judge = evaluator()
    ledger = root/'fixture-domain-events.jsonl'
    sequence = 0
    def append(event):
        nonlocal sequence
        sequence += 1
        event = {**event, 'sequence':sequence}
        data = (json.dumps(event, sort_keys=True)+'\n').encode()
        with ledger.open('ab') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        return {'id':f'fixture-event:{sequence}', 'kind':event['kind'], 'path':str(ledger)}, hashlib.sha256(data).hexdigest()
    rows, training_residuals = [], []
    cycle = engine.start_cycle('Explicit forecast then observed outcome', domain='prediction_fixture',
                               autonomous_continuation_budget=count-1)
    for index in range(count):
        phase = 'train' if index < count//3 else 'held_out' if index < 2*(count//3) else 'shifted'
        raw = .2 + (index % 3)*.05
        correction = statistics.mean(training_residuals) if version == 'V2' and training_residuals else 0.0
        prediction = raw + correction
        forecast, digest = append({'kind':'forecast', 'cycle_id':cycle['cycle_id'],
            'raw':raw, 'value':prediction, 'phase':phase, 'model':'fixed' if version=='V1' else 'fixture-residual-mean',
            'target':'fixture_scalar','unit':'fixture_units', 'horizon_steps':1,
            'training_records':len(training_residuals), 'evidence_status':'synthetic_fixture'})
        # Outcome generation happens after the forecast has been durably appended.
        actual = raw + (.4 if phase != 'shifted' else -.2)
        observation, _ = append({'kind':'observation','forecast_reference':forecast['id'],
                                  'value':actual,'unit':'fixture_units'})
        assessed = judge.evaluate_claims({'predicted_vector':{'value':prediction},
                                         'observed_vector':{'value':actual}}, {})['value']
        attempt = engine.complete_attempt(cycle['cycle_id'], attempt_reference=forecast,
            observation_references=[observation], evaluation={'schema':'ina.prediction_fixture_evaluation/V1',
            'forecast_sha256':digest,'phase':phase,'value':assessed,
            'uncertainty':'fixture error only; confidence is not calibrated'})
        choice = 'revise' if index < count-1 else 'stop'
        engine.record_choice(cycle['cycle_id'], choice)
        rows.append({'phase':phase,'raw':raw,'prediction':prediction,'actual':actual,
                     'error':assessed['error'],'abs_error':assessed['abs_error'],
                     'cycle_id':cycle['cycle_id'],'attempt_id':attempt['attempt_id']})
        if phase == 'train': training_residuals.append(actual-raw)
        if choice == 'revise':
            cycle = engine.continue_cycle(cycle['cycle_id'], choice='revise',
                intent='Next explicitly budgeted fixture forecast', autonomous=True)
    return {'rows':rows,'phase_mae':{p:statistics.mean(r['abs_error'] for r in rows if r['phase']==p)
            for p in ('train','held_out','shifted')},'forecast_count':count,
            'cycle_count':len(engine.recent_cycles(limit=20)),
            'final_choice':engine.load_cycle(cycle['cycle_id'])['last_choice'],
            'trained_on_held_out':False, 'live_ina_learning':False}


def measure_device(root: Path, samples: int = 3, *, steps: int = 12) -> dict:
    out = {}
    for version in ('V1','V2'):
        results=[]
        for _ in range(samples):
            with tempfile.TemporaryDirectory(prefix='trial-',dir=root) as directory:
                fixture = Path(directory)
                tracemalloc.start(); started=time.perf_counter()
                result=run_trial(fixture,version,count=steps)
                size=0; files=0
                for path in fixture.rglob('*'):
                    if path.is_file():
                        size+=path.stat().st_size;files+=1
                        with path.open('rb') as handle: os.fsync(handle.fileno())
                elapsed=time.perf_counter()-started
                _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                results.append({'seconds':elapsed,'bytes':size,'files':files,'python_peak_bytes':peak,
                                'phase_mae':result['phase_mae'],'cycle_count':result['cycle_count'],
                                'final_choice':result['final_choice']})
        out[version]={'samples':results,'median_seconds':statistics.median(x['seconds'] for x in results)}
    return out


def measure_hindsight_semantics(current: bool = True) -> dict:
    judge=evaluator(current=current,lifecycle=True)
    one={'timestamp':'2026-10-18T00:00:00+00:00','predicted_vector':{'clarity':.2}}
    two={'timestamp':'2026-10-18T00:01:00+00:00','predicted_vector':{'clarity':.8}}
    forecast_only=judge.evaluate_claims(one,two)
    actual=judge.evaluate_claims({**one,'observed_vector':{'clarity':.6}},two)
    data=[one.copy(),two.copy()];saved=[]
    writer=getattr(judge,"save_prediction_updates",None)
    snapshot_bounded=False
    if writer is not None:
        with tempfile.TemporaryDirectory(prefix="ina-prediction-links-") as directory:
            judge.prediction_updates_path=Path(directory)/"links.json"
            writer([{"timestamp":str(i),"prediction_update":{"reference":str(i+1)}} for i in range(101)])
            snapshot=json.loads(judge.prediction_updates_path.read_text())
            snapshot_bounded=len(snapshot["updates"])==100 and not snapshot["contains_observed_outcomes"]
    judge.load_predictions=lambda:data
    judge.load_emotion_log=lambda:[]
    judge.save_predictions=lambda rows:saved.append(rows)
    judge.save_prediction_updates=lambda rows:saved.append(rows)
    trust_calls=[]
    judge.generate_symbolic_tag=lambda *a:('fixture',[])
    judge.compute_emotional_drift=lambda *a:{}
    judge.record_emotional_drift=lambda *a:None
    judge.build_reflection=lambda *a:None
    judge.adjust_trust=lambda rows:(trust_calls.append(rows) or (0.0,0.0))
    judge.save_hindsight_map=lambda rows:None
    # Exercise both actual lifecycle implementations with writes intercepted.
    judge.run()
    return {
        'pending_link_snapshot_is_bounded':snapshot_bounded,
        'next_forecast_not_an_outcome':not forecast_only,
        'explicit_outcome_still_evaluated':actual['clarity']['actual']==.6,
        'original_forecast_preserved':data[0]['predicted_vector']=={'clarity':.2},
        'revision_link_is_not_observation':bool(data[0].get('prediction_update'))
            and data[0]['prediction_update']['is_observation'] is False,
        'missing_observation_remains_pending':data[0].get('outcome_status')=='awaiting_observation',
        'forecast_updates_saved_without_lessons':len(saved)==1 and not trust_calls,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='explicit disposable fixture parent only')
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--steps',type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.samples<=5: parser.error('samples must be 1..5')
    print(json.dumps({'benchmark':'prediction_experience_loop','benchmark_version':'V1',
        'baseline_revision':BASELINE,'results':measure_device(args.root,args.samples,steps=args.steps),
        'scope':'synthetic domain adapter; existing engine and pinned pure Hindsight scoring',
        'unavailable':['live Ina learning','probability calibration','missing-outcome/security tests while Ina running',
                       'end-to-end responsiveness','long-term retention costs']},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
