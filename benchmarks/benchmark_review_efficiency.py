"""Explicit bounded V1/V2 comparisons in disposable storage; no live-runtime calls."""
from __future__ import annotations
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import statistics
import platform
import sys
import tempfile
import time
from unittest.mock import patch

from historical_source import historical_module, historical_text

BASELINE = 'e13c6b4517f4df9859148a5f96cca0b4814bb9d0'


def _module(path, version):
    if version == 'V1':
        return historical_module(path, BASELINE)
    return __import__(path[:-3])


def monitor_fixture(version):
    """Exercise the actual caller body with fixed inputs and no manager import."""
    from types import SimpleNamespace
    source = historical_text('model_manager.py', BASELINE) if version == 'V1' else Path('model_manager.py').read_text()
    tree = ast.parse(source)
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'monitor_hunger', '_clamp01'}]
    initial = {'emotion_snapshot': {'joy': .2, 'stress': .1, 'intensity': .2},
               'hunger_level': .6, 'fitness_level': .7, 'current_energy': .5,
               'sleep_pressure': .1, 'last_hunger_update_ts': 90.0}
    current = dict(initial); counts = {'reads':0, 'writes':0}
    def get(key, default=None):
        counts['reads'] += 1; return current.get(key, default)
    def snapshot(**kw):
        counts['reads'] += 1; return dict(current)
    def update(key, value):
        counts['writes'] += 1; current[key] = value
    def many(values, **kw):
        counts['writes'] += 1; current.update(values)
    ns = {'Any':object, 'time':SimpleNamespace(time=lambda:100.0), 'CHILD':'Fixture',
          'get_inastate':get, 'get_inastate_snapshot':snapshot,
          'update_inastate':update, 'update_inastate_many':many,
          '_load_offer_store':lambda now:{}, '_active_offers':lambda store:[],
          '_build_nutrition_options_with_offers':lambda *a:[],
          '_update_nutrition_snapshot':lambda *a, **k:None, '_maybe_auto_meal':lambda *a:None,
          '_NUTRITION_TARGET':.64, '_NUTRITION_DECAY_PER_SEC':.000045, '_NUTRITION_MIN':.05}
    exec(compile(ast.Module(body=body,type_ignores=[]),'monitor_fixture','exec'),ns)
    ns['monitor_hunger']()
    return current, counts


def evaluate(version='V2', *, fixture_root: Path | None = None):
    if version not in {'V1', 'V2'}:
        raise ValueError('version must be V1 or V2')
    language = _module('language_intelligence.py', version)
    state = _module('runtime_state.py', version)
    memory = _module('memory_index.py', version)
    guidance = _module('learned_media_lessons.py', version)
    cases = []
    measures = {}
    caller, caller_counts = monitor_fixture(version)
    reference, _ = monitor_fixture('V1')
    def record(name, component, passed, **evidence):
        cases.append({'case': name, 'component': component, 'correct': bool(passed), **evidence})
    record('hunger caller retains values', 'state_correctness', caller == reference)
    record('hunger caller groups its reads and writes', 'state_resources', caller_counts == {'reads':1,'writes':1})
    measures['hunger_caller_io'] = caller_counts
    for text, predicate, subject in [('I run.', 'run', 'i'), ('She ran.', 'run', 'she'),
                                     ('We are running.', 'run', 'we'), ('I do not run.', 'run', 'i')]:
        clause = language.analyze_utterance(text)['clauses'][0]
        record(text, 'predicate', clause['predicate'] == predicate and clause['subject'] == subject,
               actual=clause, expected={'predicate': predicate, 'subject': subject})
    for text, predicate in [('I am happy to run.', 'be'), ('I have a book to give.', 'have'), ('I have not opened it.', 'open')]:
        actual = language.analyze_utterance(text)['clauses'][0]['predicate']
        record(text, 'auxiliary_compatibility', actual == predicate, actual=actual, expected=predicate)
    for text in ['I give her a book.', 'I give a book to her.', 'They gave a key to us.']:
        args = language.analyze_utterance(text)['clauses'][0]['arguments']
        target = {'recipient': 'us', 'theme': 'a key'} if text.startswith('They') else {'recipient': 'her', 'theme': 'a book'}
        record(text, 'construction', args == target, actual=args, expected=target)
    for text, quantity in [('I opened 3 files.', '3'), ('You gave 2.5 coins to me.', '2.5')]:
        tokens = language.morphology(text)
        record(text, 'quantity', quantity in [t['normalized'] for t in tokens])
    unknown = language.analyze_utterance('I frobnicate widgets.')['uncertainty']['morphology']
    record('unsupported construction remains uncertain', 'uncertainty', unknown['confidence'] < .9, actual=unknown)
    with tempfile.TemporaryDirectory(prefix='ina_review_benchmark_', dir=fixture_root) as temp:
        root = Path(temp)
        source = root/'map.json'; db = root/'map.sqlite'
        rows = {f'f{i:05}': {'mtime_ns': i//2, 'tags': ['sample'] if i%2 else []} for i in range(10000)}
        source.write_text(json.dumps(rows))
        memory.ensure_memory_index_db(source, db)
        with sqlite3.connect(db) as conn:
            sql = 'SELECT frag_id,tier,filename,mtime_ns,tags_json FROM fragments ORDER BY mtime_ns DESC, frag_id DESC LIMIT 32'
            plan = [str(row) for row in conn.execute('EXPLAIN QUERY PLAN '+sql)]
            samples = []
            for _ in range(7):
                started = time.perf_counter(); result = conn.execute(sql).fetchall()
                samples.append(time.perf_counter()-started)
            measures['memory_lookup'] = {'seconds': samples, 'median_seconds': statistics.median(samples),
                'database_bytes': db.stat().st_size, 'query_plan': plan, 'rows': 10000, 'fixture': 'synthetic'}
        expected = sorted(rows, key=lambda k: (rows[k]['mtime_ns'], k), reverse=True)[:32]
        record('recent identities and tie ordering', 'memory_correctness', [r[0] for r in result] == expected)
        record('ordered query avoids temporary sort', 'memory_resources', not any('TEMP B-TREE' in row for row in plan))
        path = root/'Fixture/memory/learned_media_lessons.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'domains': {'language': [
            {'id':'match', 'alignment_keys':['work'], 'role':'isolated_vocal_stem'},
            *[{'id':f'other{i}', 'alignment_keys':['elsewhere'], 'role':'isolated_vocal_stem'} for i in range(5)]],
            'written':[{'id':'lyrics', 'alignment_keys':['work'], 'role':'lyrics'}]}}))
        before = path.read_bytes()
        kwargs = {'selection':'aligned', 'alignment_keys':('work',)} if version == 'V2' else {}
        lessons = guidance.load_output_guidance('Fixture', 'daw', base_path=root, **kwargs)['lessons']
        record('task-aligned audio and written coverage', 'guidance_selection', [r['id'] for r in lessons][:2] == ['match','lyrics'])
        record('guidance preserves source bytes and bound', 'provenance', path.read_bytes() == before and len(lessons) <= 4)
        counts = {'reads':0, 'writes':0}
        original_load = state._load_inastate_state; original_write = state.atomic_write_json
        def load(*a, **k):
            counts['reads'] += 1; return original_load(*a, **k)
        def write(*a, **k):
            counts['writes'] += 1; return original_write(*a, **k)
        with ExitStack() as stack:
            stack.enter_context(patch.object(state, '_inastate_path', lambda child=None: root/'state.json'))
            stack.enter_context(patch.object(state, '_inastate_lock_path', lambda child=None: root/'state.lock'))
            state.update_inastate('commands', [1,2], child='Fixture')
            stack.enter_context(patch.object(state, '_load_inastate_state', load))
            stack.enter_context(patch.object(state, 'atomic_write_json', write))
            if version == 'V2':
                state.get_inastate_snapshot(child='Fixture')
                state.update_inastate_many({'a':1,'b':2,'stamp':3}, child='Fixture')
            else:
                for key in ('a','b','stamp'):
                    state.get_inastate(key, child='Fixture')
                for key, value in [('a',1),('b',2),('stamp',3)]:
                    state.update_inastate(key,value,child='Fixture')
            observed = dict(counts)
            final = state.get_inastate('commands', child='Fixture')
        measures['state_io'] = observed
        record('coherent group uses two reads and one write', 'state_resources', observed == {'reads':2,'writes':1})
        record('unrelated queue preserved', 'state_correctness', final == [1,2])
    harness_source = historical_text('codex_harness.py', BASELINE) if version == 'V1' else Path('codex_harness.py').read_text()
    # Offline CLI gate only; never execute the historical server or auth path.
    node = next(n for n in ast.parse(harness_source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    class Stopped(Exception): pass
    import argparse
    ns = {'argparse':argparse, '__doc__':'fixture', 'Path':Path, 'json':json}
    def build(*a): raise Stopped()
    ns['build_config'] = build
    exec(compile(ast.Module(body=[node],type_ignores=[]),'harness_main_fixture','exec'),ns)
    with patch('sys.argv',['codex_harness.py']), patch('sys.stderr'):
        try:
            ns['main']()
        except SystemExit as exc:
            refused = exc.code == 2
        except Stopped:
            refused = False
    record('default CLI retires before configuration or server', 'harness_retirement', refused)
    return {'correct':sum(c['correct'] for c in cases), 'total':len(cases), 'cases':cases,
            'measurements':measures, 'version':version, 'baseline_revision':BASELINE,
            'live_output_quality':'unavailable', 'security_tests':'not_run',
            'environment':{'python':sys.version, 'platform':platform.platform(), 'sqlite':sqlite3.sqlite_version, 'storage':'temporary directory; no HDD/NVMe comparison'},
            'interpretation':'selection, analyzer and resource fixtures; not a global capability score'}


if __name__ == '__main__':
    print(json.dumps({v:evaluate(v) for v in ('V1','V2')},indent=2))
