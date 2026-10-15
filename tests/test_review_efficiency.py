"""Disposable-state regression checks for the 2026-10-15 review changes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from language_intelligence import analyze_utterance, morphology
from learned_media_lessons import load_output_guidance
from memory_index import ensure_memory_index_db, indexed_fragment_rows, index_is_current
import runtime_state as rs


def test_snapshot_is_detached_and_batch_preserves_other_writers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rs.update_inastate_many({'mood': {'joy': 1}, 'queue': [1]}, child='Fixture')
    snapshot = rs.get_inastate_snapshot(child='Fixture')
    snapshot['mood']['joy'] = 0
    assert rs.get_inastate('mood', child='Fixture') == {'joy': 1}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: rs.update_inastate_many({f'key{n}': n}, child='Fixture'), range(12)))
    rs.update_inastate_many({'mood': {'joy': 2}, 'stamp': 3}, child='Fixture')
    current = rs.get_inastate_snapshot(child='Fixture')
    assert current['queue'] == [1]
    assert current['stamp'] == 3
    assert all(current[f'key{n}'] == n for n in range(12))


def test_v2_sidecar_upgrade_keeps_touches_without_reading_source(tmp_path, monkeypatch):
    import memory_index as mi
    source = tmp_path / 'map.json'
    db = tmp_path / 'map.sqlite'
    source.write_text(json.dumps({'a': {'mtime_ns': 9}, 'b': {'mtime_ns': 9}, 'c': {'mtime_ns': 3}}))
    assert ensure_memory_index_db(source, db)
    with sqlite3.connect(db) as conn:
        conn.execute('DROP INDEX idx_fragments_recent')
        conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        conn.execute("UPDATE fragments SET last_seen='retained-touch' WHERE frag_id='a'")
    def no_reparse(*args, **kwargs):
        raise AssertionError('migration reparsed source map')
    monkeypatch.setattr(mi, 'iter_json_object_items', no_reparse)
    assert index_is_current(source, db)
    assert ensure_memory_index_db(source, db)
    with sqlite3.connect(db) as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='idx_fragments_recent'").fetchone()
    assert ensure_memory_index_db(source, db, upgrade_recent_index=True)
    assert index_is_current(source, db)
    assert [r['frag_id'] for r in indexed_fragment_rows(db, limit=2)] == ['b', 'a']
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT last_seen FROM fragments WHERE frag_id='a'").fetchone()[0] == 'retained-touch'
        plan = conn.execute('EXPLAIN QUERY PLAN SELECT frag_id FROM fragments ORDER BY mtime_ns DESC, frag_id DESC LIMIT 2').fetchall()
    assert any('idx_fragments_recent' in str(row) for row in plan)
    assert not any('TEMP B-TREE' in str(row) for row in plan)


def test_language_preserves_quantities_and_alternating_transfer_roles():
    for text in ('I run.', 'She ran.', 'She is running.', 'I do not run.'):
        clause = analyze_utterance(text)['clauses'][0]
        assert clause['predicate'] == 'run'
        assert clause['subject'] in {'i', 'she'}
    assert analyze_utterance('I do not run.')['clauses'][0]['negated']
    a = analyze_utterance('I give her a book.')['clauses'][0]
    b = analyze_utterance('I give a book to her.')['clauses'][0]
    assert a['arguments'] == b['arguments'] == {'recipient': 'her', 'theme': 'a book'}
    tokens = morphology('I opened 3 files and 2.5 boxes: λ⊙.')
    assert {'3', '2.5', 'λ', '⊙', ':'} <= {t['normalized'] for t in tokens}
    assert analyze_utterance('I frobnicate widgets.')['uncertainty']['morphology']['confidence'] < .9
    assert analyze_utterance('')['uncertainty']['morphology']['confidence'] < .9


def test_guidance_candidate_prevents_domain_starvation_and_keeps_default(tmp_path):
    path = tmp_path / 'Fixture/memory/learned_media_lessons.json'
    path.parent.mkdir(parents=True)
    lessons = {'domains': {'language': [
        {'id': 'relevant', 'role': 'isolated_vocal_stem', 'alignment_keys': ['work']},
        *[{'id': f'other{n}', 'role': 'isolated_vocal_stem', 'alignment_keys': ['other']} for n in range(5)]
    ], 'written': [{'id': 'lyric', 'role': 'lyrics', 'alignment_keys': ['work']}]}}
    path.write_text(json.dumps(lessons))
    before = path.read_bytes()
    legacy = load_output_guidance('Fixture', 'daw', base_path=tmp_path)
    assert [r['id'] for r in legacy['lessons']] == ['other4', 'other3', 'other2', 'other1']
    aligned = load_output_guidance('Fixture', 'daw', base_path=tmp_path, selection='aligned', alignment_keys=('work',))
    assert [r['id'] for r in aligned['lessons']][:2] == ['relevant', 'lyric']
    assert len(aligned['lessons']) == 4
    assert path.read_bytes() == before


def test_harness_deprecation_is_offline_and_precedes_root_or_authentication(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'codex_harness.py'
    status = subprocess.run([sys.executable, str(script), '--deprecation-status', '--root', str(tmp_path/'absent')], capture_output=True, text=True, timeout=10)
    assert status.returncode == 0
    assert json.loads(status.stdout)['status'] == 'deprecated'
    default = subprocess.run([sys.executable, str(script), '--root', str(tmp_path/'absent')], capture_output=True, text=True, timeout=10)
    assert default.returncode == 2
    assert 'deprecated' in default.stderr
    assert 'Workspace is not a directory' not in default.stderr


def test_semantic_current_evaluator_observes_results_and_records_identity(monkeypatch):
    from module_benchmarks import benchmark_module
    import language_processing as lp
    result, = benchmark_module('semantic_topology', ('V3',))
    assert result.total == 6 and result.correct == 6
    assert set(result.component_scores) == {'word_sense'}
    assert result.implementation_identity['status'] == 'recorded'
    assert len(result.implementation_identity['changes_sha256']) == 64
    original = lp.resolve_text_vocab_meanings
    def wrong(*args, **kwargs):
        return [{**row, 'symbol':'wrong'} for row in original(*args, **kwargs)]
    monkeypatch.setattr(lp, 'resolve_text_vocab_meanings', wrong)
    broken, = benchmark_module('semantic_topology', ('V3',))
    assert broken.correct == 0


def test_working_tree_identity_changes_with_untracked_and_tracked_source(tmp_path, monkeypatch):
    import historical_source as hs
    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True, timeout=10)
    git('init')
    source = tmp_path/'example.py'
    source.write_text('VALUE = 1\n')
    git('add','example.py')
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture')
    monkeypatch.setattr(hs, '_REPOSITORY', tmp_path)
    clean = hs.working_tree_identity()
    source.write_text('VALUE = 2\n')
    dirty = hs.working_tree_identity()
    (tmp_path/'candidate.py').write_text('OTHER = 3\n')
    extra = hs.working_tree_identity()
    assert clean['head'] == dirty['head'] == extra['head']
    assert len({r['changes_sha256'] for r in (clean,dirty,extra)}) == 3


def test_hunger_caller_matches_historical_values_with_grouped_io():
    from benchmarks.benchmark_review_efficiency import monitor_fixture
    old, old_counts = monitor_fixture('V1')
    new, new_counts = monitor_fixture('V2')
    assert old == new
    assert old_counts == {'reads':6, 'writes':3}
    assert new_counts == {'reads':1, 'writes':1}


def test_auxiliary_selection_preserves_copulas_and_possession():
    assert analyze_utterance('I am happy to run.')['clauses'][0]['predicate'] == 'be'
    assert analyze_utterance('I have a book to give.')['clauses'][0]['predicate'] == 'have'
    assert analyze_utterance('I have not opened it.')['clauses'][0]['predicate'] == 'open'


def test_index_upgrade_timeout_and_unsupported_schema_preserve_database(tmp_path, monkeypatch):
    import memory_index as mi
    source=tmp_path/'map.json'; db=tmp_path/'map.sqlite'
    source.write_text(json.dumps({f'f{i}': {'mtime_ns':i} for i in range(2000)}))
    assert mi.ensure_memory_index_db(source,db)
    with sqlite3.connect(db) as conn:
        conn.execute('DROP INDEX idx_fragments_recent')
        conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
    stamps=iter([0.0]+[5.0]*100)
    monkeypatch.setattr(mi.time,'monotonic',lambda:next(stamps))
    report=mi.upgrade_recent_index_sidecar(db,max_seconds=0.1)
    assert not report['upgraded']
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()==('2',)
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='idx_fragments_recent'").fetchone()
        conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    assert mi.upgrade_recent_index_sidecar(db)['reason']=='unsupported_schema'
