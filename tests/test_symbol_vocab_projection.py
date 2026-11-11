import json,hashlib,sqlite3
from pathlib import Path
import pytest
from symbol_vocab_store import import_symbol_vocabulary,load_symbol_subset
from benchmarks.benchmark_symbol_vocab_projection import assess


def test_registered_projection_comparison():
    old,new=assess('V1'),assess('V2')
    assert old['correct']<old['total']
    assert new['correct']==new['total'],new['cases']


def test_streaming_unicode_boundary_preserves_every_supplied_field(tmp_path):
    source=tmp_path/'vocab.json';dest=tmp_path/'vocab.sqlite'
    data={'first':{'word':'Lantern','payload':'🙂'*20000,'embedding':[1e-200,-3.14,0.0], 'optional':None},
          'λ':{'word':'lantern','other':{'quoted':'a " b','apostrophe':"it's"}},'unknown':42}
    source.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8');before=source.read_bytes()
    result=import_symbol_vocabulary(source,dest,reserve_bytes=0)
    assert result['source_sha256']==hashlib.sha256(before).hexdigest() and result['records']==3
    assert load_symbol_subset(source,symbols=list(data),database=dest)==data
    assert list(load_symbol_subset(source,words=['LANTERN'],database=dest))==['first','λ']
    assert source.read_bytes()==before


def test_invalid_or_oversized_import_never_publishes_partial_database(tmp_path):
    for i,text in enumerate(('[]','{"a":1,}','{\v"a":1}','{"a":1,"a":2}','{"a":1} trailing','{"a":')):
        source=tmp_path/f'{i}.json';dest=tmp_path/f'{i}.sqlite';source.write_text(text)
        with pytest.raises((ValueError,sqlite3.IntegrityError)):
            import_symbol_vocabulary(source,dest,reserve_bytes=0)
        assert not dest.exists() and source.read_text()==text
    assert not list(tmp_path.glob('.symbol-vocab-*'))


def test_budgets_and_existing_destinations_are_preserved(tmp_path):
    source=tmp_path/'vocab.json';source.write_text(json.dumps({'a':{'word':'one'},'b':{'word':'two'}}))
    for kwargs in ({'max_rows':1},{'max_source_bytes':2},{'max_row_bytes':2},{'max_database_bytes':1}):
        dest=tmp_path/'failed.sqlite'
        with pytest.raises(ValueError):import_symbol_vocabulary(source,dest,reserve_bytes=0,**kwargs)
        assert not dest.exists()
    dest=tmp_path/'existing.sqlite';dest.write_bytes(b'keep me')
    with pytest.raises(ValueError):import_symbol_vocabulary(source,dest,reserve_bytes=0)
    assert dest.read_bytes()==b'keep me'


def test_projection_stays_read_only_and_refuses_source_change(tmp_path):
    source=tmp_path/'vocab.json';dest=tmp_path/'vocab.sqlite';source.write_text('{"a":{"word":"one"}}')
    import_symbol_vocabulary(source,dest,reserve_bytes=0);before=dest.read_bytes()
    assert load_symbol_subset(source,symbols=['a'],database=dest)=={'a':{'word':'one'}}
    assert dest.read_bytes()==before
    source.write_text('{"a":{"word":"two"}}')
    assert load_symbol_subset(source,symbols=['a'],database=dest) is None


def test_empty_projection_and_selector_limits(tmp_path):
    source=tmp_path/'vocab.json';dest=tmp_path/'vocab.sqlite';source.write_text('{}')
    import_symbol_vocabulary(source,dest,reserve_bytes=0)
    assert load_symbol_subset(source,words=['absent'],database=dest)=={}
    for kwargs in ({'symbols':'bad'},{'symbols':['x']*257},{'symbols':['x'],'limit':0}):
        with pytest.raises(ValueError):load_symbol_subset(source,database=dest,**kwargs)


def test_learning_path_status_uses_only_metadata_and_never_repairs(tmp_path,monkeypatch):
    from language_path_status import inspect_language_paths
    import os
    root=tmp_path/'memory';(root/'experiences').mkdir(parents=True)
    vocab=root/'text_vocab.json';vocab.write_text('private vocabulary body')
    graph=root/'experiences/experience_graph.json';graph.write_text('private experience body')
    os.utime(graph,ns=(10,10));os.utime(vocab,ns=(20,20))
    def forbid(*a,**k):raise AssertionError('metadata check must not read payloads')
    monkeypatch.setattr(Path,'read_text',forbid)
    r=inspect_language_paths(root,tmp_path/'fast')
    assert r['graph_older_than_vocab'] and not r['private_payloads_read'] and not r['automatic_repair']
    assert not r['meaning_learning_is_verified'] and not r['migration_selected']


def test_exact_projection_does_not_remove_unknown_word_embedding_fallback(tmp_path):
    from benchmarks.benchmark_symbol_vocab_projection import generation_fixture
    path=tmp_path/'symbol_to_token.json';path.write_text(json.dumps({'17':{'word':'word17','embedding':[1.,0.]}}))
    import_symbol_vocabulary(path,path.with_suffix('.sqlite'),reserve_bytes=0)
    source=Path('language_processing.py').read_text()
    result,meta=generation_fixture(source,path)
    assert result['symbols']==['17'] and meta['full_loads']==0
    result,meta=generation_fixture(source,path,text='unfamiliar')
    assert result is None and meta['full_loads']==1 and meta['questions']


def test_existing_symbol_load_failure_cannot_trigger_bootstrap_or_save(tmp_path,monkeypatch):
    import language_processing as lp
    from unittest.mock import patch
    source=tmp_path/'symbol_to_token.json';source.write_text('{"retained":{"word":"keep"}}')
    before=source.read_bytes();calls=[]
    monkeypatch.setattr(lp,'_memory_root',lambda *a,**k:tmp_path)
    monkeypatch.setattr(lp,'_bootstrap_symbol_vocabulary',lambda *a:calls.append('bootstrap') or True)
    monkeypatch.setattr(lp,'save_symbol_to_token',lambda *a,**k:calls.append('save'))
    with patch.object(lp.json,'load',side_effect=MemoryError('synthetic large read failure')):
        with pytest.raises(OSError):lp.load_symbol_to_token('Fixture')
    assert calls==[] and source.read_bytes()==before
    source.write_text('[]')
    with pytest.raises(OSError):lp.load_symbol_to_token('Fixture')
    assert calls==[]


def test_existing_text_vocabulary_failure_defers_update_instead_of_erasing_it(tmp_path,monkeypatch):
    import ast
    from typing import Optional,Dict,Any,List
    from unittest.mock import patch
    import json
    source=tmp_path/'text_vocab.json';source.write_text('{"vocab":{"retained":{"count":5}}}')
    nodes=[n for n in ast.parse(Path('text_memory.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'load_text_vocab','update_text_vocab'}]
    saved=[]
    env={'Path':Path,'Optional':Optional,'Dict':Dict,'Any':Any,'List':List,'json':json,'_memory_root':lambda *a:tmp_path,
         '_now_iso':lambda:'fixture','tokenize_text':lambda text:['new'],'_vocab_source_category':lambda *a:'fixture',
         'MAX_SYMBOL_LINKS':6,'save_text_vocab':lambda *a,**k:saved.append(True)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<copied text vocabulary>','exec'),env)
    before=source.read_bytes()
    with patch.object(json,'load',side_effect=MemoryError('synthetic read failure')):
        with pytest.raises(OSError):env['update_text_vocab']('new',child='Fixture')
    assert saved==[] and source.read_bytes()==before


def test_failed_vocabulary_serialization_never_replaces_the_previous_store(tmp_path):
    import ast,os,tempfile,json
    from typing import Any
    from unittest.mock import patch
    for name in ('language_processing.py','text_memory.py'):
        source=Path(name).read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_atomic_write_json')
        env={'Path':Path,'Any':Any,'os':os,'tempfile':tempfile,'json':json}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied atomic writer>','exec'),env)
        target=tmp_path/'vocab.json';original=b'{"kept":{"count":42}}';target.write_bytes(original)
        def fail(payload,stream,**kwargs):stream.write('{"partial":');raise OSError('synthetic interrupted write')
        with patch.object(json,'dump',side_effect=fail):
            with pytest.raises(OSError):env['_atomic_write_json'](target,{'new':1})
        assert target.read_bytes()==original
        assert not list(tmp_path.glob('*.tmp'))
        env['_atomic_write_json'](target,{'unicode':'λ','nested':{'value':3.14}})
        assert json.loads(target.read_text())=={'unicode':'λ','nested':{'value':3.14}}


def test_grounded_adapter_known_words_can_use_only_prompt_rows(tmp_path,monkeypatch):
    from lm_studio_adapter import LMStudioAdapter
    import language_processing as lp
    from types import SimpleNamespace
    root=tmp_path/'AI_Children';memory=root/'Fixture'/'memory';memory.mkdir(parents=True)
    source=memory/'symbol_to_token.json';source.write_text(json.dumps({'a':{'word':'lamp'},'b':{'word':'other'}}))
    import_symbol_vocabulary(source,source.with_suffix('.sqlite'),reserve_bytes=0)
    result=LMStudioAdapter._load_known_words(SimpleNamespace(child='Fixture',_base_path=root),words=['lamp'])
    assert result=={'lamp':'a'}
