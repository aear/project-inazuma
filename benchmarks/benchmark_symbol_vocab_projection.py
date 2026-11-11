"""Indexed projection vs the pinned complete-JSON symbol reader."""
from pathlib import Path
import ast,json,tempfile
BASELINE='71b80f8f011373ff759afec79931094d31355df0'


def generation_fixture(source,path,text='word17'):
    """Actual symbolic-generation body with anonymous learning/audio collaborators."""
    from types import SimpleNamespace as N
    import language_processing as lp
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='generate_symbolic_reply_from_text')
    captured={'full_loads':0,'questions':[]}
    def full(*a,**k):
        captured['full_loads']+=1
        return json.loads(path.read_text())
    def dual(symbols,**kwargs):
        captured['vocabulary']=kwargs.get('symbol_to_token_vocab')
        return {'text':'λ','native_text':'λ','gloss_text':'word17'}
    def insights(symbols,vocab,unknown):
        captured['insight_words']=[(vocab.get(sym) or {}).get('word') for sym in symbols]
        return None
    env={**vars(lp),'_memory_root':lambda *a,**k:path.parent,'load_symbol_to_token':full,
         '_ensure_vocab_embeddings':lambda *a,**k:False,'save_symbol_to_token':lambda *a,**k:None,
         'load_output_guidance':lambda *a,**k:{},'_load_text_vocab_links_scoped':lambda *a,**k:{},
         '_shadow_audit_text_vocab_mappings':lambda *a,**k:{},'guess_language_code':lambda *a:'en',
         '_EMBEDDER':N(embed_text=lambda *a,**k:[0.,1.],cosine=lambda *a:0.),
         'adaptive_symbol_limit':lambda *a,**k:k['available_symbols'],'speak_symbolically':lambda *a,**k:None,
         'build_dual_symbolic_message':dual,'_build_reply_transformer_insights':insights,
         'seed_self_question':lambda value:captured['questions'].append(value)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied symbolic generation>','exec'),env)
    result=env['generate_symbolic_reply_from_text'](text,child='Fixture',base_path=path.parent,playback=False)
    return result,captured


def assess(version):
    from symbol_vocab_store import import_symbol_vocabulary,load_symbol_subset
    if version=='V1':
        from historical_source import historical_text
        source=historical_text('language_processing.py',BASELINE)
    else:source=Path('language_processing.py').read_text()
    rows=[]
    def add(name,component,ok):rows.append({'case':name,'component':component,'correct':bool(ok)})
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'symbol_to_token.json';database=path.with_suffix('.sqlite')
        payload={str(i):{'word':'word'+str(i),'embedding':[i/10,1.], 'native':{'surface':'∘⊙'},'sources':['fixture']} for i in range(128)}
        payload['17']['unicode']='λ🙂'*15000
        path.write_text(json.dumps(payload));receipt=import_symbol_vocabulary(path,database,reserve_bytes=0)
        names={'load_symbol_to_token','_load_symbol_subset_scoped'}
        functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
        from typing import Optional,Dict,Any
        env={'Optional':Optional,'Dict':Dict,'Any':Any,'Path':Path,'json':json,
             '_memory_root':lambda *a,**k:path.parent,'_bootstrap_symbol_vocabulary':lambda *a:False}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<copied symbol reader>','exec'),env)
        if '_load_symbol_subset_scoped' in env: result=env['_load_symbol_subset_scoped']('Fixture',symbols=['17'])
        else:result=env['load_symbol_to_token']('Fixture')
        add('streamed records retain provenance and unknown fields','conversion',version=='V2' and receipt['records']==128 and len(receipt['source_sha256'])==64)
        generated,captured=generation_fixture(source,path)
        add('same exact-word symbol selected','generation',generated['symbols']==['17'])
        add('fresh projection avoids complete vocabulary in generation','bounded_lookup',captured['full_loads']==0)
        add('realiser and insight metadata preserved','generation',captured['insight_words']==['word17'] and captured['vocabulary']['17']==payload['17'])
        add('requested symbol value preserved','parity',result['17']==payload['17'])
        add('scoped lookup excludes unrelated payloads','bounded_lookup',set(result)=={'17'})
        if version=='V2':
            result=load_symbol_subset(path,words=['WORD17']);add('indexed case-normalised word lookup','bounded_lookup',result=={'17':payload['17']})
            add('missing word is empty fresh result','bounds',load_symbol_subset(path,words=['absent'])=={})
            add('row budget refuses truncated result','bounds',load_symbol_subset(path,words=['word17','word18'],limit=1) is None)
            payload['17']['word']='changed';path.write_text(json.dumps(payload))
            add('stale projection refused','freshness',load_symbol_subset(path,symbols=['17']) is None)
            result=env['_load_symbol_subset_scoped']('Fixture',symbols=['17'])
            add('legacy writer updates remain visible','compatibility',result['17']==payload['17'])
        else:
            for name,component in [('indexed case-normalised word lookup','bounded_lookup'),('missing word is empty fresh result','bounds'),('row budget refuses truncated result','bounds'),('stale projection refused','freshness'),('legacy writer updates remain visible','compatibility')]:
                add(name,component,False)
    # Copied grounded-adapter method: exact same word result, bounded inventory.
    from types import SimpleNamespace as N
    import language_processing as lp
    if version=='V1':
        from historical_source import historical_text
        adapter_source=historical_text('lm_studio_adapter.py',BASELINE)
    else:adapter_source=Path('lm_studio_adapter.py').read_text()
    node=next(n for n in ast.walk(ast.parse(adapter_source)) if isinstance(n,ast.FunctionDef) and n.name=='_load_known_words')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'AI_Children';memory=root/'Fixture'/'memory';memory.mkdir(parents=True)
        path=memory/'symbol_to_token.json';path.write_text(json.dumps({'a':{'word':'lamp'},'b':{'word':'other'}}))
        import_symbol_vocabulary(path,path.with_suffix('.sqlite'),reserve_bytes=0)
        from typing import Dict,Iterable,Optional
        env={'Dict':Dict,'Iterable':Iterable,'Optional':Optional,'load_symbol_to_token':lambda *a,**k:json.loads(path.read_text()),
             '_load_symbol_subset_scoped':lp._load_symbol_subset_scoped,'load_text_vocab_word_symbol_index':lp.load_text_vocab_word_symbol_index}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied grounded word inventory>','exec'),env)
        if version=='V2':known=env['_load_known_words'](N(child='Fixture',_base_path=root),words=['lamp'])
        else:known=env['_load_known_words'](N(child='Fixture',_base_path=root))
        add('grounded adapter retains exact word mapping','parity',known.get('lamp')=='a')
        add('grounded adapter requests only prompt vocabulary','bounded_lookup',known=={'lamp':'a'})
    # Exercise the real read-failure branches without touching learned data.
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'symbol_to_token.json';path.write_text('{"kept":{"word":"keep"}}')
        ns={'Path':Path,'json':json,'Optional':__import__('typing').Optional,'Dict':__import__('typing').Dict,'Any':object,
            '_memory_root':lambda *a,**k:path.parent,'_bootstrap_symbol_vocabulary':lambda *a:True}
        writes=[];ns['save_symbol_to_token']=lambda *a,**k:writes.append(True)
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='load_symbol_to_token')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied reader>','exec'),ns)
        with patch.object(json,'load',side_effect=MemoryError('fixture')):
            try:ns['load_symbol_to_token']('Fixture')
            except OSError:pass
        add('failed symbol read never bootstraps over retained vocabulary','preservation',not writes)
        if version=='V1':
            from historical_source import historical_text
            text_source=historical_text('text_memory.py',BASELINE)
        else:text_source=Path('text_memory.py').read_text()
        node=next(n for n in ast.parse(text_source).body if isinstance(n,ast.FunctionDef) and n.name=='load_text_vocab')
        ns['_now_iso']=lambda:'fixture';(path.parent/'text_vocab.json').write_text('{"vocab":{}}')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied frequency reader>','exec'),ns)
        refused=False
        with patch.object(json,'load',side_effect=MemoryError('fixture')):
            try:ns['load_text_vocab']('Fixture')
            except OSError:refused=True
        add('failed frequency read is not an empty learning state','preservation',refused)
    import os
    from typing import Any
    for file_name in ('language_processing.py','text_memory.py'):
        if file_name=='language_processing.py':writer_source=source
        elif version=='V1':writer_source=text_source
        else:writer_source=Path(file_name).read_text()
        node=next(n for n in ast.parse(writer_source).body if isinstance(n,ast.FunctionDef) and n.name=='_atomic_write_json')
        ns={'Path':Path,'Any':Any,'json':json,'os':os,'tempfile':tempfile}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<copied atomic vocabulary writer>','exec'),ns)
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'vocab.json';before=b'{"kept":1}';target.write_bytes(before)
            def fail(payload,stream,**kwargs):stream.write('{"partial":');raise OSError('fixture')
            with patch.object(json,'dump',side_effect=fail):
                try:ns['_atomic_write_json'](target,{'new':1})
                except OSError:pass
            add('partial write preserves '+file_name,'preservation',target.read_bytes()==before)
    if version=='V2':
        from language_path_status import inspect_language_paths
        result=inspect_language_paths(Path('/nonexistent-fixture-root'))
        add('read-only learning-path diagnostics disclose unavailable evidence','diagnostics',result['graph_older_than_vocab'] is None and not result['meaning_learning_is_verified'] and not result['automatic_repair'])
        add('metadata-only tool is discoverable','diagnostics','"language_path_status"' in Path('personal_tool_runtime.py').read_text())
    else:
        add('read-only learning-path diagnostics disclose unavailable evidence','diagnostics',False)
        add('metadata-only tool is discoverable','diagnostics',False)
    return {'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'synthetic complete-JSON reader versus fresh standalone SQLite projection; no live conversion or storage placement'}
