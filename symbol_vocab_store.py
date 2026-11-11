"""Explicit streaming SQLite projection of a symbol vocabulary; never auto-migrates.

JSON remains authoritative. Freshness is a stat change detector, not proof of
custody. Readers return None for a missing/stale/unusable projection so callers
retain their established compatibility path. No training or bootstrap writes.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile


def index_path_for(source):
    source=Path(source)
    try:child=source.parts[source.parts.index('AI_Children')+1]
    except (ValueError,IndexError):return source.with_suffix('.sqlite')
    from storage_layout import fast_runtime_path
    return fast_runtime_path(child,'symbol_to_token.sqlite',source.with_suffix('.sqlite'),subdir='index',
                             root_keys=('fast_index_root','fast_runtime_root','fast_root'))


def _stamp(path):
    st=path.stat()
    return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]


def _pairs(stream, digest, *, max_bytes, max_row):
    decoder=json.JSONDecoder();buffer='';position=0;read=0;eof=False
    def refill():
        nonlocal buffer,position,read,eof
        buffer=buffer[position:];position=0
        raw=stream.read(65536)
        if not raw:eof=True;return
        read+=len(raw);digest.update(raw)
        if read>max_bytes:raise ValueError('source byte budget')
        # A text decoder below handles UTF-8 boundaries between binary chunks.
        buffer+=utf8.decode(raw)
    import codecs
    utf8=codecs.getincrementaldecoder('utf-8')()
    def whitespace():
        nonlocal position
        while True:
            while position<len(buffer) and buffer[position] in ' \t\r\n':position+=1
            if position<len(buffer) or eof:return
            refill()
    def token(character):
        nonlocal position
        whitespace()
        if position>=len(buffer) or buffer[position]!=character:raise ValueError('invalid root mapping')
        position+=1
    def value():
        nonlocal position
        whitespace()
        while True:
            try:
                item,end=decoder.raw_decode(buffer,position)
                if len(buffer[position:end].encode('utf-8'))>max_row:raise ValueError('row budget')
                # A scalar at the buffer edge can still be an incomplete number.
                if end==len(buffer) and not eof:
                    start=position;refill();position=0;continue
                position=end;return item
            except json.JSONDecodeError:
                if eof or len(buffer)-position>max_row:raise ValueError('invalid or oversized JSON record')
                refill()
    token('{');whitespace()
    if position<len(buffer) and buffer[position]=='}':position+=1
    else:
        while True:
            key=value()
            if not isinstance(key,str) or len(key)>512:raise ValueError('bounded string symbol required')
            token(':');item=value();yield key,item
            whitespace()
            if position<len(buffer) and buffer[position]=='}':position+=1;break
            token(',')
    whitespace()
    if position<len(buffer):raise ValueError('trailing source content')
    utf8.decode(b'',final=True)


def import_symbol_vocabulary(source,destination,*,max_source_bytes=2*1024**3,
                             max_rows=500000,max_row_bytes=8*1024**2,
                             max_database_bytes=4*1024**3,reserve_bytes=1024**3):
    """Explicit one-shot projection; refuses overwrite and publishes only success."""
    source=Path(source);destination=Path(destination)
    if destination.exists() or source.resolve()==destination.resolve():raise ValueError('destination must be new')
    initial=_stamp(source)
    if initial[2]>max_source_bytes:raise ValueError('source byte budget')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free<reserve_bytes+min(initial[2]*2,max_database_bytes):raise ValueError('free-space reserve')
    fd,name=tempfile.mkstemp(prefix='.symbol-vocab-',suffix='.sqlite',dir=destination.parent);os.close(fd)
    temp=Path(name);db=None;count=0;digest=hashlib.sha256();rows_digest=hashlib.sha256()
    try:
        db=sqlite3.connect(temp)
        db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA synchronous=FULL')
        db.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE symbols(symbol TEXT PRIMARY KEY,word TEXT,ordinal INTEGER UNIQUE,value TEXT NOT NULL); CREATE INDEX words_idx ON symbols(word,ordinal);')
        with db,source.open('rb') as stream:
            for symbol,record in _pairs(stream,digest,max_bytes=max_source_bytes,max_row=max_row_bytes):
                count+=1
                if count>max_rows:raise ValueError('row budget')
                word=record.get('word') if isinstance(record,dict) else None
                word=word.lower() if isinstance(word,str) else None
                encoded=json.dumps(record,ensure_ascii=False)
                rows_digest.update(json.dumps([symbol,encoded],ensure_ascii=False).encode()+b'\n')
                db.execute('INSERT INTO symbols VALUES (?,?,?,?)',(symbol,word,count,encoded))
                if count%128==0:
                    pages=db.execute('PRAGMA page_count').fetchone()[0];size=db.execute('PRAGMA page_size').fetchone()[0]
                    if pages*size>max_database_bytes or shutil.disk_usage(destination.parent).free<reserve_bytes:raise ValueError('database/free-space budget')
            if _stamp(source)!=initial:raise ValueError('source changed during import')
            receipt={'schema':1,'source_stat':initial,'source_sha256':digest.hexdigest(),'records':count,
                     'source_path':str(source.resolve()),'rows_sha256':rows_digest.hexdigest(),'authoritative':False}
            db.execute('INSERT INTO metadata VALUES (?,?)',('manifest',json.dumps(receipt)))
        if temp.stat().st_size>max_database_bytes or shutil.disk_usage(destination.parent).free<reserve_bytes:raise ValueError('database/free-space budget')
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('database check failed')
        observed=hashlib.sha256()
        for symbol,value in db.execute('SELECT symbol,value FROM symbols ORDER BY ordinal'):
            observed.update(json.dumps([symbol,value],ensure_ascii=False).encode()+b'\n')
        if observed.hexdigest()!=rows_digest.hexdigest():raise ValueError('row readback mismatch')
        if _stamp(source)!=initial:raise ValueError('source changed during verification')
        db.close();db=None
        # Hard-link publication is atomic and refuses a concurrent destination.
        os.link(temp,destination);temp.unlink()
        return receipt
    finally:
        if db is not None:db.close()
        for suffix in ('','-journal'):
            candidate=Path(str(temp)+suffix)
            if candidate.exists():candidate.unlink()


def load_symbol_subset(source,*,symbols=(),words=(),database=None,limit=256):
    source=Path(source);path=Path(database) if database is not None else index_path_for(source)
    def selected(values):
        if isinstance(values,(str,bytes)):raise ValueError('a bounded collection is required')
        result=[]
        for i,value in enumerate(values):
            if i>=256:raise ValueError('selector budget')
            if not isinstance(value,str) or len(value)>512:raise ValueError('selector must be bounded text')
            if value not in result:result.append(value)
        return result
    symbols=selected(symbols);words=[w.lower() for w in selected(words)]
    if not symbols and not words:return None
    if not isinstance(limit,int) or not 1<=limit<=1024:raise ValueError('row limit')
    if not path.exists():return None
    db=None
    try:
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.25)
        row=db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        manifest=json.loads(row[0])
        if manifest.get('schema')!=1 or manifest.get('source_stat')!=_stamp(source) or manifest.get('source_path')!=str(source.resolve()):return None
        clauses=[];parameters=[]
        for column,values in [('symbol',symbols),('word',words)]:
            if values:
                clauses.append(column+' IN ('+','.join('?' for _ in values)+')');parameters.extend(values)
        rows=db.execute('SELECT symbol,value FROM symbols WHERE '+' OR '.join(clauses)+' ORDER BY ordinal LIMIT ?',parameters+[limit+1])
        result={};budget=0
        for symbol,value in rows:
            budget+=len(value.encode('utf-8'))
            if len(result)>=limit or budget>8*1024**2:return None
            result[symbol]=json.loads(value)
        return result
    except (OSError,sqlite3.Error,ValueError,TypeError,IndexError):return None
    finally:
        if db is not None:db.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();print(json.dumps(import_symbol_vocabulary(args.source,args.destination),indent=2))
