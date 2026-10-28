"""Explicit SQLite social-map backend; no automatic migration or path selection."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib,json,os,sqlite3,time
from pathlib import Path
VERSION=1
APPLICATION_ID=0x494E4153
MAX_SOURCE_BYTES=4*1024*1024
MAX_CONTACTS=10000
MAX_ENTRY_BYTES=65536
MAX_PAGES=65536  # 256 MiB at the fixed 4096-byte page size.


def _payload(entry):
    if not isinstance(entry,dict) or not isinstance(entry.get('user_id'),(str,int)) or isinstance(entry.get('user_id'),bool) or not str(entry['user_id']):
        raise ValueError('contact requires a nonempty string/integer user_id')
    text=json.dumps(entry,ensure_ascii=False,allow_nan=False)
    if len(text.encode())>MAX_ENTRY_BYTES:raise ValueError('contact payload budget')
    return text


def _entries(raw):
    if isinstance(raw,list):return raw
    if isinstance(raw,dict):return raw['people'] if isinstance(raw.get('people'),list) else [raw]
    raise ValueError('unsupported JSON social map shape')


@contextmanager
def connect(path,*,write=False):
    p=Path(path).absolute()
    db=sqlite3.connect(p.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,timeout=.2)
    try:
        db.execute('PRAGMA busy_timeout=200')
        if db.execute('PRAGMA application_id').fetchone()[0]!=APPLICATION_ID or db.execute('PRAGMA user_version').fetchone()[0]!=VERSION or db.execute('PRAGMA page_size').fetchone()[0]!=4096:
            raise ValueError('unrecognized social database version')
        if write:
            db.execute('PRAGMA synchronous=FULL');db.execute(f'PRAGMA max_page_count={MAX_PAGES}');db.execute('BEGIN IMMEDIATE')
        yield db
        if write:db.commit()
    except BaseException:
        if write:db.rollback()
        raise
    finally:db.close()


def import_json(source,target):
    """Create an inactive snapshot. Refuse conflicts, ambiguity and silent loss."""
    source,target=Path(source),Path(target)
    with source.open('rb') as handle:data=handle.read(MAX_SOURCE_BYTES+1)
    if len(data)>MAX_SOURCE_BYTES:raise ValueError('source size budget')
    raw=json.loads(data);entries=_entries(raw)
    if len(entries)>MAX_CONTACTS:raise ValueError('contact count budget')
    seen=set();encoded=[]
    for entry in entries:
        text=_payload(entry);key=str(entry['user_id'])
        if key in seen:raise ValueError('duplicate user_id requires review')
        seen.add(key);encoded.append((key,text))
    target.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(descriptor)
    db=None
    try:
        db=sqlite3.connect(str(target),timeout=.2)
        db.execute('PRAGMA page_size=4096');db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        db.execute(f'PRAGMA max_page_count={MAX_PAGES}')
        db.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY,value BLOB NOT NULL);
CREATE TABLE contacts(user_id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE NOT NULL,payload_json TEXT NOT NULL);
CREATE TABLE changes(sequence INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,changed_at TEXT NOT NULL,before_json TEXT,after_json TEXT NOT NULL);''')
        db.execute(f'PRAGMA application_id={APPLICATION_ID}');db.execute(f'PRAGMA user_version={VERSION}')
        with db:
            db.execute('INSERT INTO metadata VALUES(?,?)',('source_json',data))
            db.execute('INSERT INTO metadata VALUES(?,?)',('source_sha256',hashlib.sha256(data).hexdigest()))
            db.execute('INSERT INTO metadata VALUES(?,?)',('payload_bytes',str(sum(len(text.encode()) for key,text in encoded))))
            count=0
            for key,text in encoded:
                db.execute('INSERT INTO contacts VALUES(?,?,?)',(key,count,text));count+=1
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('database integrity check failed')
        db.close();db=None
        if load(target)!=entries:raise ValueError('import roundtrip mismatch')
        return {'schema_version':VERSION,'source_sha256':hashlib.sha256(data).hexdigest(),'contacts':count,'active':False,'journal_mode':'delete','unknown_fields_preserved':True}
    except BaseException:
        if db is not None:db.close()
        target.unlink(missing_ok=True)
        raise


def _decode(key, text):
    if len(text.encode()) > MAX_ENTRY_BYTES:
        raise ValueError('contact read budget')
    entry = json.loads(text)
    _payload(entry)
    if str(entry['user_id']) != str(key):
        raise ValueError('contact identity mismatch')
    return entry


def load(path):
    result=[];total=0
    with connect(path) as db:
        rows=db.execute('SELECT user_id,substr(payload_json,1,?) FROM contacts ORDER BY ordinal LIMIT ?', (MAX_ENTRY_BYTES+1,MAX_CONTACTS+1))
        for key,text in rows:
            total+=len(text.encode())
            if len(result)>=MAX_CONTACTS or total>MAX_SOURCE_BYTES:raise ValueError('contact read budget')
            result.append(_decode(key,text))
    return result


def find(path,user_id):
    with connect(path) as db:
        row=db.execute('SELECT substr(payload_json,1,?) FROM contacts WHERE user_id=?',(MAX_ENTRY_BYTES+1,str(user_id))).fetchone()
    return _decode(user_id,row[0]) if row else None


def mutate(path,user_id,edit):
    """Serialize read-modify-write for one contact, retaining unknown fields."""
    key=str(user_id)
    with connect(path,write=True) as db:
        old=db.execute('SELECT ordinal,substr(payload_json,1,?) FROM contacts WHERE user_id=?',(MAX_ENTRY_BYTES+1,key)).fetchone()
        entries=[_decode(key,old[1])] if old else []
        if not edit(entries):return False
        if len(entries)!=1 or str(entries[0].get('user_id'))!=key:raise ValueError('mutation changed contact identity')
        payload=_payload(entries[0])
        stored_size=db.execute("SELECT value FROM metadata WHERE key='payload_bytes'").fetchone()
        current_size=int(stored_size[0]) if stored_size else db.execute('SELECT COALESCE(SUM(length(CAST(payload_json AS BLOB))),0) FROM contacts').fetchone()[0]
        next_size=current_size-(len(old[1].encode()) if old else 0)+len(payload.encode())
        if next_size>MAX_SOURCE_BYTES:raise ValueError('aggregate contact payload budget')
        db.execute("INSERT INTO metadata VALUES('payload_bytes',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(next_size),))
        if old:ordinal=old[0]
        else:
            count=db.execute('SELECT COUNT(*) FROM contacts').fetchone()[0]
            if count>=MAX_CONTACTS:raise ValueError('contact write budget')
            ordinal=db.execute('SELECT COALESCE(MAX(ordinal),-1)+1 FROM contacts').fetchone()[0]
        db.execute('INSERT INTO contacts VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET payload_json=excluded.payload_json',(key,ordinal,payload))
        db.execute('INSERT INTO changes(user_id,changed_at,before_json,after_json) VALUES(?,?,?,?)',(key,datetime.now(timezone.utc).isoformat(),old[1] if old else None,payload))
    return True


def export_json(path,target):
    """Explicit consistent JSON export; no automatic compatibility dual writer."""
    with connect(path) as db:
        raw=json.loads(db.execute("SELECT value FROM metadata WHERE key='source_json'").fetchone()[0])
        rows=db.execute('SELECT user_id,substr(payload_json,1,?) FROM contacts ORDER BY ordinal LIMIT ?',(MAX_ENTRY_BYTES+1,MAX_CONTACTS+1))
        entries=[];total=0
        for key,text in rows:
            total+=len(text.encode())
            if len(entries)>=MAX_CONTACTS or total>MAX_SOURCE_BYTES:raise ValueError('export byte budget')
            entries.append(_decode(key,text))
    if isinstance(raw,dict) and isinstance(raw.get('people'),list):raw['people']=entries;output=raw
    elif isinstance(raw,dict) and len(entries)==1:output=entries[0]
    else:output=entries
    # Destination conflicts require deliberate review, never silent overwrite.
    with Path(target).open('x',encoding='utf-8') as handle:
        json.dump(output,handle,ensure_ascii=False,indent=2);handle.write('\n')
    return {'contacts':len(entries),'destination':str(target)}
