"""Ina-chosen, bounded index maintenance. No memory payloads are moved or read."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Mapping
import uuid

from io_utils import file_lock
from memory_index import _ensure_schema, upgrade_recent_index_sidecar
from storage_layout import format_child_path


def _locations(child: str, root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    if Path(child).name != child or child in {'.','..'}:
        raise ValueError('child must be a single path component')
    result={'durable':root/'AI_Children'/child/'memory'/'memory_map.sqlite'}
    layout=config.get('storage_layout') or {}
    for key in ('fast_index_root','fast_runtime_root','fast_root'):
        path=format_child_path(layout.get(key),child)
        if path is None:continue
        if not path.is_absolute():path=root/path
        if key=='fast_root':path=path/'AI_Children'/child/'memory'/'fast_runtime'/'index'
        elif key=='fast_runtime_root':path=path/'index'
        result[key]=path/'memory_map.sqlite'
    # No caller-supplied file path can become a maintenance target.
    return result


def inspect_indexes(child: str, root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    locations=[]
    for name,path in _locations(child,root,config).items():
        row={'location':name,'path':str(path),'exists':path.is_file(),'payloads_read':False}
        if row['exists']:
            stat=path.stat()
            row['bytes']=stat.st_size
            identity_parts=[str(path.resolve()),str(stat.st_ino),str(stat.st_size),str(stat.st_mtime_ns)]
            for suffix in ('-wal','-journal'):
                sidecar=Path(str(path)+suffix)
                if sidecar.exists():
                    side_stat=sidecar.stat();identity_parts.extend([suffix,str(side_stat.st_size),str(side_stat.st_mtime_ns)])
            row['identity']=hashlib.sha256(':'.join(identity_parts).encode()).hexdigest()
            try:
                with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.1) as conn:
                    version=conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                    row['schema_version']=version[0] if version else None
                    row['recent_index']=bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_fragments_recent'").fetchone())
            except sqlite3.Error:
                row['status']='unavailable_or_busy'
        locations.append(row)
    return {'schema':'ina.index_inspection/V1','locations':locations,'private':True,
            'moves_memory':False,'automatic_trigger':False}


def _receipt(personal: Path, value: dict[str, Any]) -> dict[str, Any]:
    path=personal/'Notes'/'memory_maintenance_receipts.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    value={**value,'recorded_at':datetime.now(timezone.utc).isoformat()}
    with file_lock(path.with_suffix('.lock')):
        with path.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')
    return {**value,'receipt_path':str(path)}


def trial_index(personal: Path, *, fixture_root: Path | None = None, storage: str = "temporary") -> dict[str, Any]:
    """One fixed synthetic trial; retained receipt identifies its limited scope."""
    timings={}
    with tempfile.TemporaryDirectory(prefix='ina_index_trial_', dir=fixture_root) as temp:
        db=Path(temp)/'fixture.sqlite'
        with sqlite3.connect(db) as conn:
            _ensure_schema(conn)
            conn.execute('DROP INDEX idx_fragments_recent')
            conn.execute("INSERT INTO meta VALUES ('schema_version','2')")
            conn.executemany('INSERT INTO fragments(frag_id,mtime_ns) VALUES (?,?)',((f'f{i:05}',i//2) for i in range(4096)))
            conn.commit()
            query='SELECT frag_id FROM fragments ORDER BY mtime_ns DESC,frag_id DESC LIMIT 64'
            started=time.monotonic();before=conn.execute(query).fetchall();timings['before_seconds']=time.monotonic()-started
        upgraded=upgrade_recent_index_sidecar(db,max_seconds=5)
        with sqlite3.connect(db) as conn:
            started=time.monotonic();after=conn.execute(query).fetchall();timings['after_seconds']=time.monotonic()-started
    return _receipt(personal,{'schema':'ina.index_trial/V1','trial_id':uuid.uuid4().hex,
        'status':'verified' if upgraded['upgraded'] and before==after else 'failed',
        'selection_preserved':before==after,'fixture':'4096 synthetic rows; temporary files removed after trial','storage':storage,
        'live_memory_accessed':False,'moves_memory':False,'timings':timings,
        'does_not_establish':'live performance, available space, readiness or authority to migrate'})


def trial_configured_index(child: str, root: Path, config: Mapping[str, Any], personal: Path,
                           *, location: str = "temporary") -> dict[str, Any]:
    if location == "temporary":
        return trial_index(personal)
    path = _locations(child,root,config).get(location)
    if path is None or not path.parent.is_dir():
        return {'status':'unavailable','live_memory_accessed':False}
    if shutil.disk_usage(path.parent).free < 64*1024*1024:
        return {'status':'deferred','reason':'free_space_reserve','live_memory_accessed':False}
    # Only disposable data on the chosen configured device; never open its index.
    return trial_index(personal,fixture_root=path.parent,storage=location)


def change_index(child: str, root: Path, config: Mapping[str, Any], personal: Path,
                 *, location: str, confirmation: str, choice: str, max_seconds: float = 5) -> dict[str, Any]:
    """One chosen transactional index upgrade or rollback, with stale-plan refusal."""
    if choice in {'defer','decline'}:
        return {'status':'deferred' if choice=='defer' else 'declined','modified':False}
    if choice not in {'upgrade','rollback'}:
        raise ValueError('choice must be upgrade, rollback, defer or decline')
    max_seconds=float(max_seconds)
    if not .1 <= max_seconds <= 10:raise ValueError('max_seconds must be 0.1..10')
    inspection=inspect_indexes(child,root,config)
    row=next((r for r in inspection['locations'] if r['location']==location),None)
    if row is None or not row.get('exists') or row.get('status'):
        return {'status':'unavailable','modified':False}
    if not confirmation or confirmation!=row.get('identity'):
        return {'status':'inspect_again','reason':'missing_or_stale_confirmation','modified':False}
    path=_locations(child,root,config)[location]
    if path.is_symlink():return {'status':'deferred','reason':'symlink_target_requires_review','modified':False}
    if shutil.disk_usage(path.parent).free < row['bytes']*2+64*1024*1024:
        return {'status':'deferred','reason':'free_space_reserve','modified':False}
    if row.get('schema_version') not in {'2','3'}:
        return {'status':'deferred','reason':'unsupported_schema','modified':False}
    _receipt(personal,{'schema':'ina.index_maintenance/V1','status':'requested',
        'choice':choice,'location':location,'path':str(path),'before':row,'moves_memory':False})
    if choice=='upgrade':
        report=upgrade_recent_index_sidecar(path,max_seconds=max_seconds)
        ok=report['upgraded']
    else:
        conn=None; started=time.monotonic();ok=False
        try:
            conn=sqlite3.connect(path.resolve().as_uri()+'?mode=rw',uri=True,timeout=.25)
            conn.set_progress_handler(lambda:int(time.monotonic()-started>=max_seconds),1000)
            conn.execute('BEGIN IMMEDIATE')
            # Drop only this derived ordering index; never touch fragments/tags.
            conn.execute('DROP INDEX IF EXISTS idx_fragments_recent')
            conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
            conn.commit();ok=True;report={'reason':'ordering_index_removed','elapsed_seconds':time.monotonic()-started}
        except sqlite3.Error:
            if conn is not None:
                conn.set_progress_handler(None,0);conn.rollback()
            report={'reason':'deferred_or_busy'}
        finally:
            if conn is not None:conn.close()
    return _receipt(personal,{'schema':'ina.index_maintenance/V1','status':'complete' if ok else 'deferred',
        'choice':choice,'location':location,'path':str(path),'moves_memory':False,
        'fragment_payloads_changed':False,'report':report,'before':row,
        'after':next(r for r in inspect_indexes(child,root,config)['locations'] if r['location']==location)})
