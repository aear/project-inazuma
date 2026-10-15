"""Explicit synthetic end-to-end tool availability and outcome comparison."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import tempfile

from historical_source import historical_module
from instruction_authority import LOCAL_VOLUNTARY_CODE_AUTHORITY, seal_code_command
from memory_index import ensure_memory_index_db
from personal_tool_runtime import execute_personal_tool_command

BASELINE='e13c6b4517f4df9859148a5f96cca0b4814bb9d0'


def evaluate(version='V2'):
    if version=='V1':
        catalog=historical_module('personal_tool_runtime.py',BASELINE).capability_catalog()
        available='memory_index_change' in catalog['commands']
        return {'total':1,'correct':int(available),'cases':[{'case':'bounded maintenance tools available','component':'availability','correct':available}],
                'baseline_revision':BASELINE,'status':'unavailable_historically' if not available else 'available'}
    if version!='V2':raise ValueError('version must be V1 or V2')
    cases=[]
    def record(name,passed):cases.append({'case':name,'component':'maintenance_outcome','correct':bool(passed)})
    with tempfile.TemporaryDirectory(prefix='ina_maintenance_tools_') as temp:
        root=Path(temp);source=root/'AI_Children/Fixture/memory/memory_map.json';source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'a':{'mtime_ns':1}}));before=source.read_bytes();db=source.with_suffix('.sqlite')
        ensure_memory_index_db(source,db)
        with sqlite3.connect(db) as conn:
            conn.execute('DROP INDEX idx_fragments_recent');conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        def run(action,**kwargs):
            command=seal_code_command({'action':action,'source':'ina_voluntary_choice',**kwargs},LOCAL_VOLUNTARY_CODE_AUTHORITY)
            return execute_personal_tool_command(command,child='Fixture',project_root=root,
                config={'ina_hdd_writable_path':str(root/'personal')})['value']
        trial=run('memory_index_trial');record('trial produces verified independent fixture output',trial['status']=='verified')
        plan=run('memory_index_inspect')['locations'][0]
        record('stale confirmation does not execute',run('memory_index_change',location='durable',confirmation='stale',choice='upgrade')['status']=='inspect_again')
        upgraded=run('memory_index_change',location='durable',confirmation=plan['identity'],choice='upgrade')
        record('chosen upgrade changes actual schema and preserves selected identities',upgraded['status']=='complete' and upgraded['after']['recent_index'] and upgraded['report']['selection_preserved'])
        plan=run('memory_index_inspect')['locations'][0]
        rollback=run('memory_index_change',location='durable',confirmation=plan['identity'],choice='rollback')
        record('chosen rollback removes only derived ordering index',rollback['status']=='complete' and not rollback['after']['recent_index'])
        record('source bytes retained and private receipts written',source.read_bytes()==before and Path(rollback['receipt_path']).is_file())
        record('decline does not schedule work',run('memory_index_change',choice='decline')['status']=='declined')
    return {'total':len(cases),'correct':sum(c['correct'] for c in cases),'cases':cases,'live_memory_accessed':False,'ina_choice_observed':False}


if __name__=='__main__':print(json.dumps({v:evaluate(v) for v in ('V1','V2')},indent=2))
