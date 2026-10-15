import json
from pathlib import Path
import sqlite3

import pytest
from instruction_authority import LOCAL_VOLUNTARY_CODE_AUTHORITY, InstructionAuthorityError, seal_code_command
from personal_tool_runtime import execute_personal_tool_command, capability_catalog
from memory_index import ensure_memory_index_db


def _fixture(tmp_path):
    root=tmp_path/'project'; personal=tmp_path/'personal'
    source=root/'AI_Children/Fixture/memory/memory_map.json'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'a':{'mtime_ns':1,'tags':['one']},'b':{'mtime_ns':2}}))
    db=source.with_suffix('.sqlite')
    ensure_memory_index_db(source,db)
    with sqlite3.connect(db) as conn:
        conn.execute('DROP INDEX idx_fragments_recent')
        conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
    return root,personal,source,db


def _command(root,personal,action,**kwargs):
    command=seal_code_command({'action':action,'source':'ina_voluntary_choice',**kwargs},LOCAL_VOLUNTARY_CODE_AUTHORITY)
    return execute_personal_tool_command(command,child='Fixture',project_root=root,
                config={'ina_hdd_writable_path':str(personal)})['value']


def test_catalog_and_inspection_are_passive_and_read_only(tmp_path):
    root,personal,source,db=_fixture(tmp_path)
    before=db.read_bytes()
    assert {'memory_index_inspect','memory_index_trial','memory_index_change'} <= set(capability_catalog()['commands'])
    inspection=_command(root,personal,'memory_index_inspect')
    assert inspection['automatic_trigger'] is False
    assert inspection['locations'][0]['schema_version']=='2'
    assert not inspection['locations'][0]['payloads_read']
    assert db.read_bytes()==before


def test_tool_completes_chosen_trial_upgrade_and_rollback_without_moving_memory(tmp_path):
    root,personal,source,db=_fixture(tmp_path)
    original_source=source.read_bytes()
    trial=_command(root,personal,'memory_index_trial')
    assert trial['status']=='verified' and not trial['live_memory_accessed']
    before=_command(root,personal,'memory_index_inspect')['locations'][0]
    upgraded=_command(root,personal,'memory_index_change',location='durable',confirmation=before['identity'],choice='upgrade')
    assert upgraded['status']=='complete' and upgraded['after']['recent_index']
    assert upgraded['report']['selection_preserved']
    after=_command(root,personal,'memory_index_inspect')['locations'][0]
    rollback=_command(root,personal,'memory_index_change',location='durable',confirmation=after['identity'],choice='rollback')
    assert rollback['status']=='complete' and not rollback['after']['recent_index']
    assert rollback['after']['schema_version']=='2'
    assert source.read_bytes()==original_source
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT frag_id FROM fragments ORDER BY frag_id').fetchall()==[('a',),('b',)]
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='idx_fragment_tags_tag'").fetchone()
    receipts=[json.loads(line) for line in Path(rollback['receipt_path']).read_text().splitlines()]
    assert [r['status'] for r in receipts]==['verified','requested','complete','requested','complete']


def test_tools_require_real_choice_authority_and_refuse_stale_or_unknown_targets(tmp_path):
    root,personal,source,db=_fixture(tmp_path)
    with pytest.raises(InstructionAuthorityError):
        execute_personal_tool_command({'action':'memory_index_change','source':'ina_voluntary_choice','choice':'upgrade'},
            child='Fixture',project_root=root,config={'ina_hdd_writable_path':str(personal)})
    with pytest.raises(InstructionAuthorityError):
        seal_code_command({'action':'memory_index_change','source':'discord','choice':'upgrade'},LOCAL_VOLUNTARY_CODE_AUTHORITY)
    assert _command(root,personal,'memory_index_change',choice='upgrade',location='/etc/private',confirmation='x')['status']=='unavailable'
    assert _command(root,personal,'memory_index_change',choice='upgrade',location='durable',confirmation='stale')['status']=='inspect_again'
    assert _command(root,personal,'memory_index_change',choice='defer')['status']=='deferred'
    assert _command(root,personal,'memory_index_change',choice='decline')['status']=='declined'
    assert not (personal/'Notes/memory_maintenance_receipts.jsonl').exists()


def test_chosen_device_trial_never_changes_its_real_index_or_leaves_scratch(tmp_path):
    root,personal,source,db=_fixture(tmp_path)
    original=db.read_bytes()
    result=_command(root,personal,'memory_index_trial',location='durable')
    assert result['status']=='verified' and result['storage']=='durable'
    assert db.read_bytes()==original
    assert not list(db.parent.glob('ina_index_trial_*'))
    assert _command(root,personal,'memory_index_trial',location='/arbitrary')['status']=='unavailable'
