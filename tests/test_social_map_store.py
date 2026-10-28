import json,sqlite3,threading
from pathlib import Path
import pytest
import social_map as api
import social_map_store as store
from benchmarks.benchmark_social_storage import fixture_entries,assess


def setup_store(tmp_path,raw=None):
 source=tmp_path/'social.json';source.write_text(json.dumps(raw if raw is not None else fixture_entries(),ensure_ascii=False))
 database=tmp_path/'social.sqlite';report=store.import_json(source,database)
 cfg={'discord':{'social_map_path':str(source),'social_map_backend':'sqlite','social_map_database_path':str(database)}}
 return source,database,cfg,report


def test_versioned_preservation_and_authority_comparison():
 before,after=assess('V1'),assess('V2')
 assert before['correct']<before['total']
 assert after['correct']==after['total'],after


def test_import_roundtrip_export_and_source_bytes_retained(tmp_path):
 raw={'people':fixture_entries(),'uninterpreted_root_field':{'retain':'value'}}
 source,db,cfg,report=setup_store(tmp_path,raw);original=source.read_bytes()
 assert store.load(db)==raw['people'] and report['contacts']==3 and not report['active']
 with store.connect(db) as connection:
  assert connection.execute("SELECT value FROM metadata WHERE key='source_json'").fetchone()[0]==original
 assert db.stat().st_mode & 0o077==0
 assert api.record_dm_attempt(17,'New label',config=cfg)
 assert api.is_high_trust(17,config=cfg)
 out=tmp_path/'export.json';store.export_json(db,out)
 assert json.loads(out.read_text())['uninterpreted_root_field']==raw['uninterpreted_root_field']
 assert source.read_bytes()==original
 with pytest.raises(FileExistsError):store.export_json(db,out)


def test_duplicate_ids_and_import_conflicts_refuse_without_source_mutation(tmp_path):
 source=tmp_path/'source.json';source.write_text(json.dumps([{'user_id':1},{'user_id':'1'}]));before=source.read_bytes()
 with pytest.raises(ValueError):store.import_json(source,tmp_path/'dup.sqlite')
 assert source.read_bytes()==before and not (tmp_path/'dup.sqlite').exists()
 source.write_text('[]');target=tmp_path/'existing.sqlite';target.write_bytes(b'preserve')
 with pytest.raises(FileExistsError):store.import_json(source,target)
 assert target.read_bytes()==b'preserve'


def test_serial_transactions_preserve_concurrent_contact_edits(tmp_path):
 source,db,cfg,report=setup_store(tmp_path);results=[]
 def update(number):results.append(api.update_social_entry(17,tags=[f'worker_{number}'],config=cfg))
 workers=[threading.Thread(target=update,args=(n,)) for n in range(4)]
 for thread in workers:thread.start()
 for thread in workers:thread.join(timeout=2);assert not thread.is_alive()
 assert results==[True]*4
 contact=api.find_social_entry(17,config=cfg)
 assert set(contact['tags'])=={'owner',*(f'worker_{n}' for n in range(4))}
 with store.connect(db) as connection:assert connection.execute('SELECT COUNT(*) FROM changes').fetchone()[0]==4


def test_locked_database_refuses_write_and_releases_cleanly(tmp_path):
 source,db,cfg,report=setup_store(tmp_path)
 with store.connect(db,write=True) as locked:
  assert api.update_social_entry(17,tags=['not_committed'],config=cfg) is False
 assert 'not_committed' not in api.find_social_entry(17,config=cfg)['tags']
 assert api.update_social_entry(17,tags=['after_release'],config=cfg)


def test_unknown_schema_and_missing_backend_do_not_fall_back(tmp_path):
 source,db,cfg,report=setup_store(tmp_path)
 with sqlite3.connect(db) as c:c.execute('PRAGMA user_version=999')
 assert api.load_social_map(cfg)==[] and not api.is_high_trust(17,config=cfg)
 assert not api.update_social_entry(17,config=cfg)
 assert api.get_owner_user_id(cfg) is None
 missing=tmp_path/'missing.sqlite';cfg['discord']['social_map_database_path']=str(missing)
 assert api.find_social_entry(17,config=cfg) is None and not missing.exists()


def test_json_updates_keep_link_wrapper_and_bad_source(tmp_path):
 target=tmp_path/'target.json';target.write_text(json.dumps({'people':fixture_entries(),'extra':'retain'}))
 link=tmp_path/'legacy.json';link.symlink_to(target)
 cfg={'discord':{'social_map_path':str(link)}}
 assert api.update_social_entry(29,tags=['new_tag'],config=cfg)
 assert link.is_symlink() and json.loads(target.read_text())['extra']=='retain'
 target.write_text('broken witness')
 assert not api.record_dm_attempt(11,'Fixture',config=cfg)
 assert target.read_text()=='broken witness'


def test_unavailable_storage_mount_never_creates_fallback(tmp_path):
 source=tmp_path/'absent_volume/social.json'
 cfg={'discord':{'social_map_path':str(source),'social_map_storage_mount':str(tmp_path/'absent_volume')}}
 assert api.load_social_map(cfg)==[] and not api.is_high_trust(17,config=cfg)
 assert api.update_social_entry(17,config=cfg) is False
 assert not source.parent.exists()


def test_payload_size_budget_and_rollback_on_callback_failure(tmp_path):
 source,db,cfg,report=setup_store(tmp_path);before=store.find(db,17)
 def bad(entries):entries[0]['display_name']='x'*(store.MAX_ENTRY_BYTES+1);return True
 with pytest.raises(ValueError):store.mutate(db,17,bad)
 assert store.find(db,17)==before
 with store.connect(db) as connection:
  plan=connection.execute('EXPLAIN QUERY PLAN SELECT payload_json FROM contacts WHERE user_id=?',('17',)).fetchall()
 assert any('INDEX' in str(row).upper() for row in plan)


def test_existing_move_seam_and_legacy_readers_follow_current_updates(tmp_path):
 from storage_migration import migrate_file_and_link
 from historical_source import historical_module
 source=tmp_path/'hdd/social_map.json';source.parent.mkdir();source.write_text(json.dumps(fixture_entries()))
 target=tmp_path/'nvme/social_map.json';original=source.read_bytes()
 report=migrate_file_and_link(source,target,apply=True,child=None)
 assert report['cutover'] and source.is_symlink()
 assert Path(report['backup']).read_bytes()==original
 cfg={'discord':{'social_map_path':str(target)}}
 assert api.update_social_entry(29,tags=['post_cutover'],config=cfg)
 legacy=historical_module('social_map.py','1f5293506dc87c3f1592e7d26b923c58545a7757')
 assert 'post_cutover' in legacy.find_social_entry(29,config={'discord':{'social_map_path':str(source)}})['tags']
 assert source.is_symlink() and Path(report['backup']).read_bytes()==original


def test_aggregate_payload_budget_and_old_snapshot_counter(tmp_path,monkeypatch):
 source,db,cfg,report=setup_store(tmp_path)
 with store.connect(db,write=True) as connection:
  connection.execute("DELETE FROM metadata WHERE key='payload_bytes'")
 assert api.update_social_entry(17,tags=['counter_backfill'],config=cfg)
 with store.connect(db) as connection:
  current=int(connection.execute("SELECT value FROM metadata WHERE key='payload_bytes'").fetchone()[0])
 before=store.find(db,17)
 monkeypatch.setattr(store,'MAX_SOURCE_BYTES',current+16)
 def expand(entries):entries[0]['extra_payload']='x'*128;return True
 with pytest.raises(ValueError):store.mutate(db,17,expand)
 assert store.find(db,17)==before
