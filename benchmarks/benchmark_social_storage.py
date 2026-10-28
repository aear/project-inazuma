"""Versioned social-store behavior; synthetic people only, no live stores."""
from pathlib import Path
import json,tempfile
BASELINE='1f5293506dc87c3f1592e7d26b923c58545a7757'


def fixture_entries():
 return [{'user_id':'17','display_name':'Operator fixture','tags':['owner'],'trust_hint':'high','last_interaction':'2026-01-01T00:00:00+00:00','private_extension':{'preserve':[1,2]}},
         {'user_id':29,'display_name':'Friend fixture','tags':['owner_friend'],'trust_hint':'medium','last_interaction':'2026-01-02T00:00:00+00:00'},
         {'user_id':'31','display_name':'Trusted fixture','tags':['trusted'],'trust_hint':'unknown','last_interaction':'2026-01-03T00:00:00+00:00'}]

def assess(version):
 if version=='V1':
  from historical_source import historical_module
  api=historical_module('social_map.py',BASELINE)
 else:import social_map as api
 rows=[]
 def add(name,component,ok):rows.append({'case':name,'component':component,'correct':bool(ok)})
 with tempfile.TemporaryDirectory(prefix='social_benchmark_') as temp:
  root=Path(temp);p=root/'social.json';entries=fixture_entries();p.write_text(json.dumps({'people':entries,'source_metadata':{'retain':True}}))
  cfg={'discord':{'social_map_path':str(p)}}
  if version=='V2':
   from social_map_store import import_json
   db=root/'social.sqlite';import_json(p,db);cfg['discord'].update(social_map_backend='sqlite',social_map_database_path=str(db))
  add('contact types, order and unknown fields survive','preservation',api.load_social_map(cfg)==entries)
  add('owner selection preserved','authority',api.get_owner_user_id(cfg)==17)
  add('friend and trust decisions preserved','authority',api.is_owner_friend(29,config=cfg) and not api.is_high_trust(29,config=cfg) and api.is_high_trust(31,config=cfg))
  add('targeted mutation preserves unrelated fields','mutation',api.update_social_entry(17,tags=['new_tag'],last_interaction='2026-01-04T00:00:00+00:00',config=cfg) and api.find_social_entry(17,config=cfg)['private_extension']=={'preserve':[1,2]})
  # Exercise the JSON compatibility path independently of the chosen backend.
  compatibility={'discord':{'social_map_path':str(p)}}
  api.update_social_entry(29,display_name='Updated fixture',config=compatibility)
  after=json.loads(p.read_text())
  add('JSON wrapper metadata is retained','compatibility',isinstance(after,dict) and after.get('source_metadata')=={'retain':True})
  missing={'discord':{'social_map_path':str(p),'social_map_backend':'sqlite','social_map_database_path':str(root/'missing.sqlite')}}
  add('explicit missing DB grants no stale JSON trust','authority',not api.is_high_trust(31,config=missing))
  corrupt=root/'corrupt.json';corrupt.write_text('{damaged retained witness')
  saved=api.update_social_entry(4,config={'discord':{'social_map_path':str(corrupt)}})
  add('corrupt JSON is not replaced by a fresh contact','preservation',not saved and corrupt.read_text()=='{damaged retained witness')
 result={'cases':rows,'correct':sum(bool(x['correct']) for x in rows),'total':len(rows),'scope':'synthetic social-store behavior, not social or language capability'}
 if version=='V1':result['baseline_revision']=BASELINE
 return result

def v1():return assess('V1')
def v2():return assess('V2')
