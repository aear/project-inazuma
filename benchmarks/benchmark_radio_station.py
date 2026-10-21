from pathlib import Path
import sys
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import wave
from time import perf_counter
from historical_source import historical_module
BASELINE='26250eded439c36b8f2a88dc2e090f19fd353445'


def measure(version='V2'):
    import radio_station as radio
    from ina_desktop.files import VirtualFileSystem,configured_drives
    from personal_tool_runtime import capability_catalog
    catalog=historical_module('personal_tool_runtime.py',BASELINE).capability_catalog() if version=='V1' else capability_catalog()
    commands=catalog['commands'];start=perf_counter()
    checks={'chosen_station_tools_reachable':all(k in commands for k in ('radio_publish','radio_stop','radio_prepare_clip','radio_render_announcement','radio_remove_clip'))}
    if version=='V1':
        checks.update({key:False for key in ('finite_program','private_paths_not_published','public_copy_preserves_original','stop_and_remove','native_voice_stays_private','atomic_public_asset_updates')})
    else:
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg={'ina_hdd_writable_path':str(root/'personal'),'radio_station':{'enabled':True,'root':str(root/'radio')}}
            fs=VirtualFileSystem(configured_drives(cfg,'Fixture',project_root=root));fs.ensure_writable_roots()
            wav=root/'personal'/'fixture.wav'
            with wave.open(str(wav),'wb') as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(48000);h.writeframes(b'\0\0'*4800)
            original=wav.read_bytes()
            def converter(source,work):(work/'input.opus').write_bytes(b'OggS'+b'\0'*20+b'OpusHead'+b'fixture')
            clip=radio.prepare_clip(fs,cfg,relative_path='fixture.wav',title='chosen fixture',kind='song',publication='ina_authored_public',converter=converter)
            public=radio.publish_program(cfg,clips=[clip['clip_id']])
            checks['finite_program']=public['programme']['repeat'] is False and len(public['programme']['clips'])==1
            checks['private_paths_not_published']=str(root) not in json.dumps(public)
            checks['public_copy_preserves_original']=wav.read_bytes()==original and clip['public_copy_prepared']
            result=radio.render_announcement(fs,symbols=['⟡'],child='Fixture',renderer=lambda symbols,**kw:Path(kw['record_path']).write_bytes(original))
            checks['native_voice_stays_private']=result['private'] and not result['on_air']
            radio.stop_program(cfg);radio.remove_clip(cfg,clip_id=clip['clip_id'])
            checks['stop_and_remove']=radio.status(cfg)['clips']==[] and wav.exists()
            from tools.setup_radio_station import atomic_copy
            before=root/'before';after=root/'after';before.write_bytes(b'old');after.write_bytes(b'new')
            with before.open('rb') as old_handle:
                atomic_copy(after,before,0o600)
                checks['atomic_public_asset_updates']=old_handle.read()==b'old' and before.read_bytes()==b'new'

    return {'signals':checks,'elapsed_seconds':perf_counter()-start,'scope':'synthetic control flow; audio and live HTTP tested separately'}

if __name__=='__main__':print(json.dumps({v:measure(v) for v in ('V1','V2')},indent=2))
