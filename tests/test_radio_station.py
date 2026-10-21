import hashlib
import json
from pathlib import Path
import wave
import pytest
from ina_desktop.files import VirtualFileSystem, configured_drives
import radio_station as radio


def setup(tmp_path):
    cfg={'ina_hdd_writable_path':str(tmp_path/'personal'),
         'radio_station':{'enabled':True,'root':str(tmp_path/'station')}}
    fs=VirtualFileSystem(configured_drives(cfg,'Fixture',project_root=tmp_path));fs.ensure_writable_roots()
    path=tmp_path/'personal'/'song.wav'
    with wave.open(str(path),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000);wav.writeframes(b'\0\0'*4800)
    return fs,cfg,path


def fake_converter(source,work):
    (work/'input.opus').write_bytes(b'OggS'+b'\0'*20+b'OpusHead'+b'fixture audio')


def prepare(fs,cfg,**extra):
    return radio.prepare_clip(fs,cfg,relative_path='song.wav',title='<script>my song</script>',
                              kind='song',publication='ina_authored_public',converter=fake_converter,**extra)


def test_station_finite_playlist_and_stop_preserve_original(tmp_path):
    fs,cfg,path=setup(tmp_path);original=path.read_bytes();clip=prepare(fs,cfg)
    programme=radio.publish_program(cfg,clips=[clip['clip_id'],clip['clip_id']])
    assert len(programme['programme']['clips'])==2
    assert programme['programme']['repeat'] is False
    assert radio.stop_program(cfg)['on_air'] is False
    assert path.read_bytes()==original
    exported=json.loads((tmp_path/'station/public/station.json').read_text())
    assert exported['clips']==[]
    assert 'relative_path' not in json.dumps(programme)


def test_external_instructions_cannot_publish():
    from instruction_authority import seal_code_command, InstructionAuthorityError
    for action in ('radio_prepare_clip','radio_publish','radio_render_announcement','radio_stop'):
        with pytest.raises(InstructionAuthorityError):seal_code_command({'action':action,'source':'discord'},None)
        with pytest.raises(InstructionAuthorityError):seal_code_command({'action':action,'source':'ina_voluntary_choice'},None)


def test_traversal_symlinks_non_wav_and_unapproved_material_refused(tmp_path):
    fs,cfg,path=setup(tmp_path)
    for relative in ('../escape.wav','/etc/passwd','song.mp4'):
        with pytest.raises((ValueError,PermissionError,FileNotFoundError)):
            radio.prepare_clip(fs,cfg,relative_path=relative,title='x',kind='song',publication='ina_authored_public',converter=fake_converter)
    (path.parent/'link.wav').symlink_to(path)
    with pytest.raises(PermissionError):radio.prepare_clip(fs,cfg,relative_path='link.wav',title='x',kind='song',publication='ina_authored_public',converter=fake_converter)
    with pytest.raises(PermissionError):radio.prepare_clip(fs,cfg,relative_path='song.wav',title='x',kind='song',publication='external_reference',converter=fake_converter)


def test_export_parent_symlink_rejected(tmp_path):
    fs,cfg,path=setup(tmp_path);root=tmp_path/'station';root.mkdir();(root/'public').symlink_to(path.parent,target_is_directory=True)
    with pytest.raises(PermissionError):prepare(fs,cfg)


def test_unknown_tampered_and_nonfinite_programs_refused(tmp_path):
    fs,cfg,path=setup(tmp_path);clip=prepare(fs,cfg)
    for ids in ([],['../private'],['a'*64],[clip['clip_id']]*33):
        with pytest.raises(ValueError):radio.publish_program(cfg,clips=ids)
    with pytest.raises(ValueError):radio.publish_program(cfg,clips=[clip['clip_id']],start_at=float('nan'))
    (tmp_path/'station/public/media'/(clip['clip_id']+'.opus')).write_bytes(b'changed')
    with pytest.raises(ValueError):radio.publish_program(cfg,clips=[clip['clip_id']])


def test_remove_export_does_not_remove_original(tmp_path):
    fs,cfg,path=setup(tmp_path);clip=prepare(fs,cfg)
    radio.publish_program(cfg,clips=[clip['clip_id']])
    assert radio.remove_clip(cfg,clip_id=clip['clip_id'])['on_air'] is False
    assert radio.status(cfg)['clips']==[]
    assert path.exists()


def test_native_announcement_is_private_and_not_played(tmp_path):
    fs,cfg,path=setup(tmp_path);calls=[]
    def render(symbols,**kw):
        calls.append((symbols,kw));Path(kw['record_path']).write_bytes(path.read_bytes())
    result=radio.render_announcement(fs,symbols=['⟡'],child='Fixture',renderer=render)
    assert result['private'] and not result['on_air']
    assert calls[0][1]['playback'] is False


def test_busy_station_yields_instead_of_waiting(tmp_path):
    fs,cfg,path=setup(tmp_path);root=radio._settings(cfg)
    with radio._lock(root):
        with pytest.raises(BlockingIOError):radio.stop_program(cfg)


def test_actual_dispatcher_status_and_stop(tmp_path):
    fs,cfg,path=setup(tmp_path)
    from personal_tool_runtime import execute_personal_tool_command, capability_catalog
    from instruction_authority import seal_code_command, LOCAL_VOLUNTARY_CODE_AUTHORITY
    for action in ('radio_status','radio_stop'):
        assert action in capability_catalog()['commands']
        command=seal_code_command({'action':action,'source':'ina_voluntary_choice'},LOCAL_VOLUNTARY_CODE_AUTHORITY)
        value=execute_personal_tool_command(command,child='Fixture',project_root=tmp_path,config=cfg)['value']
        assert value is not None


def test_only_named_private_audio_drives_are_accepted(tmp_path):
    fs,cfg,path=setup(tmp_path)
    with pytest.raises(PermissionError):
        radio.prepare_clip(fs,cfg,relative_path='song.wav',source_drive='code',title='x',kind='song',publication='ina_authored_public',converter=fake_converter)


def test_render_directory_tool_reaches_existing_song_files(tmp_path,monkeypatch):
    fs,cfg,path=setup(tmp_path)
    render=tmp_path/'AI_Children/Fixture/memory/music_studio/renders';render.mkdir(parents=True)
    (render/'my_song.wav').write_bytes(path.read_bytes())
    from personal_tool_runtime import execute_personal_tool_command
    from instruction_authority import seal_code_command, LOCAL_VOLUNTARY_CODE_AUTHORITY
    actual=radio.prepare_clip
    def prepare_stub(*args,**kw):return actual(*args,converter=fake_converter,**kw)
    monkeypatch.setattr(radio,'prepare_clip',prepare_stub)
    command=seal_code_command({'action':'radio_prepare_clip','source':'ina_voluntary_choice',
        'source_drive':'music_renders','relative_path':'my_song.wav','title':'my song','kind':'song','publication':'ina_authored_public'},LOCAL_VOLUNTARY_CODE_AUTHORITY)
    value=execute_personal_tool_command(command,child='Fixture',project_root=tmp_path,config=cfg)['value']
    assert value['public_copy_prepared'] and not value['original_modified']


def test_installer_replaces_live_assets_without_truncating_old_readers(tmp_path):
    from tools.setup_radio_station import atomic_copy
    old=tmp_path/'old';new=tmp_path/'new';old.write_bytes(b'old asset');new.write_bytes(b'new asset')
    with old.open('rb') as existing_reader:
        atomic_copy(new,old,0o600)
        assert existing_reader.read()==b'old asset'
    assert old.read_bytes()==b'new asset'
