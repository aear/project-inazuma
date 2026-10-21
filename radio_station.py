"""Voluntary finite radio programmes. Public files only; no HTTP control server."""
from __future__ import annotations
import hashlib
import fcntl
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
import wave
from io_utils import atomic_write_json

MAX_SOURCE = 256*1024*1024
MAX_EXPORT = 32*1024*1024
MAX_CLIPS = 100
MAX_BYTES = 2*1024*1024*1024
ASSET = re.compile(r'^[0-9a-f]{64}$')


def _regular(path: Path, limit: int):
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise PermissionError('symlinks are not radio inputs or destinations')
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > limit:
        raise ValueError('radio file type/size limit')


def _settings(config):
    settings=config.get('radio_station') or {}
    if settings.get('enabled') is not True:
        raise PermissionError('radio tools are not configured')
    root=Path(settings['root'])
    if not root.is_absolute() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise PermissionError('radio root must be a configured absolute non-symlink directory')
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    for child in (root/'public',root/'public'/'media'):
        if child.is_symlink():raise PermissionError('export directory symlink')
    (root/'public'/'media').mkdir(parents=True,exist_ok=True,mode=0o700)
    return root


@contextmanager
def _lock(root):
    fd=os.open(root/'station.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _serialized(function):
    def call(*args,**kwargs):
        config=kwargs.get('config')
        if config is None:config=args[1] if function.__name__=='prepare_clip' else args[0]
        root=_settings(config)
        with _lock(root):return function(*args,**kwargs)
    return call


def _catalog(root):
    p=root/'catalog.json'
    if not p.exists(): return {}
    _regular(p,100000)
    data=json.loads(p.read_text())
    if not isinstance(data,dict) or len(data)>MAX_CLIPS:raise ValueError('catalog limit')
    for key,row in data.items():
        if not ASSET.fullmatch(key) or not isinstance(row,dict):raise ValueError('catalog identity')
        if row.get('kind') not in {'song','speech'} or not isinstance(row.get('title'),str) or len(row['title'])>160:
            raise ValueError('catalog metadata')
        duration=row.get('duration')
        if type(duration) not in (int,float) or not math.isfinite(duration) or not .05<=duration<=1200:
            raise ValueError('catalog duration')
        if type(row.get('bytes')) is not int or not 1<=row['bytes']<=MAX_EXPORT:raise ValueError('catalog size')
    return data


def convert_isolated(source: Path, work: Path):
    """Only a copied PCM WAV and output directory enter the decoder namespace."""
    bwrap=shutil.which('bwrap')
    if not bwrap: raise RuntimeError('decoder isolation unavailable')
    command=['/usr/bin/prlimit','--as=1073741824','--cpu=120','--fsize=33554432','--nofile=64','--',
        bwrap,'--unshare-all','--die-with-parent','--new-session','--clearenv',
        '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib',
        '--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
        '--dir','/etc','--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache',
        '--bind',str(work),'/work','--ro-bind',str(source),'/work/input.wav',
        '--ro-bind',str(Path(__file__).with_name('music_delivery.py')),'/music_delivery.py','--chdir','/',
        '--setenv','PATH','/usr/bin:/bin','--setenv','HOME','/tmp',
        '/usr/bin/python','-c',
        'import json;from music_delivery import ensure_opus_sidecar;print(json.dumps(ensure_opus_sidecar("/work/input.wav",public_export=True,timeout_seconds=120)))']
    result=subprocess.run(command,capture_output=True,timeout=130,check=False)
    if result.returncode or not (work/'input.opus').is_file():
        raise RuntimeError('isolated audio conversion failed')
    receipt=json.loads(result.stdout[-10000:])
    if receipt.get('status')!='converted':raise RuntimeError('audio conversion refused')


@_serialized
def prepare_clip(fs, config, *, relative_path, title, kind, publication, source_drive="ina_hdd", converter=convert_isolated):
    if publication!='ina_authored_public' or kind not in {'song','speech'}:
        raise PermissionError('choose a public Ina-authored song or speech clip explicitly')
    if not isinstance(title,str) or not title.strip() or len(title)>160:raise ValueError('title limit')
    if not isinstance(relative_path,str) or Path(relative_path).suffix.lower()!='.wav':raise ValueError('choose one PCM WAV')
    root=_settings(config);catalog=_catalog(root)
    if len(catalog)>=MAX_CLIPS:raise ValueError('radio clip quota reached')
    # Reuse existing capability-scoped path checks. Never accept an arbitrary host path.
    if source_drive not in {'ina_hdd','music_renders'} or source_drive not in fs.drives:
        raise PermissionError('choose personal files or Ina music renders only')
    source=fs._path(fs.drives[source_drive],relative_path,existing=True)
    _regular(source,MAX_SOURCE)
    with tempfile.TemporaryDirectory(prefix='convert-',dir=root) as directory:
        work=Path(directory);copy=work/'input.wav'
        fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as incoming,copy.open('xb') as outgoing:
            if not stat.S_ISREG(os.fstat(incoming.fileno()).st_mode):raise ValueError('regular file required')
            total=0
            while block:=incoming.read(1024*1024):
                total+=len(block)
                if total>MAX_SOURCE:raise ValueError('source grew beyond limit')
                outgoing.write(block)
        with wave.open(str(copy),'rb') as wav:
            duration=wav.getnframes()/wav.getframerate()
            if wav.getcomptype()!='NONE' or wav.getnchannels() not in {1,2} or not .05<=duration<=1200:
                raise ValueError('PCM channel/duration limit')
        converter(copy,work)
        opus=work/'input.opus';_regular(opus,MAX_EXPORT)
        data=opus.read_bytes()
        if not data.startswith(b'OggS') or b'OpusHead' not in data[:256]:raise ValueError('Opus output required')
        digest=hashlib.sha256(data).hexdigest()
        if sum(int(row['bytes']) for row in catalog.values())+len(data)>MAX_BYTES:raise ValueError('radio byte quota')
        destination=root/'public'/'media'/(digest+'.opus')
        if destination.is_symlink():raise PermissionError('export symlink refused')
        if not destination.exists():
            with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.export-',delete=False) as handle:
                handle.write(data);staging=Path(handle.name)
            try:os.chmod(staging,0o600);os.replace(staging,destination)
            finally:staging.unlink(missing_ok=True)
        catalog[digest]={'title':title,'kind':kind,'duration':duration,'bytes':len(data)}
        atomic_write_json(root/'catalog.json',catalog,ensure_ascii=False)
    return {'clip_id':digest,'title':title,'kind':kind,'duration':duration,'public_copy_prepared':True,
            'on_air':False,'original_modified':False}


@_serialized
def publish_program(config, *, clips, start_at=None):
    root=_settings(config);catalog=_catalog(root)
    if not isinstance(clips,list) or not 1<=len(clips)<=32:raise ValueError('choose 1..32 clips')
    stamp=time.time() if start_at is None else float(start_at)
    if not math.isfinite(stamp) or abs(stamp-time.time())>86400:raise ValueError('start within one day')
    selected=[]
    for clip in clips:
        if not isinstance(clip,str) or not ASSET.fullmatch(clip) or clip not in catalog:raise ValueError('unknown clip ID')
        path=root/'public'/'media'/(clip+'.opus');_regular(path,MAX_EXPORT)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=clip:raise ValueError('export digest changed')
        row=catalog[clip]
        selected.append({'id':clip,'title':row['title'],'kind':row['kind'],'duration':row['duration']})
    duration=sum(row['duration'] for row in selected)
    if duration>7200:raise ValueError('finite programme exceeds two hours')
    value={'schema':'ina.radio_program/V1','revision':uuid.uuid4().hex,'start_at':stamp,
           'clips':selected,'duration':duration,'repeat':False}
    atomic_write_json(root/'public'/'station.json',value,ensure_ascii=False)
    return {'on_air':True,'programme':value,'public_url':'https://radio.ai-nation.co.uk/',
            'internet_reachability':'not_verified'}


@_serialized
def stop_program(config):
    root=_settings(config)
    atomic_write_json(root/'public'/'station.json',{'schema':'ina.radio_program/V1',
        'revision':uuid.uuid4().hex,'clips':[],'repeat':False},ensure_ascii=False)
    return {'on_air':False,'note':'previously downloaded public copies cannot be recalled'}


def status(config):
    root=_settings(config);catalog=_catalog(root)
    return {'clips':[{'id':key,**row} for key,row in catalog.items()],
            'publication':'explicit choice only','public_url':'https://radio.ai-nation.co.uk/',
            'internet_reachability':'not_verified','listener_control_api':False}


@_serialized
def remove_clip(config, *, clip_id):
    root=_settings(config);catalog=_catalog(root)
    if not isinstance(clip_id,str) or not ASSET.fullmatch(clip_id):raise ValueError('clip ID required')
    if clip_id not in catalog:raise ValueError('unknown clip')
    # Stop before retiring an asset; a publisher cannot select it during removal.
    atomic_write_json(root/'public'/'station.json',{'schema':'ina.radio_program/V1',
        'revision':uuid.uuid4().hex,'clips':[],'repeat':False},ensure_ascii=False)
    path=root/'public'/'media'/(clip_id+'.opus');_regular(path,MAX_EXPORT);path.unlink()
    del catalog[clip_id];atomic_write_json(root/'catalog.json',catalog,ensure_ascii=False)
    return {'removed':clip_id,'on_air':False,'downloaded_copies_recallable':False}


def render_announcement(fs, *, symbols, child, renderer=None):
    if not isinstance(symbols,list) or not 1<=len(symbols)<=8 or any(
        not isinstance(s,str) or not s or len(s)>64 for s in symbols):raise ValueError('choose 1..8 native voice symbols')
    if renderer is None:
        from language_processing import speak_symbolically
        renderer=speak_symbolically
    relative='Radio Speech/'+uuid.uuid4().hex+'.wav'
    path=fs._path(fs.drives['ina_hdd'],relative)
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    renderer(symbols,child=child,record_path=path,playback=False,record_format='wav')
    _regular(path,MAX_SOURCE)
    return {'relative_path':relative,'private':True,'on_air':False,
            'next_choice':'radio_prepare_clip with kind=speech and explicit public permission'}
