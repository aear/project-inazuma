"""Real extracted gateway handlers with synthetic I/O; never starts Discord."""
from __future__ import annotations
import ast,asyncio,logging,threading,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from discord_storage_worker import BoundedDiscordStorage,StorageUnavailable
BASELINE='1f5293506dc87c3f1592e7d26b923c58545a7757'


def handler_class(source, *, release=None, trusted=False, failed=False):
    original=ast.parse(source);client=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='InaDiscordClient')
    names={'on_message','on_message_edit','_storage_call','_social_flags','_application_allowed_async','_application_command_allowed','_ingest_image_attachments','_invite_someone_to_empty_voice'}
    nodes=[n for n in client.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    shell=ast.parse('from __future__ import annotations\nclass Harness: pass\n');shell.body[1].body=nodes
    calls=[]
    def check(*args,**kwargs):
        calls.append('lookup')
        if release is not None:release.wait(.3)
        if failed:raise OSError('synthetic disk unavailable')
        return trusted
    def read(*a,**k):
        check();return []
    env={'asyncio':asyncio,'time':time,'logger':logging.getLogger('storage_fixture'),
         'BoundedDiscordStorage':BoundedDiscordStorage,'StorageUnavailable':StorageUnavailable,
         'SAKURA_USER_ID':1,'get_discord_config':lambda:{},'load_root_config':lambda:{},
         'is_owner_friend':check,'is_high_trust':lambda *a,**k:False,'load_social_map':read,
          'get_memory_guard_level':lambda:calls.append('memory_guard') or 'normal',
         'get_inastate':lambda key:check() or {'user_id':'2','sent_at':123},
         'get_high_trust_contacts':lambda **k:[],
         'select_voice_invitation_candidate':lambda cfg,**k:calls.append(('invitation_context',k)) or {'selected':False},
         'record_voice_cognition':lambda *a,**k:None,
         'record_dm_attempt':lambda **k:False,'VOICE_JOIN_COMMANDS':{'/join'},'VOICE_LEAVE_COMMANDS':{'/leave'},
         'datetime':datetime,'timezone':timezone,'infer_message_edit':lambda *a:{'likely_reason':'fixture'}}
    exec(compile(ast.fix_missing_locations(shell),'<extracted Discord handlers>','exec'),env)
    return env['Harness'],calls


def message(*,uid=2,dm=True,text='/join'):
    return SimpleNamespace(author=SimpleNamespace(id=uid,bot=False,display_name='Fixture'),
        guild=None if dm else SimpleNamespace(id=30),channel=SimpleNamespace(id=20),content=text,id=40,attachments=[])


def make_client(cls):
    c=cls();c.child='Fixture';c.user=SimpleNamespace(id=99);c.text_channel=None;c.joined=[];c.contacts=[]
    async def join(msg):c.joined.append(msg.author.id)
    async def noop(*a,**k):return None
    async def context(*a):return []
    c._handle_voice_join=join;c._handle_voice_leave=noop;c._remember_last_dm_contact=lambda m:None
    c._record_social_contact=lambda m:c.contacts.append(m.author.id);c._recent_message_context=context
    c._roleplay_mode=lambda m:None
    return c


def shutdown(client):
    worker=getattr(client,'_storage_worker',None)
    if worker is not None:worker.close()


def source(version):
    if version=='V1':
        from historical_source import historical_text
        return historical_text('discord_bridge.py',BASELINE)
    return Path(__file__).resolve().parents[1].joinpath('discord_bridge.py').read_text()


async def stalled_route(version):
    release=threading.Event();timer=threading.Timer(.08,release.set);timer.daemon=True
    cls,calls=handler_class(source(version),release=release)
    client=make_client(cls)
    pulse=asyncio.Event();start=time.monotonic();observed={}
    def heartbeat():observed['heartbeat_seconds']=time.monotonic()-start;pulse.set()
    asyncio.get_running_loop().call_later(.005,heartbeat);timer.start()
    try:
        if version=='V2':
            original=client._storage_call
            async def short(function,*args,default=None,**kwargs):
                worker=getattr(client,'_storage_worker',None)
                if worker is None:worker=client._storage_worker=BoundedDiscordStorage()
                try:return await worker.call(function,*args,timeout=.015,**kwargs)
                except Exception:return default
            client._storage_call=short
        await client.on_message(message())
        await asyncio.wait_for(pulse.wait(),.3)
        observed.update({'authority_granted':bool(client.joined),'lookup_calls':len(calls)})
        return observed
    finally:release.set();timer.cancel();shutdown(client)


def assess(version):
    observations=[asyncio.run(stalled_route(version)) for _ in range(3)]
    rows=[{'case':'heartbeat runs while storage is blocked','component':'responsiveness','correct':all(x['heartbeat_seconds']<.04 for x in observations)},
          {'case':'unavailable or untrusted lookup grants no authority','component':'authority','correct':all(not x['authority_granted'] for x in observations)}]
    if version=='V2':
      async def checks():
        cls,calls=handler_class(source(version));c=make_client(cls)
        try:
          await c.on_message(message(uid=1));owner_ok=c.joined==[1] and not calls
        finally:shutdown(c)
        cls,calls=handler_class(source(version),trusted=True);c=make_client(cls)
        try:await c.on_message(message());friend_ok=c.joined==[2]
        finally:shutdown(c)
        cls,calls=handler_class(source(version),failed=True);c=make_client(cls)
        try:await c.on_message(message());error_ok=not c.joined
        finally:shutdown(c)
        return owner_ok,friend_ok,error_ok
      values=asyncio.run(checks())
    else:
      # Execute the historical owner path rather than infer it from a marker.
      async def owner_check():
        cls,calls=handler_class(source(version));c=make_client(cls)
        await c.on_message(message(uid=1));return c.joined==[1] and not calls
      async def other_checks():
        cls,calls=handler_class(source(version),trusted=True);c=make_client(cls)
        await c.on_message(message());friend_ok=c.joined==[2]
        cls,calls=handler_class(source(version),failed=True);c=make_client(cls)
        try:
          await c.on_message(message());error_ok=not c.joined
        except OSError:error_ok=False
        return friend_ok,error_ok
      values=(asyncio.run(owner_check()),*asyncio.run(other_checks()))
    rows.extend({'case':case,'component':component,'correct':value} for case,component,value in zip(
      ('configured owner needs no social-map authority','verified friend remains permitted','lookup error is contained'),
      ('owner_access','compatibility','failure_isolation'),values))
    async def image_check():
      cls,calls=handler_class(source(version));c=make_client(cls)
      try:
        result=await c._ingest_image_attachments(message())
        return result==[] and 'memory_guard' not in calls
      finally:shutdown(c)
    rows.append({'case':'text-only event skips storage-backed image guard','component':'text_path','correct':asyncio.run(image_check())})
    async def invitation_check():
      cls,calls=handler_class(source(version),failed=True);c=make_client(cls)
      c.voice_channel=SimpleNamespace(guild=SimpleNamespace(members=[]))
      try:
        try:result=await c._invite_someone_to_empty_voice({})
        except OSError:return False
        return result is False and not any(isinstance(x,tuple) and x[0]=='invitation_context' for x in calls)
      finally:shutdown(c)
    rows.append({'case':'unavailable invitation cooldown evidence defers invitation','component':'authority','correct':asyncio.run(invitation_check())})
    report={'cases':rows,'correct':sum(bool(x['correct']) for x in rows),'total':len(rows),'observations':observations,
            'scope':'extracted actual message handlers; synthetic stalled disk; shortened test deadlines; no live client or files'}
    if version=='V1':report['baseline_revision']=BASELINE
    return report

def v1():return assess('V1')
def v2():return assess('V2')
