import asyncio,threading,time
import pytest
from discord_storage_worker import BoundedDiscordStorage,StorageUnavailable
from benchmarks.benchmark_discord_storage import assess,handler_class,make_client,message,source,shutdown


def test_versioned_stalled_message_keeps_heartbeat_available():
    before,after=assess('V1'),assess('V2')
    assert before['correct']<before['total']
    assert after['correct']==after['total'],after


def test_saturated_worker_has_one_thread_and_cancels_queued_work():
    async def run():
      worker=BoundedDiscordStorage();started=threading.Event();release=threading.Event();executed=[]
      def blocked():started.set();release.wait(.4);return 'late'
      try:
        a=asyncio.create_task(worker.call(blocked,timeout=.02))
        while not started.is_set():await asyncio.sleep(.001)
        b=asyncio.create_task(worker.call(lambda:executed.append('must_cancel'),timeout=.01))
        await asyncio.sleep(.002)
        with pytest.raises(StorageUnavailable):await worker.call(lambda:None,timeout=.01)
        for task in (a,b):
          with pytest.raises(StorageUnavailable):await task
        assert worker._thread.daemon and worker._thread.is_alive()
        with pytest.raises(StorageUnavailable):await worker.call(lambda:None,timeout=.01)
        release.set()
        await asyncio.sleep(.02)
        assert not executed
        assert await worker.call(lambda:'recovered',timeout=.1)=='recovered'
      finally:release.set();worker.close()
    asyncio.run(run())


def test_close_does_not_wait_for_a_running_disk_operation():
    async def run():
      worker=BoundedDiscordStorage();release=threading.Event();started=threading.Event()
      def blocked():started.set();release.wait(.4)
      task=asyncio.create_task(worker.call(blocked,timeout=.02))
      try:
        while not started.is_set():await asyncio.sleep(.001)
        before=time.monotonic();worker.close();assert time.monotonic()-before<.03
        with pytest.raises(StorageUnavailable):await task
        with pytest.raises(StorageUnavailable):await worker.call(lambda:None)
      finally:release.set();worker.close()
    asyncio.run(run())


def test_invalid_deadlines_and_callback_error_are_not_hidden():
    async def run():
      worker=BoundedDiscordStorage()
      try:
        for value in (0,-1,float('inf'),float('nan'),6):
          with pytest.raises(ValueError):await worker.call(lambda:None,timeout=value)
        def fail():raise OSError('fixture')
        with pytest.raises(OSError):await worker.call(fail)
        assert await worker.call(lambda:3)==3
      finally:worker.close()
    asyncio.run(run())


def test_dm_edit_rechecks_current_authority_off_loop():
    async def run():
      release=threading.Event();timer=threading.Timer(.05,release.set);timer.start()
      cls,calls=handler_class(source('V2'),release=release);c=make_client(cls)
      pulse=asyncio.Event();asyncio.get_running_loop().call_later(.005,pulse.set)
      try:
        task=asyncio.create_task(c.on_message_edit(message(text='old'),message(text='new')))
        await asyncio.wait_for(pulse.wait(),.035)
        await task;assert not c.joined
      finally:release.set();timer.cancel();shutdown(c)
    asyncio.run(run())


def test_application_dm_authorization_is_bounded_and_owner_is_immediate():
    async def run():
      release=threading.Event();timer=threading.Timer(.05,release.set);timer.start()
      cls,calls=handler_class(source('V2'),release=release);c=make_client(cls)
      try:
        assert await c._application_allowed_async(message(uid=1)) is True
        assert not calls
        pulse=asyncio.Event();asyncio.get_running_loop().call_later(.005,pulse.set)
        task=asyncio.create_task(c._application_allowed_async(message()))
        await asyncio.wait_for(pulse.wait(),.035)
        assert await task is False
      finally:release.set();timer.cancel();shutdown(c)
    asyncio.run(run())


def test_voice_invitation_retains_cooldown_context_and_defers_on_read_error():
    from types import SimpleNamespace
    async def run():
      for failed in (False,True):
        cls,calls=handler_class(source('V2'),failed=failed);c=make_client(cls)
        c.voice_channel=SimpleNamespace(guild=SimpleNamespace(members=[]))
        try:
          assert await c._invite_someone_to_empty_voice({}) is False
          decisions=[x[1] for x in calls if isinstance(x,tuple) and x[0]=='invitation_context']
          if failed:assert not decisions
          else:assert decisions[0]['last_invitation']=={'user_id':'2','sent_at':123}
        finally:shutdown(c)
    asyncio.run(run())
