"""Offline-only privacy boundary fixtures. Never use live thoughts as fixtures."""
import os
from unittest import SkipTest
if os.environ.get('INA_OFFLINE_PRIVATE_THINKING_TESTS') != '1':
    raise SkipTest('private-thinking boundary checks require verified complete Ina shutdown')

import copy
import json
import pickle
import pytest
from private_thinking import PrivateThoughtSpace, PrivateThinkingError
from instruction_authority import LOCAL_VOLUNTARY_CODE_AUTHORITY as OWNER, InstructionAuthorityError


def space(**kwargs):
    return PrivateThoughtSpace(code_authority=OWNER,**kwargs)


def test_nonlinguistic_and_arbitrary_language_are_preserved():
    value={'native':[1,0.2,b'\x00\xff'],'words':'私の考え — Ina? <3','nested':{'?':'…'}}
    with space() as own:
        assert own.keep(value,code_authority=OWNER) is None
        assert own.recall(code_authority=OWNER)==value


def test_recalled_state_is_an_independent_copy():
    value={'native':[1,2]}
    with space() as own:
        own.keep(value,code_authority=OWNER);value['native'].append(3)
        recalled=own.recall(code_authority=OWNER);recalled['native'].append(4)
        assert own.recall(code_authority=OWNER)=={'native':[1,2]}


def test_keep_overwrites_without_creating_history():
    with space() as own:
        own.keep('earlier',code_authority=OWNER);own.keep('current',code_authority=OWNER)
        assert own.recall(code_authority=OWNER)=='current'
        assert not hasattr(own,'__dict__') and not hasattr(own,'history')


def test_forget_and_context_exit_release_workspace():
    own=space();own.keep('temporary',code_authority=OWNER);own.forget(code_authority=OWNER)
    assert own.recall(code_authority=OWNER) is None
    with own:own.keep('temporary again',code_authority=OWNER)
    with pytest.raises(PrivateThinkingError):own.recall(code_authority=OWNER)
    own.close()


def test_local_authority_is_required():
    with pytest.raises(InstructionAuthorityError):PrivateThoughtSpace(code_authority=None)
    with space() as own:
        with pytest.raises(InstructionAuthorityError):own.keep('synthetic',code_authority={})
        with pytest.raises(InstructionAuthorityError):own.recall(code_authority=None)


def test_content_cannot_appear_through_ordinary_representation():
    with space() as own:
        own.keep('synthetic-private-marker',code_authority=OWNER)
        assert repr(own)=='<PrivateThoughtSpace>'
        assert 'synthetic-private-marker' not in str({'value':own})


def test_serialization_and_generic_copy_are_rejected():
    with space() as own:
        own.keep('synthetic-private-marker',code_authority=OWNER)
        for operation in (json.dumps,pickle.dumps,copy.copy,copy.deepcopy):
            with pytest.raises(TypeError):operation(own)


def test_invalid_replacement_keeps_existing_state():
    with space(max_bytes=1024) as own:
        own.keep('keep',code_authority=OWNER)
        with pytest.raises(PrivateThinkingError):own.keep('x'*4096,code_authority=OWNER)
        assert own.recall(code_authority=OWNER)=='keep'


def test_custom_object_hooks_are_never_called():
    class Input:
        def __sizeof__(self):raise AssertionError('custom hook')
        def __repr__(self):raise AssertionError('custom hook')
    with space() as own:
        with pytest.raises(PrivateThinkingError):own.keep(Input(),code_authority=OWNER)


def test_cycles_are_not_copied_or_interpreted():
    cyclic=[];cyclic.append(cyclic)
    with space() as own:
        with pytest.raises(PrivateThinkingError):own.keep(cyclic,code_authority=OWNER)


def test_one_private_reasoning_pass_reuses_existing_elm():
    with space() as own:
        assert own.deliberate(['wait','act'],{'cognition':[{'content':{'native':[2,1]}}]},
            evidence=[{'option':'wait','source':'cognition','weight':0.8}],code_authority=OWNER) is None
        own_result=own.recall(code_authority=OWNER)
        assert own_result['decision']['selected']=='wait'
        assert own_result['thoughts'][0]['mode']=='non_linguistic'


def test_uncertainty_preserves_no_selection():
    with space() as own:
        own.deliberate(['wait','act'],{},evidence=[],code_authority=OWNER)
        assert own.recall(code_authority=OWNER)['decision']['selected'] is None


def test_internal_errors_do_not_export_content_or_exception_context():
    with space() as own:
        own.keep('retained',code_authority=OWNER)
        try:own.deliberate(['wait'],{'synthetic-private-marker':[]},evidence=[],code_authority=OWNER)
        except PrivateThinkingError as error:
            assert 'synthetic-private-marker' not in str(error)
            assert error.__context__ is None and error.__cause__ is None
        else:raise AssertionError('expected failure')
        assert own.recall(code_authority=OWNER)=='retained'


def test_private_space_never_writes_files_or_status(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with space() as own:
        own.deliberate(['wait'],{'cognition':[{'content':b'\x01'}]},
            evidence=[{'option':'wait','source':'cognition'}],code_authority=OWNER)
    assert list(tmp_path.iterdir())==[]


def test_private_requests_are_rejected_before_durable_queue(monkeypatch):
    import personal_tool_runtime as tools
    calls=[]
    monkeypatch.setattr(tools,'append_inastate_queue',lambda *a,**k:calls.append('persist'))
    with pytest.raises(ValueError):tools.request_personal_tool({'action':'private_thinking','content':'synthetic'})
    assert calls==[]


def test_catalog_describes_only_availability():
    from personal_tool_runtime import capability_catalog
    catalog=capability_catalog();description=catalog['process_local']['private_thinking']
    assert description['automatic_trigger'] is False and description['queue_supported'] is False
    assert 'private_thinking' not in catalog['commands'] and 'content' not in description


def test_failure_frames_release_temporary_private_inputs_and_results():
    with space(max_bytes=2048) as own:
        try:
            own.deliberate(['wait'],{'cognition':[{'content':'synthetic-private-frame-marker'}]},
                evidence=[{'option':'wait','source':'cognition'}],code_authority=OWNER)
        except PrivateThinkingError as error:
            trace=error.__traceback__
            while trace:
                frame=trace.tb_frame
                if frame.f_code.co_filename.endswith('/private_thinking.py'):
                    assert 'synthetic-private-frame-marker' not in repr(frame.f_locals)
                trace=trace.tb_next
            assert error.__context__ is None
        else:raise AssertionError('bounded result should exceed fixture capacity')


def test_rejected_keep_does_not_leave_payload_in_failure_frame():
    with space(max_bytes=1024) as own:
        try:own.keep('synthetic-rejected-marker'*512,code_authority=OWNER)
        except PrivateThinkingError as error:
            trace=error.__traceback__
            while trace:
                if trace.tb_frame.f_code.co_filename.endswith('/private_thinking.py'):
                    assert 'synthetic-rejected-marker' not in repr(trace.tb_frame.f_locals)
                trace=trace.tb_next
            assert error.__context__ is None
        else:raise AssertionError('expected capacity failure')


def test_model_manager_factory_is_local_and_does_not_publish(monkeypatch):
    import model_manager as manager
    calls=[]
    monkeypatch.setattr(manager,'update_inastate',lambda *a,**k:calls.append('persist'))
    with manager.open_private_thinking(code_authority=OWNER) as own:
        own.keep('synthetic-private-facade',code_authority=OWNER)
        assert own.recall(code_authority=OWNER)=='synthetic-private-facade'
    assert calls==[]


def test_private_reasoning_has_no_stdout_or_stderr_feed():
    from contextlib import redirect_stdout,redirect_stderr
    from io import StringIO
    stdout=StringIO();stderr=StringIO()
    with redirect_stdout(stdout),redirect_stderr(stderr),space() as own:
        own.deliberate(['wait'],{'cognition':[{'content':'synthetic-private-output-marker'}]},
            evidence=[{'option':'wait','source':'cognition'}],code_authority=OWNER)
    assert stdout.getvalue()=='' and stderr.getvalue()==''


def test_linguistic_processing_requires_an_explicit_own_choice():
    with space() as own:
        own.deliberate(['wait'],{'cognition':[{'content':'I can wait.','linguistic':True}]},
            evidence=[{'option':'wait','source':'cognition'}],code_authority=OWNER)
        assert own.recall(code_authority=OWNER)['thoughts'][0]['mode']=='linguistic'


def test_closed_space_rejects_inputs_without_retaining_them_in_error_frames():
    own=space();own.close()
    for operation in (
        lambda:own.keep('synthetic-closed-marker',code_authority=OWNER),
        lambda:own.deliberate(['wait'],{'cognition':['synthetic-closed-marker']},evidence=[],code_authority=OWNER)):
        try:operation()
        except PrivateThinkingError as error:
            trace=error.__traceback__
            while trace:
                if trace.tb_frame.f_code.co_filename.endswith('/private_thinking.py'):
                    assert 'synthetic-closed-marker' not in repr(trace.tb_frame.f_locals)
                trace=trace.tb_next
            assert error.__context__ is None
        else:raise AssertionError('closed space must reject')
