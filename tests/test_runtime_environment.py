from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import runtime_environment as env


def test_project_python_preserves_venv_symlink(tmp_path):
    p=tmp_path/'venv/bin/python';p.parent.mkdir(parents=True);p.symlink_to(sys.executable)
    assert env.project_python(tmp_path)==str(p)


def test_missing_venv_uses_current_interpreter(tmp_path):
    assert env.project_python(tmp_path)==sys.executable


def test_real_probe_imports_stdlib_without_ina():
    assert env.check_environment(sys.executable,('json','math'))['ok']


def test_real_probe_reports_missing_module():
    r=env.check_environment(sys.executable,('ina_fixture_nonexistent_module',))
    assert not r['ok'] and r['errors']==[{'module':'ina_fixture_nonexistent_module','error':'ModuleNotFoundError'}]


def test_timeout_is_actionable(monkeypatch):
    def fail(*a,**k):raise subprocess.TimeoutExpired('fixture',1)
    monkeypatch.setattr(env.subprocess,'run',fail)
    assert env.check_environment(sys.executable)['reason']=='dependency_probe_timeout'


def test_native_probe_failure_is_not_ready(monkeypatch):
    monkeypatch.setattr(env.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=-11))
    assert env.check_environment(sys.executable)['reason']=='interpreter_probe_failed'


def test_gui_selects_environment_before_imports(monkeypatch,tmp_path):
    target=tmp_path/'venv/bin/python';target.parent.mkdir(parents=True);target.touch()
    calls=[]
    class Reexec(Exception):pass
    def execute(path,args):calls.append((path,args));raise Reexec()
    monkeypatch.setattr(env.os,'execv',execute)
    try:env.prepare_gui(tmp_path)
    except Reexec:pass
    assert calls[0][0]==str(target) and calls[0][1][1]==str(tmp_path/'GUI.py')


def test_gui_stops_before_runtime_when_probe_fails(monkeypatch,tmp_path):
    monkeypatch.setattr(env,'project_python',lambda root:sys.executable)
    monkeypatch.setattr(env,'check_environment',lambda *a:{'ok':False,'errors':[{'module':'discord'}]})
    try:env.prepare_gui(tmp_path)
    except SystemExit as exc:assert 'discord' in str(exc) and 'requirements.txt' in str(exc)
    else:raise AssertionError('GUI must stop')


def test_dependency_probe_limits_numeric_pools(monkeypatch):
    calls=[]
    def run(*a,**k):
        calls.append(k)
        return SimpleNamespace(returncode=0,stdout='{"ok": true}')
    monkeypatch.setattr(env.subprocess,'run',run)
    assert env.check_environment(sys.executable)['ok']
    assert all(calls[0]['env'][n]=='1' for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))


def test_core_candidate_not_selected_before_review(tmp_path):
    p=tmp_path/'.runtime/core315/bin/python';p.parent.mkdir(parents=True);p.touch()
    assert env.project_python(tmp_path)==sys.executable


def test_validated_core_and_bridge_choose_distinct_environments(tmp_path):
    core=tmp_path/'.runtime/core315';core.mkdir(parents=True);(core/'.ina-ready').touch()
    assert env.project_python(tmp_path)==str(core/'bin/python')
    assert env.project_python(tmp_path,service='world_server')==str(core/'bin/python')
    assert env.project_python(tmp_path,service='discord_bridge')==str(tmp_path/'venv/bin/python')


def test_missing_selected_interpreter_does_not_fall_back_to_another_python(tmp_path):
    core=tmp_path/'.runtime/core315';core.mkdir(parents=True);(core/'.ina-ready').touch()
    assert env.project_python(tmp_path)!=sys.executable
    assert not env.check_environment(env.project_python(tmp_path))['ok']


def test_core_probe_does_not_require_discord():
    assert 'discord' not in env.GUI_MODULES
    assert 'audioop' in env.GUI_MODULES and 'pydub' in env.GUI_MODULES
    assert 'discord' in env.DISCORD_MODULES


def test_versioned_split_interpreter_contract():
    from benchmarks.benchmark_python_runtime_split import assess
    result=assess('V2')
    assert result['correct']==result['total'],result


def test_python_children_keep_the_selected_environment(monkeypatch,tmp_path):
    monkeypatch.setattr(env,'project_python',lambda root:sys.executable)
    monkeypatch.setattr(env,'check_environment',lambda *a:{'ok':True})
    monkeypatch.setenv('PATH','/usr/bin')
    env.prepare_gui(tmp_path)
    r=subprocess.run(['python','-I','-c','import sys;print(sys.prefix)'],capture_output=True,text=True,timeout=10,check=True)
    assert r.stdout.strip()==sys.prefix


def test_python_environment_preserves_inputs_and_other_variables():
    base={'PATH':'/usr/bin','EXAMPLE':'preserve'}
    actual=env.python_environment('/fixture/venv/bin/python',base)
    assert actual['PATH'].split(env.os.pathsep)[0]=='/fixture/venv/bin'
    assert actual['EXAMPLE']=='preserve' and base=={'PATH':'/usr/bin','EXAMPLE':'preserve'}
