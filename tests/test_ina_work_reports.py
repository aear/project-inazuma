import pytest
from ina_work_reports import inspect_python_source, prepare_work_report, inspect_code_file


def test_connected_reporting_and_diagnostics():
    from benchmarks.benchmark_work_reports import evaluate
    assert all(evaluate('V2')['signals'].values())


def test_source_is_never_executed_and_repeated_names_are_scope_local():
    result = inspect_python_source("raise RuntimeError('must not execute')\nclass A:\n def same(self): pass\nclass B:\n def same(self): pass\n")
    assert not result['executed']
    assert not result['findings']
    duplicate = inspect_python_source('def a(): pass\ndef a(): pass\n')
    assert duplicate['findings'][0]['kind'] == 'repeated_definition'


def test_source_limits_and_partial_traversal_are_disclosed():
    with pytest.raises(ValueError):
        inspect_python_source('a'*65537)
    with pytest.raises(ValueError):
        inspect_python_source('⟡'*30000)
    assert not inspect_python_source('x=1\n'*2000)['complete']
    many = inspect_python_source('try:\n pass\nexcept:\n pass\n'*40)
    assert len(many['findings']) == 32
    assert not many['complete']


def test_private_paths_and_symlinks_are_rejected(tmp_path):
    from ina_desktop.files import VirtualFileSystem, configured_drives
    fs = VirtualFileSystem(configured_drives({}, 'Fixture', project_root=tmp_path))
    for name in ('../escape.py','/absolute.py','.private/secret.py','AI_Children/Ina/memory/a.py','venv/a.py','data.json'):
        with pytest.raises(ValueError): inspect_code_file(fs,name)
    (tmp_path/'outside.py').write_text('pass')
    (tmp_path/'link.py').symlink_to(tmp_path/'outside.py')
    with pytest.raises(PermissionError): inspect_code_file(fs,'link.py')


def test_read_budget_preserves_default_and_can_be_lowered(tmp_path):
    from ina_desktop.files import VirtualFileSystem, configured_drives
    fs = VirtualFileSystem(configured_drives({}, 'Fixture', project_root=tmp_path))
    (tmp_path/'large.py').write_text('x'*70000)
    assert len(fs.read('code','large.py')) == 70000
    with pytest.raises(ValueError): inspect_code_file(fs,'large.py')
    with pytest.raises(ValueError): fs.read('code','large.py',max_bytes=0)


def test_reports_preserve_unknowns_and_require_explicit_uncertainty():
    payload = {'kind':'capability','title':'Unknown','observations':['⟡'], 'uncertainties':['not measured']}
    report = prepare_work_report(payload)
    assert report['observations'] == ['⟡']
    assert report['measurements'] == []
    assert not report['promotion_authorized']
    del payload['uncertainties']
    with pytest.raises(ValueError): prepare_work_report(payload)


def test_readback_cannot_select_arbitrary_file(tmp_path):
    from ina_work_reports import read_work_report
    from ina_desktop.files import VirtualFileSystem, configured_drives
    fs = VirtualFileSystem(configured_drives({}, 'Fixture', project_root=tmp_path))
    with pytest.raises(ValueError): read_work_report(fs,'../../private')


def test_new_actions_need_real_authority():
    from instruction_authority import seal_code_command, InstructionAuthorityError
    for action in ('work_report_write','work_report_read','bug_source_inspect','bug_file_inspect'):
        with pytest.raises(InstructionAuthorityError):
            seal_code_command({'action':action,'source':'ina_voluntary_choice'},None)


def test_benchmark_registry():
    from module_benchmarks import benchmark_module
    rows = benchmark_module('work_reports')
    assert [r.version for r in rows] == ['V1','V2']
    assert rows[1].correct == rows[1].total
