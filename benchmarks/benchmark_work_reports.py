"""Pinned generic-note baseline versus connected private reporting/diagnostics."""
from pathlib import Path
import sys
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
from time import perf_counter
from historical_source import historical_module
from instruction_authority import LOCAL_VOLUNTARY_CODE_AUTHORITY, seal_code_command

BASELINE = 'a31f1323ee4bb589d0ca0bf02aea87201b69cd21'


def evaluate(version='V2'):
    import personal_tool_runtime
    runtime = historical_module('personal_tool_runtime.py', BASELINE) if version == 'V1' else personal_tool_runtime
    signals = {}
    start = perf_counter()
    with tempfile.TemporaryDirectory(prefix='ina-report-fixture-') as directory:
        root = Path(directory)
        cfg = {'ina_hdd_writable_path': str(root / 'private')}
        def call(command):
            command = seal_code_command({**command, 'source': 'ina_voluntary_choice'}, LOCAL_VOLUNTARY_CODE_AUTHORITY)
            return runtime.execute_personal_tool_command(command, child='Fixture', project_root=root, config=cfg)['value']
        note = call({'action':'write_note', 'title':'observations', 'text':'⟡ native observation'})
        signals['native_authored_note_retained'] = Path(note['path']).read_text() == '⟡ native observation'
        available = 'work_report_write' in runtime.capability_catalog()['commands']
        signals['structured_reports_discoverable'] = available
        for kind in ('drawing','music','experiment','capability','bug'):
            if not available:
                signals[kind + '_report_roundtrip'] = False
                continue
            value = call({'action':'work_report_write', 'report':{
                'kind':kind, 'title':'⟡ chosen report', 'observations':['What I observed'],
                'uncertainties':['Outcome not yet independently checked'],
                'evidence':[{'reference':'asset:version-1', 'origin':'fixture', 'independence_group':'one'}],
                'measurements':[{'name':'case result', 'value':'unavailable', 'method':'not measured'}],
                'expected':'Expected result', 'actual':'Observed result', 'reproduction_steps':['Repeat chosen action']}})
            stored = call({'action':'work_report_read', 'report_id':value['report']['report_id']})['report']
            signals[kind + '_report_roundtrip'] = (stored == value['report'] and not stored['delivered']
                and not stored['promotion_authorized'] and stored['measurements'][0]['value'] == 'unavailable')
        if available:
            first = call({'action':'work_report_write','report':{'kind':'drawing','title':'Same title',
                'observations':['first'], 'uncertainties':[]}})
            second = call({'action':'work_report_write','report':{'kind':'drawing','title':'Same title',
                'observations':['revised'], 'uncertainties':[], 'previous_report':first['report']['report_id']}})
            signals['revision_retains_original'] = first['path'] != second['path'] and 'first' in Path(first['path']).read_text()
            code = 'def sample(items=[]):\n    try:\n        return items\n    except:\n        return []\n'
            (root/'fixture.py').write_text(code)
            findings = call({'action':'bug_file_inspect','relative_path':'fixture.py'})
            signals['connected_static_file_candidates'] = {r['kind'] for r in findings['findings']} == {'mutable_default','bare_except'}
            signals['source_hash_and_no_execution'] = bool(findings['sha256']) and not findings['executed']
            malformed = call({'action':'bug_source_inspect','code':'def broken('})
            signals['syntax_failure_is_reported'] = malformed['findings'][0]['kind'] == 'syntax_error'
            clean = call({'action':'bug_source_inspect','code':'def one(x=None):\n    return x\n'})
            signals['safe_minimal_pair_no_flag'] = not clean['findings'] and 'correctness' in clean['assurance']
            (root/'large.py').write_text('x'*70000)
            refused = False
            try:
                call({'action':'bug_file_inspect','relative_path':'large.py'})
            except ValueError:
                refused = True
            signals['file_budget_rejected_before_parse'] = refused
            from instruction_authority import InstructionAuthorityError
            refused = False
            try:
                seal_code_command({'action':'work_report_write','source':'ina_voluntary_choice'},None)
            except InstructionAuthorityError:
                refused = True
            signals['self_label_cannot_authorize_report'] = refused
        else:
            for name in ('revision_retains_original','connected_static_file_candidates','source_hash_and_no_execution',
                         'syntax_failure_is_reported','safe_minimal_pair_no_flag','file_budget_rejected_before_parse',
                         'self_label_cannot_authorize_report'):
                signals[name] = False
    return {'baseline_revision':BASELINE, 'signals':signals, 'elapsed_seconds':perf_counter()-start,
            'live_creative_quality_and_bug_detection_recall':'unavailable; synthetic structural fixtures only'}


def main():
    result = {v:evaluate(v) for v in ('V1','V2')}
    print(json.dumps(result,indent=2))
    return 0 if all(result['V2']['signals'].values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
