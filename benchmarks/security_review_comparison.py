"""V1 at 2202f60 versus V2, run only in a memory-inaccessible fixture sandbox.

Materialize the historical files with git archive before running; do not pass a
live store as the historical directory. All Git work uses temporary repositories.
"""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(root, label):
    git_module = load(label + '_git', root / 'harness_git.py')
    lab = load(label + '_lab', root / 'authorized_security_lab.py')
    kernel = load(label + '_kernel', root / 'kernel_lab.py')
    cognition = load(label + '_cognition', root / 'experience_cognition.py')
    lexical = load(label + '_lexical', root / 'lexical_reference.py')
    projects = load(label + '_projects', root / 'external_project_registry.py')
    signals = {}
    with tempfile.TemporaryDirectory(prefix='ina_review_bench_') as folder:
        work = Path(folder)
        def git(*args):
            return subprocess.run(['git', *args], cwd=work, check=True, capture_output=True)
        git('init', '-b', 'main')
        git('config', 'user.name', 'Fixture')
        git('config', 'user.email', 'fixture@example.test')
        (work / 'a').write_text('old\n')
        git('add', 'a'); git('commit', '-m', 'initial')
        git('remote', 'add', 'origin', 'https://github.com/example/repo.git')
        client = git_module.HarnessGit(work)
        (work / 'a').write_text('x' * (256 * 1024 + 1) + '\nTAIL\n')
        try:
            client.prepare_commit(['a'], 'oversized')
            signals['oversized_review_rejected'] = False
        except git_module.HarnessGitError:
            signals['oversized_review_rejected'] = True
        git('remote', 'set-url', '--push', 'origin', 'https://attacker.example/repo.git')
        try:
            client._github_remote('origin')
            signals['malicious_push_destination_rejected'] = False
        except git_module.HarnessGitError:
            signals['malicious_push_destination_rejected'] = True
        git('remote', 'set-url', '--push', 'origin', 'https://github.com/example/repo.git')
        calls = []
        original = client._run
        def counted(args, **kwargs):
            if args and args[0] == 'status': calls.append(True)
            return original(args, **kwargs)
        client._run = counted
        for _ in range(8): client.status()
        signals['workspace_scans_for_eight_polls'] = len(calls)
    draft = lab.create_engagement(platform='hack_the_box_labs', target='10.10.10.10',
        allowed_actions=['tcp_connect'], expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        consent_text='synthetic consent; no real target assigned', rules_url='https://www.hackthebox.com', ai_policy='ai_native')
    draft['target'] = '127.0.0.1'
    try:
        lab.authorize_action(draft, target='127.0.0.1', action='tcp_connect')
        signals['tampered_lab_record_rejected'] = False
    except lab.LabAuthorizationError:
        signals['tampered_lab_record_rejected'] = True
    signals['caller_signature_claim_cannot_authorize'] = not kernel.source_manifest('6.18.54', sha256='a'*64, signature_verified=True)['build_authorized']
    signals['shared_source_remains_uncertain'] = cognition.assess_uncertainty({
        'candidate_answer':'hypothesis', 'evidence':{'causal':['a'], 'sensory':['b']},
        'evidence_origins':{'a':'same','b':'same'},
    })['status'] == 'uncertain'
    def bounded():
        for index in range(64): yield {'state_id':str(index)}
        raise RuntimeError('overconsumed')
    try:
        cognition.gate_transient_state(bounded(), limit=64)
        signals['iterator_budget_enforced'] = True
    except RuntimeError:
        signals['iterator_budget_enforced'] = False
    class Session:
        def get(self, url, headers=None):
            return {'url':url, 'body':json.dumps({'en':[{'definitions':[{'definition':'sense'}]*12}]*12}).encode()}
    signals['definition_budget_enforced'] = len(lexical.lookup_definition('word', session=Session())['definitions']) <= 48
    with tempfile.TemporaryDirectory(prefix='ina_project_bench_') as folder:
        work = Path(folder)
        ina, external = work / 'ina', work / 'external'
        ina.mkdir(); external.mkdir()
        projects.register_project(name='Fixture', path=external, inazuma_root=ina)
        signals['revocable_source_reader'] = False
        if hasattr(projects, 'read_project_source'):
            try:
                projects.read_project_source('Fixture', inazuma_root=ina)
            except projects.ProjectRegistryError:
                signals['revocable_source_reader'] = True
    return signals


def main():
    historical = Path(sys.argv[1])
    result = {'baseline_revision':'2202f60', 'V1':measure(historical, 'review_v1'),
              'V2':measure(Path(__file__).resolve().parents[1], 'review_v2'),
              'unavailable':['live_provider_commit_message_quality','live_network_containment',
                             'Ina_learning_or_intelligence_gain','real_kernel_boot','visual_GUI_review'],
              'evidence_scope':'implementation fixtures only; separate signals, no aggregate readiness score'}
    print(json.dumps(result, indent=2))
    return 0 if all(value is True or key == 'workspace_scans_for_eight_polls' and value == 1 for key,value in result['V2'].items()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
