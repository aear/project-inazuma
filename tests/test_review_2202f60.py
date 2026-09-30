"""Adversarial regressions use temporary repos, synthetic records and fake sockets."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import subprocess
import threading

import pytest

import authorized_security_lab as lab
from harness_git import HarnessGit, HarnessGitError, MAX_GIT_OUTPUT
from kernel_lab import KernelLabError, source_manifest, verify_kernel_signature
from codex_harness import AppServerClient, BoundedEvents


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def repo(root):
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.test')
    (root / 'a').write_text('old\n')
    git(root, 'add', 'a')
    git(root, 'commit', '-m', 'fixture')
    git(root, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
    return HarnessGit(root)


def test_changes_beyond_256k_never_receive_or_pass_review(tmp_path):
    client = repo(tmp_path)
    original = git(tmp_path, 'rev-parse', 'HEAD')
    (tmp_path / 'a').write_text('reviewed\n')
    receipt = client.prepare_commit(['a'], 'change')
    (tmp_path / 'a').write_text('x' * (MAX_GIT_OUTPUT + 1) + '\nHIDDEN CHANGE\n')
    with pytest.raises(HarnessGitError, match='256 KiB'):
        client.prepare_commit(['a'], 'oversized')
    with pytest.raises(HarnessGitError, match='256 KiB'):
        client.commit(receipt['receipt_id'], receipt['diff_sha256'])
    assert git(tmp_path, 'rev-parse', 'HEAD') == original


def test_malicious_pushurl_rejected_before_dns_or_transport(tmp_path, monkeypatch):
    client = repo(tmp_path)
    git(tmp_path, 'remote', 'set-url', '--push', 'origin', 'https://attacker.example/steal.git')
    def forbidden(*args, **kwargs):
        raise AssertionError('network must not run')
    monkeypatch.setattr('harness_git.resolve_public_addresses', forbidden)
    with pytest.raises(HarnessGitError, match='destinations'):
        client.push(remote='origin', expected_head=git(tmp_path, 'rev-parse', 'HEAD'), confirmation='PUSH REVIEWED COMMIT')


def test_url_rewrite_rejected_before_transport(tmp_path):
    client = repo(tmp_path)
    for kind in ('insteadOf', 'pushInsteadOf'):
        git(tmp_path, 'config', f'url.https://attacker.example/.{kind}', 'https://github.com/')
        with pytest.raises(HarnessGitError, match='rewrite'):
            client.fetch('origin')


def test_git_poll_cache_requires_event_or_manual_refresh(tmp_path, monkeypatch):
    client = repo(tmp_path)
    calls = []
    original = client._scan_status
    def scan():
        calls.append(True)
        return original()
    monkeypatch.setattr(client, '_scan_status', scan)
    first = client.status()
    for _ in range(12):
        assert client.status() == first
    assert len(calls) == 1
    (tmp_path / 'a').write_text('changed')
    assert client.status(refresh=True)['dirty'] is True
    assert len(calls) == 2


def engagement():
    consent = 'Fixture human verified account assignment and autonomous AI rules.'
    draft = lab.create_engagement(platform='hack_the_box_labs', target='10.10.10.10:443',
        allowed_actions=['tcp_connect'], expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        consent_text=consent, rules_url='https://www.hackthebox.com/legal/aup', ai_policy='ai_native')
    reviewed = lab.review_engagement(draft, consent_text=consent, reviewer='fixture reviewer', confirmed_digest=draft['engagement_sha256'])
    return draft, reviewed


def test_forged_and_tampered_lab_records_are_rejected():
    draft, approved = engagement()
    with pytest.raises(lab.LabAuthorizationError, match='unreviewed'):
        lab.authorize_action(draft, target=draft['target'], action='tcp_connect')
    for key, value in [('target', '127.0.0.1:443'), ('allowed_actions', ['persistence']), ('ai_policy', 'human_only')]:
        changed = dict(approved, **{key: value})
        changed['engagement_sha256'] = hashlib.sha256(lab._encoded(changed)).hexdigest()
        with pytest.raises(lab.LabAuthorizationError, match='tampered'):
            lab.authorize_action(changed, target=changed['target'], action='tcp_connect')


def test_execution_enforces_peer_port_and_single_attempt(monkeypatch):
    _, approved = engagement()
    connections = []
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): assert value == 3.0
        def connect(self, address): connections.append(address)
        def getpeername(self): return ('10.10.10.10', 443)
    monkeypatch.setattr(lab.socket, 'socket', lambda *args: FakeSocket())
    with pytest.raises(lab.LabAuthorizationError, match='outside'):
        lab.execute_tcp_connect(approved, target='10.10.10.11:443')
    assert connections == []
    assert lab.execute_tcp_connect(approved, target=approved['target'])['payload_sent'] is False
    assert connections == [('10.10.10.10', 443)]
    with pytest.raises(lab.LabAuthorizationError, match='consumed'):
        lab.execute_tcp_connect(approved, target=approved['target'])


def test_execution_rejects_peer_mismatch_and_expired_approval(monkeypatch):
    _, approved = engagement()
    with pytest.raises(lab.LabAuthorizationError, match='expired'):
        lab.authorize_action(approved, target=approved['target'], action='tcp_connect',
                             now=datetime.now(timezone.utc)+timedelta(days=1))
    class WrongPeer:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): pass
        def connect(self, address): pass
        def getpeername(self): return ('127.0.0.1', 443)
    monkeypatch.setattr(lab.socket, 'socket', lambda *args: WrongPeer())
    with pytest.raises(lab.LabAuthorizationError, match='peer'):
        lab.execute_tcp_connect(approved, target=approved['target'])


def test_commit_uses_reviewed_snapshot_and_preserves_unrelated_staging(tmp_path, monkeypatch):
    client = repo(tmp_path)
    (tmp_path / 'a').write_text('reviewed\n')
    (tmp_path / 'unrelated').write_text('keep staged\n')
    git(tmp_path, 'add', 'unrelated')
    review = client.prepare_commit(['a'], 'reviewed')
    original = client._run
    def race(args, **kwargs):
        if 'commit-tree' in args:
            (tmp_path / 'a').write_text('unreviewed race\n')
        return original(args, **kwargs)
    monkeypatch.setattr(client, '_run', race)
    client.commit(review['receipt_id'], review['diff_sha256'])
    assert git(tmp_path, 'show', 'HEAD:a') == 'reviewed'
    assert git(tmp_path, 'diff', '--cached', '--name-only') == 'unrelated'
    assert 'unreviewed race' in git(tmp_path, 'diff', '--', 'a')


def test_kernel_boolean_and_forged_receipt_never_grant_build():
    result = source_manifest('6.18.54', sha256='a'*64, signature_verified=True,
                             verification={'sha256': 'a'*64, 'seal': 'fake'})
    assert not result['signature_verified'] and not result['build_authorized']


def test_real_signature_valid_modified_source_and_wrong_signer(tmp_path):
    home = tmp_path / 'keys'
    home.mkdir(mode=0o700)
    def gpg(*args):
        return subprocess.run(['gpg', '--homedir', str(home), '--batch', '--pinentry-mode', 'loopback', '--passphrase', '', *args], capture_output=True, check=True)
    gpg('--quick-generate-key', 'Synthetic Fixture <fixture@example.test>', 'ed25519', 'sign', '1d')
    listing = gpg('--with-colons', '--list-keys').stdout.decode()
    fingerprint = next(line.split(':')[9] for line in listing.splitlines() if line.startswith('fpr:'))
    keyring = tmp_path / 'trusted.gpg'
    keyring.write_bytes(gpg('--export', fingerprint).stdout)
    source = tmp_path / 'synthetic.tar'
    source.write_bytes(b'not a kernel; signature verification fixture only')
    signature = tmp_path / 'synthetic.sig'
    gpg('--output', str(signature), '--detach-sign', str(source))
    receipt = verify_kernel_signature(source, signature, keyring=keyring, trusted_fingerprints=(fingerprint,))
    manifest = source_manifest('6.18.54', sha256=hashlib.sha256(source.read_bytes()).hexdigest(), verification=receipt)
    assert manifest['signature_verified'] is True
    with pytest.raises(KernelLabError, match='signer'):
        verify_kernel_signature(source, signature, keyring=keyring, trusted_fingerprints=('0'*40,))
    source.write_bytes(b'changed after signing')
    with pytest.raises(KernelLabError, match='verification failed'):
        verify_kernel_signature(source, signature, keyring=keyring, trusted_fingerprints=(fingerprint,))
    assert source_manifest('6.18.54', sha256=hashlib.sha256(source.read_bytes()).hexdigest(), verification=receipt)['build_authorized'] is False


def test_model_notification_cannot_be_spoofed_by_other_thread():
    client = object.__new__(AppServerClient)
    client.thread_id, client.turn_id = 'current', 'turn'
    client.active_model, client.requested_model = None, 'requested'
    client.events = BoundedEvents()
    client._handle_notification('thread/settings/updated', {'threadId':'other', 'threadSettings':{'model':'wrong'}})
    assert client.active_model is None
    client._handle_notification('thread/settings/updated', {'threadId':'current', 'threadSettings':{'model':'resolved'}})
    assert client.active_model == 'resolved' and client.requested_model == 'requested'
    client._handle_notification('model/rerouted', {'threadId':'current', 'turnId':'stale', 'toModel':'wrong'})
    assert client.active_model == 'resolved'
    client._handle_notification('model/rerouted', {'threadId':'current', 'turnId':'turn', 'toModel':'rerouted'})
    assert client.active_model == 'rerouted' and client.requested_model == 'requested'


def test_commit_message_is_editable_draft_and_never_commits(tmp_path):
    client = object.__new__(AppServerClient)
    client.git = repo(tmp_path)
    (tmp_path / 'a').write_text('new\n')
    head = git(tmp_path, 'rev-parse', 'HEAD')
    client._commit_generation_lock = threading.Lock()
    client._commit_generation = {}
    client.account = lambda **kwargs: {'type':'chatgpt'}
    def request(method, params):
        if method == 'thread/start':
            assert params['ephemeral'] is True and params['permissions'] == ':read-only'
            return {'thread': {'id':'draft'}, 'sandbox': {'type':'readOnly'}}
        if method == 'turn/start':
            client._handle_notification('item/completed', {'threadId':'draft', 'item':{'type':'agentMessage','text':json.dumps({'message':'Update fixture wording'})}})
            client._handle_notification('turn/completed', {'threadId':'draft','turn':{'status':'completed'}})
            return {'turn': {'id':'draft-turn'}}
        return {}
    client.request = request
    result = client.generate_commit_message(['a'])
    assert result['message'] == 'Update fixture wording' and result['committed'] is False
    assert git(tmp_path, 'rev-parse', 'HEAD') == head
