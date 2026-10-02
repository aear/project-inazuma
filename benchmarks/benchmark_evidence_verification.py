"""V2/V3 comparison with pinned historical attribution source."""
import importlib.util
import json
import sys


def measure(module):
    rows = [{'evidence_id': name, 'evidence_type': kind, 'sha256': module.hash_evidence(name.encode()),
             'origin': name, 'independence_group': name, 'supports_subject': 'candidate',
             'chain_of_custody_recorded': True}
            for name, kind in [('log', 'owned_system_log'), ('provider', 'provider_verified_account')]]
    claimed = module.assess_attribution(rows, proposed_subject='candidate')
    checked = tampered = None
    if hasattr(module, 'verify_evidence_bytes'):
        verified = [module.verify_evidence_bytes(row, row['evidence_id'].encode()) for row in rows]
        checked = module.assess_attribution(verified, proposed_subject='candidate')
        altered = {**verified[0], 'supports_subject': 'someone else'}
        tampered = module.validate_evidence(altered)
    return {
        'declared_custody_not_proof': not claimed['chain_of_custody_complete'],
        'declarations_cannot_authorize_identity': not claimed['identity_claim_authorized'],
        'checked_bytes_available_for_review': bool(checked and checked['artifact_bytes_verified']),
        'checked_bytes_not_identity': bool(checked and not checked['identity_claim_authorized']),
        'metadata_tampering_rejected': bool(tampered and not tampered['artifact_bytes_verified']),
        'no_automatic_submission': not claimed['automatic_submission_authorized'],
    }


if __name__ == '__main__':
    import threat_attribution as current
    spec = importlib.util.spec_from_file_location('historical_attribution', sys.argv[1])
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    signals = measure(current)
    print(json.dumps({'V2_pinned_ee1eb68': measure(old), 'V3': signals,
                      'end_to_end_custody_and_live_attribution': 'unavailable'}))
    raise SystemExit(0 if all(signals.values()) else 1)
