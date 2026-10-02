"""Pinned V2/V3 evidence handling and discriminating-check comparison."""
import importlib.util
import json
import sys


def measure(module):
    event = {'candidate_answer': 'candidate', 'evidence': {'causal': ['a'], 'sensory': ['b']},
             'evidence_origins': {'a': 'first', 'b': 'second'}}
    supported = module.assess_uncertainty(event)
    opposing = module.assess_uncertainty({**event, 'counterevidence_references': ['opposing']})
    empty = module.assess_uncertainty({'candidate_answer': 'guess', 'evidence': {'causal': [None]}})
    shared = module.assess_uncertainty({**event, 'evidence_origins': {'a': 'same', 'b': 'same'}})
    check = None
    incomplete = None
    if hasattr(module, 'compare_hypotheses'):
        alternatives = [{'id': 'a', 'claim': 'A'}, {'id': 'b', 'claim': 'B'}]
        check = module.compare_hypotheses({'hypotheses': alternatives, 'observation_candidates': [
            {'id': 'repeat', 'question': 'Repeat?', 'expected_outcomes': {'a': 'same', 'b': 'same'}},
            {'id': 'distinguish', 'question': 'Independent check?', 'expected_outcomes': {'a': 'yes', 'b': 'no'}}]})
        incomplete = module.compare_hypotheses({'hypotheses': alternatives, 'observation_candidates': [
            {'id': 'gap', 'question': 'Check?', 'expected_outcomes': {'a': 'yes'}}]})
    return {
        'empty_reference_abstains': empty['status'] == 'unknown',
        'counterevidence_retained_without_score': opposing['status'] == 'uncertain' and opposing['conflict_retained'],
        'supported_case_preserved': supported['status'] == 'known',
        'shared_source_still_uncertain': shared['status'] == 'uncertain',
        'discriminating_check_selected': bool(check and check['suggested_check'] == 'distinguish'),
        'missing_prediction_abstains': bool(incomplete and incomplete['suggested_check'] is None),
        'no_execution_authority': bool(check and not check['automatic_execution']),
    }


if __name__ == '__main__':
    import experience_cognition as current
    spec = importlib.util.spec_from_file_location('historical_cognition', sys.argv[1])
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    signals = measure(current)
    print(json.dumps({'V2_pinned_ee1eb68': measure(old), 'V3': signals,
                      'live_reasoning_gain_and_calibration': 'unavailable'}))
    raise SystemExit(0 if all(signals.values()) else 1)
