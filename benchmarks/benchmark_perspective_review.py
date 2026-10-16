"""Same bounded sequence for voluntary and instinct offers; no live memory needed."""
from pathlib import Path
import sys
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from time import perf_counter
from perspective_review import begin_review, review_step


def measure(variant):
    event = {'interpretation_reference': 'meaning:candidate-a', 'question': 'Which interpretation fits?',
             'countercheck': 'Look for a conflicting observation',
             'evidence_references': ['source:a'], 'contradiction': True}
    start = perf_counter()
    state = begin_review(event, variant=variant, chosen=True, max_steps=3)
    revised = review_step(state, choice='continue', observation_references=['source:b'], hypotheses=[
        {'hypothesis': 'meaning:candidate-b', 'confidence': .4,
         'evidence_references': ['source:b'], 'counterevidence_references': ['source:a']}])
    repeated = review_step(revised, choice='continue', observation_references=['source:b'])
    exits = {exit: review_step(state, choice=exit)['status'] == exit
             for exit in ('stop', 'defer', 'ask_evidence', 'remain_uncertain')}
    check = review_step(revised, choice='continue', countercheck='Check the opposite prediction')
    exhausted = review_step(check, choice='continue', observation_references=['source:c'])
    idle = begin_review({**event, 'contradiction': False}, variant=variant, chosen=False)
    blocked = False
    try:
        review_step(repeated, choice='continue', observation_references=['source:c'])
    except PermissionError:
        blocked = True
    return {'signals': {
        'one_step_yields_with_choices': revised['status'] == 'ready' and len(revised['trace']) == 1,
        'repeated_evidence_stops': repeated['status'] == 'no_progress',
        'terminal_state_cannot_resume': blocked,
        'distinct_countercheck_can_continue': check['status'] == 'ready',
        'explicit_budget_terminates': exhausted['status'] == 'budget_exhausted',
        'counterevidence_retained': revised['journey']['hypotheses'][0]['counterevidence_references'] == ['source:a'],
        'candidate_history_retained': len(revised['journey']['hypothesis_history']) == 1,
        'no_truth_or_memory_authority': not revised['journey']['hypotheses'][0]['authoritative'] and not revised['writes_memory'],
        'no_idle_start': idle is None, **{'exit_' + k: v for k, v in exits.items()}},
        'elapsed_seconds': perf_counter() - start,
        'fixture_steps': len(exhausted['trace']),
        'serialized_bytes': len(json.dumps(exhausted).encode()),
        'interpretation_quality_and_independent_evidence': 'unavailable: supplied fixture candidates, no learned revision model'}


def main():
    result = {v: measure(v) for v in ('private_tool', 'instinct')}
    print(json.dumps(result, indent=2))
    return 0 if all(all(r['signals'].values()) for r in result.values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
