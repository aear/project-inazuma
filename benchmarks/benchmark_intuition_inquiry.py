"""Bounded V2/V3 comparison; historical source must come from pinned Git."""
import importlib.util
import json
import sys


def measure(module):
    journey = module.begin_self_inquiry('What matters?', trigger_references=['event:1'], depth_budget=3)
    stopped = module.continue_self_inquiry(journey, choice='stop')
    terminal = False
    try:
        module.continue_self_inquiry(stopped, choice='deeper')
    except PermissionError:
        terminal = True
    first = module.continue_self_inquiry(journey, choice='deeper', hypotheses=[
        {'hypothesis': 'First impression', 'confidence': .8}])
    revision = module.continue_self_inquiry(first, choice='stop', hypotheses=[
        {'hypothesis': 'Revised impression', 'confidence': .3, 'counterevidence_references': ['opposing:1']}])
    kept = module.continue_self_inquiry(first, choice='stop')
    def refs():
        for _ in range(32):
            yield 'same'
        raise RuntimeError('reference budget overrun')
    bounded = True
    try:
        module.begin_self_inquiry('Bounded?', trigger_references=refs())
    except RuntimeError:
        bounded = False
    intuition = None
    if hasattr(module, 'begin_intuition_inquiry'):
        intuition = module.begin_intuition_inquiry('This matters', question='What changes?',
            countercheck='Look for a case where it does not help', trigger_references=['event:1'])
    finite = False
    try:
        module.continue_self_inquiry(journey, choice='stop', hypotheses=[
            {'hypothesis': 'Maybe', 'confidence': float('nan')}])
    except ValueError:
        finite = True
    return {
        'terminal_choice_enforced': terminal,
        'prior_candidate_preserved': bool(revision.get('hypothesis_history')),
        'counterevidence_preserved': revision['hypotheses'][0].get('counterevidence_references') == ['opposing:1'],
        'omitted_revision_preserves_hypothesis': kept['hypotheses'] == first['hypotheses'],
        'duplicate_reference_work_bounded': bounded,
        'candidates_not_authoritative': not revision['hypotheses'][0]['authoritative'],
        'nonfinite_confidence_rejected': finite,
        'intuition_has_countercheck_without_authority': bool(intuition
            and module.current_inquiry_request(intuition).get('countercheck')
            and not intuition['intuition']['grants_authority']
            and intuition['remaining_continuations'] == 0),
    }


def main():
    import self_inquiry_journey as current
    spec = importlib.util.spec_from_file_location('historical_inquiry', sys.argv[1])
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    signals = measure(current)
    print(json.dumps({'V2_pinned_0bc84c5': measure(historical), 'V3': signals,
                      'live_intuition_calibration_and_learning': 'unavailable'}))
    return 0 if all(signals.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
