"""V1 representation capability, not empirical predictive accuracy."""
from predictive_landscape import explore_landscape, revise_landscape


def measure():
    data = {'signal': {'content': 'unexplained horizon', 'subject': None}, 'time_unit': 'days',
        'start': 's', 'nodes': [
            {'id': 's', 'description': 'source', 'position': [0, 0, 0]},
            {'id': 'a', 'description': 'outcome A', 'position': [1, 1, 1], 'countercheck': 'A absent'},
            {'id': 'b', 'description': 'attention only', 'position': [2, -1, 1], 'kind': 'attention_horizon'}],
        'links': [{'from': 's', 'to': 'a', 'condition': 'if A prerequisites hold'},
                  {'from': 's', 'to': 'b', 'condition': 'if relevant'}], 'investigate_node': 'a'}
    result = explore_landscape(data)
    model = revise_landscape(data)
    revision = revise_landscape({'links': []}, model)
    return {
        'source_not_reinterpreted': result['capture']['content']['subject'] is None,
        'branches_preserved': [b['path'] for b in result['branches']] == [['s', 'a'], ['s', 'b']],
        'horizon_not_forecast': result['branches'][1]['status'] == 'attention_or_unresolved',
        'conditional_prediction_has_countercheck': result['branches'][0]['countercheck'] == 'A absent',
        'no_geometric_probability': all(b['probability'] is None for b in result['branches']),
        'bounded_optional_inquiry': result['inquiry']['remaining_continuations'] == 0,
        'no_memory_or_execution': not result['writes_memory'] and not result['automatic_execution'],
        'revision_preserves_original': revision['landscape']['capture'] == model['landscape']['capture'],
        'revision_does_not_repeat_inquiry': revision['landscape']['inquiry'] is None,
    }


if __name__ == '__main__':
    results = measure()
    print({'V1': results, 'historical_comparison': 'unavailable: new representation',
           'live_predictive_accuracy': 'unavailable'})
    raise SystemExit(0 if all(results.values()) else 1)
