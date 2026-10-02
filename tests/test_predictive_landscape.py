import pytest
from predictive_landscape import explore_landscape, revise_landscape


def test_revisions_retain_original_and_do_not_repeat_investigation():
    data = landscape()
    data['investigate_node'] = 'a'
    first = revise_landscape(data)
    second = revise_landscape({'links': []}, first)
    assert second['landscape']['capture'] == first['landscape']['capture']
    assert first['landscape']['links']
    assert second['landscape']['links'] == []
    assert second['landscape']['inquiry'] is None
    assert second['parent']['model_id'] == first['model_id']
    assert not second['scheduled_work']
    with pytest.raises(ValueError):
        revise_landscape({'signal': 'rewrite history'}, first)


def landscape():
    return {'signal': {'content': 'Something worth checking', 'felt_character': 'curiosity'},
        'time_unit': 'days', 'start': 'now', 'nodes': [
            {'id': 'now', 'description': 'Observed change', 'position': [0, 0, 0], 'kind': 'observation'},
            {'id': 'a', 'description': 'Change persists', 'position': [1, -1, 2], 'countercheck': 'Change disappears'},
            {'id': 'b', 'description': 'Change fades', 'position': [1, 1, -2], 'countercheck': 'Change persists'}],
        'links': [{'from': 'now', 'to': 'a', 'condition': 'If the cause continues', 'evidence_references': ['observation:1']},
                  {'from': 'now', 'to': 'b', 'condition': 'If the cause ends'}]}


def test_branching_is_conditional_and_geometry_not_confidence():
    result = explore_landscape(landscape())
    assert [b['path'] for b in result['branches']] == [['now', 'a'], ['now', 'b']]
    assert all(b['probability'] is None for b in result['branches'])
    assert result['branches'][1]['hypothetical_links']
    assert not result['geometry_is_evidence']
    assert result['inquiry'] is None


def test_unexplained_horizon_stays_unexplained():
    data = {'signal': {'content': '30 years', 'felt_character': 'warning', 'subject': None},
        'time_unit': 'years', 'start': 'horizon', 'nodes': [
            {'id': 'horizon', 'description': 'Attention horizon; subject unknown',
             'kind': 'attention_horizon', 'position': [30, 0, 0]}]}
    result = explore_landscape(data)
    assert result['capture']['content'] == data['signal']
    assert result['capture']['meaning'] is None
    assert result['branches'][0]['status'] == 'attention_or_unresolved'
    assert result['outcome_status'] == 'not_observed'
    data['signal']['subject'] = 'later association'
    assert result['capture']['content']['subject'] is None


def test_explicit_investigation_is_one_stage_and_retains_countercheck():
    data = landscape()
    data['investigate_node'] = 'a'
    result = explore_landscape(data)
    assert result['inquiry']['remaining_continuations'] == 0
    assert result['inquiry']['intuition']['countercheck'] == 'Change disappears'
    assert not result['automatic_execution'] and not result['writes_memory']


def test_invalid_graph_and_nonfinite_coordinates_are_rejected():
    data = landscape()
    data['links'][0]['to'] = 'now'
    with pytest.raises(ValueError):
        explore_landscape(data)
    data = landscape()
    data['nodes'][0]['position'][0] = float('nan')
    with pytest.raises(ValueError):
        explore_landscape(data)


def test_path_depth_budget_is_explicit_not_false_completion():
    data = {'signal': 'long chain', 'time_unit': 'steps', 'start': '0',
        'nodes': [{'id': str(i), 'description': 'possible step', 'position': [i, 0, 0]} for i in range(10)],
        'links': [{'from': str(i), 'to': str(i+1), 'condition': 'if next'} for i in range(9)]}
    result = explore_landscape(data)
    assert result['truncated'] and not result['branches'][0]['complete']
    assert result['branches'][0]['status'] == 'attention_or_unresolved'
