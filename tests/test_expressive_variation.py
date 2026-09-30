import pytest

from expressive_variation import compare_variation


def test_colour_is_not_meaning_or_improvement():
    before = {'colour': 'orange', 'shape': 'spiral'}
    result = compare_variation(modality='image', before=before,
                               after={'colour': 'blue', 'shape': 'spiral'}, source='human report')
    assert result['capture']['content']['changed_features'] == ['colour']
    assert result['capture']['meaning'] is None
    assert result['quality_gain'] == result['transfer_gain'] == 'unavailable'
    assert result['hypothesis']['confidence'] == 0
    assert {'emotional_expression', 'novelty_seeking'} <= set(result['cause_alternatives'])
    assert 'neither confirmed nor excluded' in result['cause_status']
    assert before['colour'] == 'orange'
    assert {trial['modality'] for trial in result['optional_trials']} == {'image', 'text', 'sound'}
    assert all(t['attempt_budget'] == 1 and t['may_decline'] for t in result['optional_trials'])
    assert not result['automatic_execution']


def test_missing_observation_is_not_variation():
    result = compare_variation(modality='sound', before={'pitch': 'low'},
                               after={'timbre': 'soft'}, source='fixture')
    assert not result['optional_trials']
    assert result['capture']['content']['unpaired_features'] == ['pitch', 'timbre']


def test_bounds_and_unrecognised_features():
    for values in ({'emotion': 'sad'}, {'colour': 'x' * 257}, {}):
        with pytest.raises(ValueError):
            compare_variation(modality='image', before=values, after={'colour': 'blue'}, source='fixture')
