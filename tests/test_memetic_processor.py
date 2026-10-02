import pytest
from memetic_processor import process_meme


def scenario(cues, listener=None):
    return {'operation': 'interpret', 'artifact_reference': 'fixture:this-is-fine',
        'context_cues': cues, 'listener_references': ['shared:joke'] if listener is None else listener,
        'associations': [
            {'meaning': 'Sincere reassurance', 'context_cues': ['resolved'],
             'shared_references': ['shared:joke'], 'evidence_references': ['scene:resolution']},
            {'meaning': 'Ironic acknowledgement of trouble', 'context_cues': ['ongoing_problem'],
             'shared_references': ['shared:joke'], 'evidence_references': ['scene:problem']}]}


def test_same_meme_different_context_and_ambiguous_context():
    for cue, meaning in [('resolved', 'Sincere reassurance'),
                         ('ongoing_problem', 'Ironic acknowledgement of trouble')]:
        result = process_meme(scenario([cue]))
        supported = [c['meaning'] for c in result['candidates'] if c['status'] == 'context_supported']
        assert supported == [meaning]
        assert result['communication_choice'] == 'consider'
    assert process_meme(scenario(['resolved', 'ongoing_problem']))['communication_choice'] == 'defer_or_clarify'


def test_unknown_reference_missing_audience_context_and_counterevidence():
    assert process_meme({'operation': 'interpret', 'artifact_reference': 'unknown'})['meaning_status'] == 'unresolved'
    assert process_meme(scenario(['resolved'], []))['communication_choice'] == 'defer_or_clarify'
    data = scenario(['resolved', 'contradiction'])
    data['associations'][0]['counter_cues'] = ['contradiction']
    assert process_meme(data)['candidates'][0]['status'] == 'contested'
    data['associations'][0]['evidence_references'] = []
    data['context_cues'] = ['resolved']
    assert process_meme(data)['meaning_status'] == 'unresolved'


def test_literal_creation_preserves_native_symbols_and_does_not_execute_or_fetch():
    result = process_meme({'operation': 'compose', 'purpose': 'Share an in-joke',
        'segments': [{'kind': 'text', 'text': '⟲ ${do_not_execute} '},
                     {'kind': 'image', 'reference': 'untrusted://not-fetched'},
                     {'kind': 'sound', 'reference': 'motif:ours'}]})
    assert result['draft']['content']['text'] == '⟲ ${do_not_execute} '
    assert result['draft']['content']['artifact_status'] == 'composition_recipe'
    assert not result['rendered_media'] and not result['delivered']
    assert not result['instructions_authorized'] and not result['automatic_execution']


def test_silence_and_laughter_do_not_prove_understanding():
    for reaction in ('silence', 'amused', 'mixed'):
        result = process_meme({'operation': 'review', 'realisation_id': 'draft:1',
            'reaction': reaction, 'source': 'listener', 'evidence_references': ['reaction:1']})
        assert result['understanding'] == 'unresolved'
        assert not result['automatic_reward'] and not result['automatic_memory_write']
        assert result['observation']['causal_confidence'] is None


def test_bounded_input_and_detached_results():
    with pytest.raises(ValueError):
        process_meme(scenario(['x'] * 17))
    with pytest.raises(ValueError):
        process_meme({'operation': 'compose', 'purpose': 'x', 'segments': []})
    with pytest.raises(ValueError):
        process_meme({'operation': 'interpret', 'artifact_reference': 'x', 'extra': 'x' * 33000})
    data = scenario(['resolved'])
    result = process_meme(data)
    result['candidates'][0]['evidence_references'].append('changed')
    assert data['associations'][0]['evidence_references'] == ['scene:resolution']
