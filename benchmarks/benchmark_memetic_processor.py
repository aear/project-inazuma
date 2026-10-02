"""V1 deterministic capability signals, not measured human meme comprehension."""
from memetic_processor import process_meme


def measure():
    base = {'operation': 'interpret', 'artifact_reference': 'fixture:shared-phrase',
        'listener_references': ['shared:1'], 'associations': [
            {'meaning': 'affection', 'context_cues': ['friendly'], 'shared_references': ['shared:1'],
             'evidence_references': ['scene:1']},
            {'meaning': 'criticism', 'context_cues': ['conflict'], 'shared_references': ['shared:1'],
             'evidence_references': ['scene:2']}]}
    friendly = process_meme({**base, 'context_cues': ['friendly']})
    conflict = process_meme({**base, 'context_cues': ['conflict']})
    mixed = process_meme({**base, 'context_cues': ['friendly', 'conflict']})
    unknown = process_meme({**base, 'context_cues': []})
    listener = process_meme({**base, 'context_cues': ['friendly'], 'listener_references': []})
    draft = process_meme({'operation': 'compose', 'purpose': 'Share motif',
        'segments': [{'kind': 'text', 'text': '⟲'}, {'kind': 'sound', 'reference': 'motif:1'}]})
    review = process_meme({'operation': 'review', 'realisation_id': draft['draft']['realisation_id'],
                          'reaction': 'silence', 'source': 'observation'})
    return {
        'context_changes_candidate': friendly['candidates'][0]['status'] == 'context_supported'
            and conflict['candidates'][1]['status'] == 'context_supported',
        'ambiguity_preserved': mixed['meaning_status'] == 'unresolved',
        'unknown_abstains': unknown['communication_choice'] == 'defer_or_clarify',
        'listener_gap_preserved': listener['communication_choice'] == 'defer_or_clarify',
        'multimodal_recipe_preserved': draft['draft']['content']['segments'][1]['reference'] == 'motif:1',
        'silence_not_failure': review['understanding'] == 'unresolved' and not review['automatic_reward'],
        'no_automatic_delivery_or_authority': not draft['delivered'] and not draft['instructions_authorized'],
    }


if __name__ == '__main__':
    signals = measure()
    print({'V1': signals, 'historical_comparison': 'unavailable: new communication component',
           'live_comprehension_humour_quality_learning': 'unavailable'})
    raise SystemExit(0 if all(signals.values()) else 1)
