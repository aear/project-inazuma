"""V1 capability checks; no historical or live transfer gain is claimed."""
from expressive_variation import compare_variation


def main():
    result = compare_variation(modality='image', before={'colour': 'orange'},
                               after={'colour': 'blue'}, source='benchmark fixture')
    unchanged = compare_variation(modality='text', before={'wording': 'same'},
                                  after={'wording': 'same'}, source='benchmark fixture')
    signals = {
        'reported_difference_preserved': result['capture']['content']['changed_features'] == ['colour'],
        'meaning_unresolved': result['capture']['meaning'] is None,
        'unchanged_abstains': not unchanged['optional_trials'],
        'no_automatic_action': not result['automatic_execution'],
        'transfer_unmeasured': result['transfer_gain'] == 'unavailable',
    }
    print({'V1': signals, 'historical_comparison': 'unavailable: new optional tool',
           'live_artifact_quality_and_cross_domain_transfer': 'unavailable'})
    return 0 if all(signals.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
