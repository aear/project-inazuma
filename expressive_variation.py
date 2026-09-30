"""Explicit, bounded comparisons; variation is evidence, not a quality reward."""
from __future__ import annotations

from emergence_capture import capture_emergence, propose_interpretation


FEATURES = {
    'image': {'colour', 'shape', 'spacing', 'repetition'},
    'text': {'wording', 'construction', 'spacing', 'repetition'},
    'sound': {'timbre', 'pitch', 'spacing', 'repetition'},
}
TRIALS = {
    'image': 'Keep the motif; optionally vary one colour or spacing choice.',
    'text': 'Keep the intended referents and uncertainty; optionally try a different construction.',
    'sound': 'Keep the motif; optionally vary one timbre or rhythmic spacing choice.',
}


def compare_variation(*, modality, before, after, source, references=()):
    """Compare caller-observed features without reading artifacts or private history.

    Source is a provenance claim, not authenticated corroboration. No inference
    about intention, emotional meaning, improvement, or successful transfer follows.
    """
    if modality not in FEATURES:
        raise ValueError('supported modalities: image, text, sound')
    if not isinstance(source, str) or not source.strip() or len(source) > 160:
        raise ValueError('an explicit bounded observation source is required')
    snapshots = []
    for values in (before, after):
        if not isinstance(values, dict) or not values or not set(values) <= FEATURES[modality]:
            raise ValueError('supply nonempty supported feature mappings')
        if any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in values.values()):
            raise ValueError('feature observations must be bounded nonempty strings')
        snapshots.append(dict(values))
    old, new = snapshots
    changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
    unresolved = sorted(old.keys() ^ new.keys())
    capture = capture_emergence(
        {'before': old, 'after': new, 'changed_features': changed,
         'unpaired_features': unresolved, 'observation_source': source},
        modality=modality, source='explicit_variation_comparison', context_references=references)
    hypothesis = propose_interpretation(
        capture, 'Reported variation may support a one-variable comparison; its cause remains unresolved.',
        confidence=0.0) if changed else None
    return {
        'schema': 'ina.expressive_variation/V1', 'capture': capture,
        'hypothesis': hypothesis,
        'cause_alternatives': ['emotional_expression', 'novelty_seeking', 'aesthetic_preference',
                               'choice', 'brush_or_tool_state', 'example_or_default', 'unknown'],
        'cause_status': 'unresolved; alternatives are neither confirmed nor excluded and may coexist',
        'self_account': 'optional evidence; no explanation required',
        'optional_trials': [
            {'modality': medium, 'proposal': proposal, 'attempt_budget': 1,
             'assessment': 'Compare both artifacts for fidelity, expressive usefulness and unwanted changes separately.',
             'status': 'untested', 'may_decline': True}
            for medium, proposal in TRIALS.items()
        ] if changed else [],
        'quality_gain': 'unavailable', 'transfer_gain': 'unavailable',
        'instructions_authorized': False, 'automatic_execution': False,
        'automatic_memory_write': False,
    }
