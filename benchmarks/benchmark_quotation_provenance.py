"""V1 surface provenance; intentional quotation remains unmeasured."""
from quotation_provenance import trace_gloss_quotations


def measure():
    result = trace_gloss_quotations(["'term'", "don't", 'term', '“term”'], ['a','b','c','d'], {'a':'text_vocab_links'})
    first, second = result['tokens']
    return {
        'mapped_typography_identified': first['punctuation_origin'] == 'retained_mapping',
        'surface_retained': first['surface'] == "'term'",
        'apostrophe_not_quotation': [r['token_index'] for r in result['tokens']] == [0,3],
        'unknown_origin_retained': second['punctuation_origin'] == 'unresolved',
        'intent_not_assumed': not first['deliberate_quotation_established'],
    }


if __name__ == '__main__':
    result = measure()
    print({'V1': result, 'historical_comparison': 'unavailable: new provenance annotation',
           'live_intent_understanding': 'unavailable'})
    raise SystemExit(0 if all(result.values()) else 1)
