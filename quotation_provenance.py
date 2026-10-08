"""Keep quoted surface forms distinct from claims about communicative intent."""
from itertools import islice


def trace_gloss_quotations(tokens, symbols, sources):
    """Inspect at most 64 aligned gloss tokens without stripping punctuation.

    Only matching outer marks are recognised. Apostrophes inside words and
    quotation spans split over tokens are not classified by this small tracer.
    """
    symbol_rows = list(islice(symbols, 64))
    rows = []
    for index, token in enumerate(islice(tokens, 64)):
        if not isinstance(token, str) or len(token) < 3 or len(token) > 2000:
            continue
        if (token[0], token[-1]) not in {("'", "'"), ('"', '"'), ('‘', '’'), ('“', '”')}:
            continue
        symbol = symbol_rows[index] if index < len(symbol_rows) else None
        source = sources.get(symbol) if symbol else None
        rows.append({'token_index': index, 'symbol': symbol, 'surface': token,
            'enclosed_surface': token[1:-1], 'mapping_source': source,
            'punctuation_origin': 'retained_mapping' if source in {'text_vocab_links', 'symbol_to_token'} else 'unresolved',
            'communicative_function': 'unresolved',
            'possible_functions': ['quotation', 'word_mention', 'distancing', 'retained_typography'],
            'deliberate_quotation_established': False})
    return {'schema': 'ina.quotation_provenance/V1', 'tokens': rows,
            'scope': 'whole_gloss_token_outer_marks_only', 'changes_surface': False,
            'requires_interpretation': False}
