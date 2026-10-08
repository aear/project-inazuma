from quotation_provenance import trace_gloss_quotations


def test_retained_marks_are_not_new_intent():
    result = trace_gloss_quotations(["'meanings'"], ['s'], {'s': 'text_vocab_links'})
    row = result['tokens'][0]
    assert row['surface'] == "'meanings'"
    assert row['punctuation_origin'] == 'retained_mapping'
    assert not row['deliberate_quotation_established']
    assert row['communicative_function'] == 'unresolved'


def test_apostrophe_unquoted_word_and_unknown_origin_are_distinct():
    result = trace_gloss_quotations(["don't", 'meanings', '“meanings”'], ['a', 'b', 'c'], {})
    assert [row['token_index'] for row in result['tokens']] == [2]
    assert result['tokens'][0]['punctuation_origin'] == 'unresolved'


def test_admission_budget_is_bounded():
    def stream():
        for _ in range(64):
            yield "'word'"
        raise AssertionError('overconsumed')
    assert len(trace_gloss_quotations(stream(), stream(), {})['tokens']) == 64
