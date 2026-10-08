def test_dual_gloss_retains_mapping_quotation_provenance(monkeypatch):
    import language_processing as lp
    monkeypatch.setattr(lp, '_load_text_vocab_links_scoped', lambda *a, **k: {'fixture': True})
    monkeypatch.setattr(lp, '_lookup_text_vocab_word', lambda *a, **k: {'word': "'term'", 'symbol_word': 'glyph'})
    monkeypatch.setattr(lp, '_candidate_native_word', lambda *a, **k: 'glyph')
    result = lp.build_dual_symbolic_message(['s'], child='fixture', native_style='glyphs',
                                           fallback_to_symbol_to_token=False)
    assert result['gloss_text'] == "'term'"
    assert result['quotation_provenance']['tokens'][0]['punctuation_origin'] == 'retained_mapping'
    assert result['expression_realisations'][1]['content']['quotation_provenance'] == result['quotation_provenance']
