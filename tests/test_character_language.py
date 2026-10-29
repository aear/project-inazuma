from character_language import analyze_characters, _one_edit
from language_intelligence import analyze_utterance, morphology
from benchmarks.benchmark_character_language import assess


def test_candidate_benchmark_preserves_existing_and_new_contracts():
    result=assess('V2')
    assert result['correct']==result['total'],[r for r in result['cases'] if not r['correct']]


def test_unicode_codepoint_identity_and_span_reconstruction():
    text="Ina’s cafe\u0301?! 👩\u200d🔬\nλ∘⊙"
    result=analyze_characters(text)
    assert result['token_namespace']=='unicode_codepoint'
    assert ''.join(map(chr,result['character_tokens']))==text
    assert result['character_categories'][text.index('\u0301')]=='Mn'
    for form in result['forms']:
        assert text[slice(*form['span'])]==form['surface']
    for mark in result['punctuation_and_symbols']:
        assert text[slice(*mark['span'])]==mark['surface']


def test_spelling_uses_only_supplied_candidates_and_never_rewrites():
    assert not analyze_characters('recieve')['forms'][0]['candidates']
    result=analyze_characters('recieve',known_words=['receive'])
    assert result['forms'][0]['candidates'][0]['text']=='receive'
    assert result['source_prefix']=='recieve' and not result['original_rewritten']
    for a,b in [('read','readd'),('read','red'),('read','raed'),('read','reax')]:assert _one_edit(a,b)
    for a,b in [('read','read'),('read','random'),('abcd','badc')]:assert not _one_edit(a,b)


def test_bounded_iterables_and_candidate_caps_are_explicit():
    seen=[]
    def words():
        for i in range(1000):seen.append(i);yield 'word'+str(i)
    data=analyze_characters('unknown',known_words=words())
    assert len(seen)==257 and not data['bounds']['known_words_complete']
    expansions=[{'form':'ABC','expansion':str(i),'source':'fixture'} for i in range(129)]
    row=analyze_characters('ABC',expansions=expansions)['forms'][0]
    assert len(row['candidates'])==4 and not row['candidate_set_complete']
    assert analyze_characters('x',known_words=None,expansions=None)['source_prefix']=='x'


def test_query_budget_and_cutoff_do_not_hide_unresolved_candidates():
    text=' '.join('x'+chr(97+i)+'xxx' for i in range(20))
    data=analyze_characters(text,known_words=['xxxxx'])
    assert not data['bounds']['typo_queries_complete']
    assert not data['forms'][-1]['candidate_set_complete']
    clipped=analyze_characters(' '*4090+'receive-not-finished',known_words=['receive'])
    assert not clipped['bounds']['source_complete'] and not clipped['forms'][-1]['complete']
    assert not clipped['forms'][-1]['candidates']


def test_short_forms_keep_multiple_contextual_meanings_and_case():
    e=[{'form':'PC','expansion':'personal computer','source':'fixture:hardware'},
       {'form':'PC','expansion':'police constable','source':'fixture:police'}]
    rows=analyze_characters('PC pc',expansions=e)['forms']
    assert len(rows[0]['candidates'])==2 and not rows[1]['candidates']
    assert all(f['selection']=='not_selected' for f in rows)
    unrelated=analyze_characters('Random Words (GPU)')['forms']
    assert all(not f['candidates'] for f in unrelated)


def test_alternative_structure_does_not_change_primary_negation_or_referents():
    original=analyze_utterance('idk')
    expanded=analyze_utterance('idk',context={'surface_expansions':[{'form':'idk','expansion':"I don't know",'source':'fixture'}]})
    assert original['clauses']==expanded['clauses'] and original['referents']==expanded['referents']
    alternative=expanded['surface_interpretations'][0]
    assert not alternative['selected'] and not alternative['meaning_certified']
    assert alternative['clauses'][0]['negated']
    tokens=morphology("Ina’s book; isn't here.")
    assert all('source_span' in t for t in tokens)
    assert next(t for t in tokens if t['surface']==';')['features']['punctuation']


def test_discord_character_evidence_reaches_symbolic_context_without_output_rewrite():
    from benchmarks.benchmark_discord_social_english import route_fixture,sources
    text='GPU?! idk…'
    result,captured,_=route_fixture(sources('V2'),text=text,draft='Perhaps.')
    assert result.text=='Perhaps.'
    evidence=result.metadata['incoming_character_analysis']
    assert evidence['source_prefix']==text
    assert captured['context']['character_analysis']==evidence


def test_symbolic_reply_keeps_character_trace_with_exact_surface(monkeypatch,tmp_path):
    import language_processing as lp
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lp,'_load_text_vocab_links_scoped',lambda *a,**k:{'links':[]})
    monkeypatch.setattr(lp,'_build_text_vocab_word_symbol_index',lambda *a,**k:{'hello':'s'})
    monkeypatch.setattr(lp,'resolve_text_vocab_meanings',lambda *a,**k:[])
    monkeypatch.setattr(lp,'_shadow_audit_text_vocab_mappings',lambda *a,**k:None)
    monkeypatch.setattr(lp,'load_output_guidance',lambda *a,**k:{})
    monkeypatch.setattr(lp,'choose_expression_symbols',lambda symbols,*a,**k:symbols)
    monkeypatch.setattr(lp,'build_dual_symbolic_message',lambda *a,**k:{'text':'native','native_text':'native'})
    result=lp.generate_symbolic_reply_from_text('Hello?!',child='fixture',playback=False)
    assert result['character_analysis']['source_prefix']=='Hello?!'
    assert result['unknown']==[]


def test_real_words_and_negation_alternatives_are_not_silently_corrected():
    exact=analyze_characters('from',known_words=['form','from'])
    assert not exact['forms'][0]['candidates']
    result=analyze_utterance('I cant go.',context={'known_words':["can't"]})
    assert not result['clauses'][0]['negated']
    alternative=result['surface_interpretations'][0]
    assert alternative['text_candidate']=="I can't go."
    assert alternative['clauses'][0]['negated'] and not alternative['selected']


def test_utterance_alternative_budget_exposes_incomplete_set():
    expansions=[{'form':'PC','expansion':'candidate '+str(i),'source':'fixture'} for i in range(5)]
    result=analyze_utterance('PC PC PC',context={'surface_expansions':expansions})
    assert len(result['surface_interpretations'])==8
    assert not result['surface_interpretations_complete']
    assert all(not item['selected'] for item in result['surface_interpretations'])
