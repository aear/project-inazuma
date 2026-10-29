"""Explicit character-evidence comparison, independent of live learned state."""
from __future__ import annotations
from pathlib import Path
BASELINE='b46dc1eca627d0d21d7fcadbe05937bfb5e5aeb1'


def assess(version):
    if version=='V1':
        from historical_source import historical_module
        analyze=historical_module('language_intelligence.py',BASELINE).analyze_utterance
    else:
        from language_intelligence import analyze_utterance as analyze
    cases=[]
    def add(name,component,correct):cases.append({'case':name,'component':component,'correct':bool(correct)})
    def trace(text,**context):return analyze(text,context=context).get('character_analysis') or {}
    def forms(text,**context):return trace(text,**context).get('forms') or []
    for text in ('Wait... what?!', 'US/us, IT/it.', 'cafe\u0301 café', '👩\u200d🔬 👍🏽', 'λ∘⊙', '\tA\nB\r\n'):
        data=trace(text)
        add('lossless character stream '+repr(text),'source_fidelity',
            ''.join(chr(i) for i in data.get('character_tokens',[]))==text and data.get('source_prefix')==text)
    data=trace('“Hello,” she said; why?')
    add('punctuation roles retain uncertainty','punctuation',
        any(m['surface']=='?' and m['interpretation_status']=='context_required' for m in data.get('punctuation_and_symbols',[])))
    data=trace('U.S.A. uses a CPU.')
    add('dotted initialism retained as one surface form','short_forms',any(f['surface']=='U.S.A.' and f['shape']=='initialism' for f in data.get('forms',[])))
    add('capital short form recognized without inventing expansion','short_forms',any(f['surface']=='CPU' and f['shape']=='capital_short_form' and not f['candidates'] for f in data.get('forms',[])))
    fs=forms('US us',surface_expansions=[{'form':'US','expansion':'United States','source':'fixture:definition'}])
    add('case-sensitive short form is not a pronoun rewrite','uncertainty',len(fs)==2 and len(fs[0]['candidates'])==1 and not fs[1]['candidates'])
    fs=forms('Graphics Processing Unit (GPU)')
    add('local long-form association retained with provenance','short_forms',any(f['surface']=='GPU' and any(c['text']=='Graphics Processing Unit' and c['source'].startswith('current_message:') for c in f['candidates']) for f in fs))
    for typo,known in [('raed','read'),('readd','read'),('rad','read'),('reax','read')]:
        # Three-character words are deliberately outside correction proposals.
        fs=forms(typo,known_words=[known]);c=[c['text'] for f in fs for c in f['candidates']]
        add('bounded edit candidate '+typo,'spelling',known in c if len(typo)>=4 else not c)
    fs=forms('raed',known_words=['read','reed'])
    add('competing spelling alternatives not selected','uncertainty',len(fs)==1 and {c['text'] for c in fs[0]['candidates']}=={'read','reed'} and fs[0]['selection']=='not_selected')
    result=analyze('idk',context={'surface_expansions':[{'form':'idk','expansion':"I don't know",'source':'fixture:explicit_expansion'}]})
    alternatives=result.get('surface_interpretations') or []
    add('abbreviation interpretation parsed as an unconfirmed alternative','composition',
        bool(alternatives) and alternatives[0]['text_candidate']=="I don't know" and alternatives[0]['selected'] is False
        and any(c.get('negated') for c in alternatives[0]['clauses']))
    add('expansion does not replace authoritative message','source_fidelity',result['text']=='idk' and result['clauses'][0].get('predicate') is None)
    data=trace('x'*5000)
    add('long input exposes complete-prefix boundary','bounds',data.get('source_length')==5000 and len(data.get('character_tokens',[]))==4096 and data.get('complete') is False)
    data=trace('a '*300)
    add('word-form ceiling exposed','bounds',len(data.get('forms',[]))==256 and data.get('bounds',{}).get('forms_complete') is False)
    data=trace('!'*200)
    add('punctuation ceiling exposed','bounds',len(data.get('punctuation_and_symbols',[]))==128 and data.get('bounds',{}).get('marks_complete') is False)
    data=trace("I didn't say that.")
    add('legacy word occurrences aligned without replacing them','alignment', [x['token'] for x in data.get('legacy_word_spans',[])]==['i',"didn't",'say','that'])
    tokens=analyze("I didn't say that.")['morphology']
    neg=[x for x in tokens if x['surface']=="didn't"]
    add('contraction expansions share original source span','alignment',len(neg)==2 and all(x.get('source_span')==[2,8] for x in neg))
    left=analyze('I told you.');right=analyze('You told me.')
    add('speaker roles remain distinct','discourse',left['clauses'][0]['subject']!=right['clauses'][0]['subject'])
    left=analyze("I didn't say she stole it.");right=analyze("I said she didn't steal it.")
    add('negation scope remains distinct','composition',[c['negated'] for c in left['clauses']]==[True,False] and [c['negated'] for c in right['clauses']]==[False,True])
    result=analyze('I cant go.',context={'known_words':["can't"]})
    variants=result.get('surface_interpretations') or []
    add('possible negation typo never silently changes primary meaning','composition',
        not result['clauses'][0]['negated'] and bool(variants) and variants[0]['clauses'][0]['negated'] and not variants[0]['selected'])
    result=analyze('PC PC PC',context={'surface_expansions':[{'form':'PC','expansion':str(i),'source':'fixture'} for i in range(5)]})
    add('interpretation budget keeps incomplete-set marker','bounds',
        len(result.get('surface_interpretations',[]))==8 and result.get('surface_interpretations_complete') is False)
    return {'correct':sum(r['correct'] for r in cases),'total':len(cases),'cases':cases,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'synthetic source-fidelity, alternative interpretation and existing grammar checks; not Ina capability assessment',
            'automatic_correction':False,'live_learning':'not_assessed'}
