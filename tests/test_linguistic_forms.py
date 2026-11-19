import os
from unittest import SkipTest
if os.environ.get('INA_OFFLINE_LINGUISTIC_TESTS')!='1':
    raise SkipTest('linguistic minimal pairs require verified complete shutdown')
from language_intelligence import analyze_utterance,morphology
from english_composition import compose_selected_english
from benchmarks.benchmark_linguistic_forms import assess,selected_event
from expression_core import create_expression_intent


def compose(event,**kwargs):
    intent=create_expression_intent('chosen synthetic expression',meaning_references=['meaning:fixture'],allowed_media=['text'])
    return compose_selected_english({'version':1,'events':[event]},intent,**kwargs)


def test_versioned_candidate_form_contract():
    r=assess('V2')
    assert r['correct']==r['total'],[c for c in r['cases'] if not c['correct']]


def test_held_out_agents_share_the_same_inflection_rules():
    for name,verb,lemma in [('Vale','teaches','teach'),('Tarin','tries','try'),('Cedar','wishes','wish'),('Jun','goes','go')]:
        c=analyze_utterance(name+' '+verb+' again.')['clauses'][0]
        assert c['subject']==name.casefold() and c['predicate']==lemma


def test_determiner_noun_phrase_is_not_a_finite_verb():
    for text,subject,verb in [('My wishes change.','my wishes','change'),('The changes help.','the changes','help'),('My open gate closed.','my open gate','close')]:
        c=analyze_utterance(text)['clauses'][0]
        assert c['subject']==subject and c['predicate']==verb


def test_question_can_keep_negation_and_modal_scope():
    e=selected_event(agent='you',predicate='open',modality='could',negated=True,
        arguments={'theme':'the door'},question_word='why',inverted=True)
    assert compose(e)['candidate_text']=='Why could you not open the door?'


def test_subject_question_does_not_use_object_question_inversion():
    assert compose(selected_event(agent='who',predicate='want',arguments={'theme':'the compass'},question_word='who'))['candidate_text']=='Who wants the compass?'
    assert compose(selected_event(agent='who',predicate='want',question_word='who',inverted=True))['status']=='unsupported_structure'


def test_question_rejects_unsupported_or_ambiguous_roles():
    for changes in ({'question_word':['who']},{'question_word':'whose'},
                    {'question_word':'where','inverted':False},
                    {'question_word':'who','inverted':True},
                    {'question_word':'where','voice':'passive'}):
        r=compose(selected_event(**changes))
        assert r['status']=='unsupported_structure' and r['candidate_text'] is None


def test_language_choice_and_uncertainty_are_still_authoritative():
    e=selected_event(question_word='where',inverted=True)
    for choice in ('native_only','clarify','silence','abstain'):
        assert compose(e,choice=choice)['candidate_text'] is None
    plan={'schema':'ina.experience_cognition/V3','epistemic_state':{'status':'unknown'}}
    assert compose(e,cognition_plan=plan)['status']=='needs_evidence'


def test_roundtrip_question_and_simple_selected_forms_keep_roles():
    events=[selected_event(predicate='teach',arguments={'recipient':'Rowan','theme':'the route'}),
            selected_event(agent='they',predicate='understand',tense='past',arguments={'theme':'the message'}),
            selected_event(agent='who',predicate='see',tense='past',arguments={'theme':'the comet'},question_word='who')]
    for event in events:
        text=compose(event)['candidate_text'];assert text
        c=analyze_utterance(text)['clauses'][0]
        assert c['predicate']==event['predicate'] and c['tense']==event['tense']
        assert c['subject']==event['agent'].casefold()
        assert bool(c['question_word'])==bool(event.get('question_word'))


def test_new_goal_remains_optional_and_unattributed():
    from english_learning_profile import build_english_learning_profile
    result=build_english_learning_profile()
    goal=next(g for g in result['goals'] if g['id']=='forms')
    assert goal['optional'] and goal['adoption']=='not_observed' and goal['ina_performance']=='not_assessed'
    assert result['automatic_practice'] is False and result['global_score'] is None


def test_ambiguous_apostrophes_are_not_forced_to_is_or_has():
    for text in ("Mira's compass.","She's ready.","She'd learned."):
        assert all(not t['features'].get('contraction') for t in morphology(text))


def test_contraction_expansion_keeps_existing_token_budget_visible():
    analysis=analyze_utterance("we'll "*300)
    assert analysis['incomplete'] and len(analysis['morphology'])<=512
