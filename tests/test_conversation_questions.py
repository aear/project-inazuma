from pathlib import Path
from conversation_questions import question_context, review_conversational_reply
from benchmarks.benchmark_response_language_policy import functions
from benchmarks.benchmark_conversation_questions import assess


def test_versioned_conversation_comparison():
    old,new=assess('V1'),assess('V2')
    assert old['correct']<old['total']
    assert new['correct']==new['total'],[r for r in new['cases'] if not r['correct']]


def test_held_out_question_forms_and_contractions():
    for text in ('Where did you leave the paint? 😊', 'Could we talk about music',
                 'Why can’t I see it?', 'Which colour do you prefer?', 'Ready？',
                 'Hello Ina, where did it go?', 'What did she say about “why?”?'):
        assert question_context(text)['answer_invitation'],text
    for text in ('“Did you leave?”', "'Where are we?'", 'They wondered why?',
                 'No pressure to reply. Where is it?', "You don't have to answer. Okay?",
                 '```\nWhere is it?\n```', 'I like the song.', 'The question was why?', 'What I know is limited.', 'Where I left it is unclear.', 'How I feel is private.'):
        assert not question_context(text)['answer_invitation'],text
    assert question_context('Where is the brush?')['requested_information']=='place'
    assert question_context('Why is that?')['requested_information']=='reason'


def test_question_weight_is_bounded_and_does_not_punish_silence():
    select=functions(Path('discord_bridge.py').read_text())
    def call(text,state=None,available=True,preference='english'):
        return select(state or {},mapped_count=4,token_count=4,adapter_available=available,
            expression_drive=.7,emotion_available=False,code_pointer_available=False,
            incoming_text=text,language_preference=preference)
    for preference in ('english','native_only','mixed','clarify','abstain'):
        plain=call('I see the moon.',preference=preference)
        question=call('Do you see the moon?',preference=preference)
        assert question['scores']['silence']==plain['scores']['silence']
        assert question['scores']['respond']>plain['scores']['respond']
        assert call('Do you see the moon?'*50,preference=preference)['scores']==question['scores']
        assert call('Where is it?',available=False,preference=preference)['scores']['respond'] is None
        for chosen in ('mirror','silence','respond'):
            assert call('Where is it?',{'text_expression_intent':chosen},preference=preference)['strategy']==chosen


def test_echo_review_is_descriptive_not_an_answer_grader():
    r=review_conversational_reply('Are you happy?', 'are you happy')
    assert r['status']=='surface_echo' and not r['echo_is_answer_evidence']
    for draft in ('Yes.', 'No.', 'I do not know.', 'Could you clarify?', '∘⊙', 'I asked myself: are you happy?'):
        r=review_conversational_reply('Are you happy?',draft)
        assert r['status']=='distinct_wording_unverified'
        assert r['answer_correctness']=='not_assessed' and r['outward_text'] is None
    assert review_conversational_reply('How are you?', '')['status']=='no_expression'


def test_question_budgets_and_quotations_stay_unresolved():
    for text in ('x'*4097, '"Did you mean it?', '`Did you mean it?', '“Did you mean it?', "'Did you mean it?"):
        r=question_context(text)
        assert not r['complete'] and not r['answer_invitation']
    assert not question_context(None)['complete']
    assert question_context('Why'+'?'*4093)['response_bonus']==.12
    assert question_context('why'+'?'*4094)['response_bonus']==0.


def test_guidance_is_available_without_becoming_a_reply():
    from social_english import social_english_options, review_social_english
    options=social_english_options()
    assert not options['question_conventions']['answer_required']
    assert not options['question_conventions']['penalty_for_silence']
    assert len(options['opportunities'])==2 and options['outward_text'] is None
    review=review_social_english('where is it',incoming_text='Where is it?')
    assert review['conversation_review']['status']=='surface_echo'
    assert review['question_context']['requested_information']=='place'
    assert review['outward_text'] is None
    for choice in ('native_only','silence','clarify','abstain'):
        r=review_social_english('∘⊙',incoming_text='Why?',choice=choice)
        assert r['status']=='choice_preserved' and r['outward_text'] is None
