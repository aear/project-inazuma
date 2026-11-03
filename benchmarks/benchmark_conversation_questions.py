"""Pinned selection comparison and bounded conversational conventions fixtures."""
from pathlib import Path
from types import SimpleNamespace
BASELINE = '0b15b4bf170ba937d2299d4ce0d7c46e36a33100'


def assess(version):
    from benchmarks.benchmark_response_language_policy import functions
    from conversation_questions import question_context, review_conversational_reply
    from benchmarks.benchmark_discord_social_english import route_fixture
    if version == 'V1':
        from historical_source import historical_text, historical_module
        source = historical_text('discord_bridge.py', BASELINE)
        buffer = historical_module('conversation_scene.py', BASELINE).ConversationSceneBuffer
        support = historical_module('social_english.py', BASELINE)
        profile_builder = historical_module('english_learning_profile.py', BASELINE).build_english_learning_profile
        assessment_source = historical_text('capability_assessments.py', BASELINE)
    else:
        source = Path('discord_bridge.py').read_text()
        from conversation_scene import ConversationSceneBuffer as buffer
        import social_english as support
        from english_learning_profile import build_english_learning_profile as profile_builder
        assessment_source = Path('capability_assessments.py').read_text()
    select = functions(source)
    rows = []
    def add(name, component, ok):
        rows.append({'case': name, 'component': component, 'correct': bool(ok)})
    def choose(text, state=None, adapter=True):
        return select(state or {}, mapped_count=4, token_count=4, adapter_available=adapter,
                      expression_drive=.8, emotion_available=False, code_pointer_available=False,
                      incoming_text=text, language_preference='english')
    for text in ('Where is the lantern?', 'Can you explain that? <3', 'Tea?',
                 'How do you know', 'Ina, do you remember the garden？',
                 'What does "?" mean?'):
        q = choose(text).get('question_context', {})
        add(text, 'recognition', q.get('answer_invitation'))
    for text in ('"Where is it?"', 'The sign says "Why?".', 'She asked where it is?',
                 '`why?`', 'https://example.test/?q=why', '???', 'Hello!',
                 'What a lovely day!', 'How lovely!', '> Is this a quote?',
                 'No need to answer. How are you?', '"Unclosed question?'):
        q = choose(text).get('question_context', {})
        add(text, 'scope', 'answer_invitation' in q and not q['answer_invitation'])
    ordinary, question = choose('The lantern is blue.'), choose('Is the lantern blue?')
    add('modest preference for an available answer', 'selection',
        abs(question['scores']['respond'] - ordinary['scores']['respond'] - .12) < 1e-8)
    add('less preference for question mirroring', 'selection',
        abs(question['scores']['mirror'] - ordinary['scores']['mirror'] + .18) < 1e-8)
    repeated = choose('Is the lantern blue????????')
    add('question marks do not stack incentives', 'bounds', repeated['scores'] == question['scores'])
    add('no answer without grounded candidate', 'grounding', choose('Where is it?',adapter=False)['scores']['respond'] is None)
    add('no extra silence penalty', 'agency', question['scores']['silence'] == ordinary['scores']['silence'])
    for strategy in ('mirror', 'silence', 'respond'):
        q = choose('Where is it?', {'text_expression_intent': {'strategy': strategy}})
        add('explicit '+strategy, 'agency', q['strategy'] == strategy and q['reason'] == 'explicit_intent')
    # Real scene class, anonymous message only. The current text fits scene bounds.
    for text, expected in [('Where is it? <3',True), ('Can you explain that',True), ('"Why?"',False)]:
        message=SimpleNamespace(id='fixture',backend='discord',direction='inbound',text=text,
            sender=SimpleNamespace(display_name='Mira',is_self=False),channel=SimpleNamespace(name='fixture'),metadata={})
        scene=buffer().observe(message)
        add('scene '+text,'scene',scene['signals']['current_is_question'] is expected)
    result, captured, _ = route_fixture(source, text='Where is it?',draft='where is it')
    review=result.metadata.get('social_english_draft_review',{}).get('conversation_review',{})
    add('route flags echo but preserves selected draft','reply_review',result.text=='where is it' and review.get('status')=='surface_echo')
    add('question frame reaches symbolic context','wiring',
        (captured.get('context') or {}).get('question_context',{}).get('answer_invitation'))
    # New helper credit is conditional on actual wiring in the compared bridge.
    wired='from conversation_questions import question_context' in source
    for first, second in [('Why is it blue?', 'why is it blue'), ('Are you okay?', 'are you okay')]:
        r=review_conversational_reply(first,second)
        add('echo '+first,'reply_review',wired and r['status']=='surface_echo' and r['answer_correctness']=='not_assessed')
    for draft in ('I cannot tell yet.', 'Could you clarify which lantern?', 'Because of the filter.', '∘⊙'):
        r=review_conversational_reply('Why is it blue?',draft)
        add('does not certify '+draft,'grounding',wired and r['status']=='distinct_wording_unverified' and r['answer_correctness']=='not_assessed')
    add('length bound remains unresolved','bounds',wired and not question_context('x'*4097+'?')['complete'])
    add('no silence penalty or forced adoption in guidance','agency',wired and not question_context('Why?')['answer_required'])
    guidance = support.social_english_options().get('question_conventions', {})
    add('voluntary question conventions are offered', 'guidance',guidance.get('answer_required') is False and guidance.get('penalty_for_silence') is False)
    paired = support.review_social_english('where is it', incoming_text='Where is it?')
    add('paired private review flags echo', 'reply_review',paired.get('conversation_review',{}).get('status')=='surface_echo' and paired['outward_text'] is None)
    info = choose('Where is the lantern?').get('question_context', {})
    add('question requests place information', 'comprehension',info.get('requested_information')=='place')
    profile = profile_builder([{'module':'conversation_questions','version':'V2','correct':1,'total':2}])
    goal = next(g for g in profile['goals'] if g['id']=='listener')
    add('listener goal retains limits without claiming Ina performance','reporting',goal['implementation_status']=='limitations_retained' and goal['optional'] and goal['ina_performance']=='not_assessed' and profile['human_quality']=='not_assessed')
    add('English assessment includes conversational fixtures','reporting','("conversation_questions", ("V2",))' in assessment_source)
    result={'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'scope':'synthetic recognition, selection, scene and copied route; no live Ina assessment',
            'live_conversation_quality':'not_assessed', 'learning_gain':'not_assessed'}
    if version=='V1': result['baseline_revision']=BASELINE
    return result
