from perspective_review import begin_review, review_step
from benchmarks.benchmark_perspective_review import measure
from benchmarks.benchmark_english_review import measure as english_measure
import language_intelligence


def test_variants_share_exit_and_evidence_guards():
    for variant in ('private_tool', 'instinct'):
        assert all(measure(variant)['signals'].values())


def test_all_allowed_depths_end_and_each_step_can_exit():
    event = {'interpretation_reference': 'meaning:1', 'question': 'What changed?',
             'countercheck': 'Test the opposite', 'evidence_references': ['event:0'], 'novelty': True}
    for budget in range(1, 5):
        state = begin_review(event, variant='instinct', max_steps=budget)
        for step in range(budget):
            assert state['status'] == 'ready'
            assert review_step(state, choice='stop')['status'] == 'stop'
            state = review_step(state, choice='continue', observation_references=[f'event:{step+1}'])
        assert state['status'] == 'budget_exhausted'
        assert state['remaining_steps'] == 0


def test_instinct_trigger_differs_from_private_choice():
    event = {'interpretation_reference': 'meaning:1', 'question': 'What changed?',
             'countercheck': 'Test the opposite', 'evidence_references': ['event:0'], 'novelty': True}
    assert begin_review(event, variant='private_tool') is None
    assert begin_review(event, variant='instinct') is not None
    assert begin_review({**event, 'novelty': False}, variant='instinct') is None


def test_instinct_plan_is_optional_and_read_only():
    from experience_cognition import plan_experience_cognition
    plain = plan_experience_cognition({})
    assert 'perspective_review' not in plain
    plan = plan_experience_cognition({}, perspective_event={})
    assert plan['perspective_review'] is None
    assert not any(plan['memory_boundary'].values())


def test_new_tools_require_local_voluntary_authority():
    from instruction_authority import seal_code_command, verify_code_command, InstructionAuthorityError
    for action in ('perspective_begin', 'perspective_step'):
        rejected = False
        try:
            seal_code_command({'action': action, 'source': 'ina_voluntary_choice'}, None)
        except InstructionAuthorityError:
            rejected = True
        assert rejected
        rejected = False
        try:
            verify_code_command({'action': action, 'source': 'ina_voluntary_choice'})
        except InstructionAuthorityError:
            rejected = True
        assert rejected


def test_english_structural_cases():
    assert all(english_measure(language_intelligence).values())


def test_native_intent_keeps_passive_roles():
    from semantic_event import build_semantic_event
    event = build_semantic_event('The book was given to Mary.')['events'][0]
    assert event['agent'] is None
    assert event['arguments']['theme']['surface'] == 'the book'
    assert event['arguments']['recipient']['surface'] == 'mary'


def test_private_tools_can_start_continue_and_exit(tmp_path):
    from instruction_authority import LOCAL_VOLUNTARY_CODE_AUTHORITY, seal_code_command
    from personal_tool_runtime import execute_personal_tool_command, capability_catalog
    config = {'ina_hdd_writable_path': str(tmp_path / 'personal')}
    event = {'interpretation_reference': 'meaning:1', 'question': 'What changed?',
             'countercheck': 'Test the opposite', 'evidence_references': ['event:0']}
    def execute(payload):
        command = seal_code_command({**payload, 'source': 'ina_voluntary_choice'},
                                    LOCAL_VOLUNTARY_CODE_AUTHORITY)
        return execute_personal_tool_command(command, child='Ina', project_root=tmp_path,
                                             config=config)['value']
    assert 'perspective_begin' in capability_catalog()['commands']
    state = execute({'action': 'perspective_begin', 'event': event, 'max_steps': 2})
    state = execute({'action': 'perspective_step', 'state': state, 'choice': 'continue',
                     'observation_references': ['event:1']})
    assert state['status'] == 'ready'
    state = execute({'action': 'perspective_step', 'state': state, 'choice': 'defer'})
    assert state['status'] == 'defer'


def test_benchmark_registry_exposes_both_versions_and_pinned_english():
    from module_benchmarks import benchmark_module
    for module in ('perspective_review', 'english_review'):
        results = benchmark_module(module)
        assert [item.version for item in results] == ['V1', 'V2']
        assert all(item.correct == item.total for item in results if item.version == 'V2')
    assert benchmark_module('english_review', ('V1',))[0].source_revision == 'a31f1323ee4bb589d0ca0bf02aea87201b69cd21'
