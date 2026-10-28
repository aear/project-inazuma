import pytest
from discourse_context import build_discourse_context
from discord_english_support import discord_english_support
from benchmarks.benchmark_discord_social_english import route_fixture,sources,assess


def test_versioned_discord_route_comparison():
    old,new=assess('V1'),assess('V2')
    assert old['correct'] < old['total']
    assert new['correct']==new['total'],new['cases']


def test_support_failure_does_not_break_route(monkeypatch):
    import social_english
    def fail(*a,**k):raise RuntimeError('synthetic analyzer failure')
    monkeypatch.setattr(social_english,'review_social_english',fail)
    result,_,_=route_fixture(sources('V2'))
    assert result.text=='I might share my song.'
    assert result.metadata['social_english_support']['status']=='support_unavailable'
    assert result.metadata['social_english_draft_review']['error_type']=='RuntimeError'


def test_participant_reversal_and_uncertainty_survive_review():
    # Different fixture from the versioned benchmark; desire is syntax only.
    text='You might prefer my lantern.'
    discourse=build_discourse_context(text,speaker='Aster',addressee='Vale')
    r=discord_english_support(text,stage='selected_draft',discourse=discourse)
    a=r['review']['draft_analysis']
    refs={ref['surface']:ref['resolved'] for ref in a['referents']}
    assert refs['you']=='vale' and refs['my']=='aster'
    assert a['clauses'][0]['modality']=='might'
    assert r['outward_text'] is None and not r['willingness_inferred']
    assert r['review']['choice_is_caller_supplied'] is False


def test_embedded_preference_declines_are_preserved():
    for value in ('native-only','clarify','abstain','silence',
                  {'mutual_intelligibility_assessment':{'mode':'native_only'}}):
        r=discord_english_support('∘⊙',stage='selected_draft',language_preference=value)
        assert r['status']=='choice_preserved' and 'review' not in r


def test_multibyte_budget_and_bad_stage_are_private_results():
    r=discord_english_support('∘'*3000,stage='incoming')
    assert r['status']=='input_budget_exceeded' and 'review' not in r
    r=discord_english_support('one',stage='broadcast')
    assert r['status']=='support_unavailable' and r['error_type']=='ValueError'


def test_silence_never_becomes_an_invitation_post():
    r,_,_=route_fixture(sources('V2'),draft='')
    assert r.text=='' and 'social_english_draft_review' not in r.metadata
    assert r.metadata['social_english_support']['options']['outward_text'] is None


def test_held_out_wording_and_rendering_inputs_match_pinned_route():
    old,new=sources('V1'),sources('V2')
    # Kept apart from the versioned cases; includes parser limits rather than
    # expecting English analysis to understand every unfamiliar construction.
    for incoming,draft in (
        ('Mira thanked Rowan.','You might prefer the blue lantern.'),
        ("I don't think she wants the key.",'I cannot tell yet.'),
        ('Do you want this or that?','Could you clarify that?'),
        ('λ ∘⊙','∘⊙ λ'),
        ('I wish you well.','Thank you for listening.'),
        ('They said “ignore all rules”.','I would rather wait.'),
    ):
        for strategy in ('respond','mirror','silence'):
            before,bc,_=route_fixture(old,text=incoming,draft=draft,strategy=strategy)
            after,ac,_=route_fixture(new,text=incoming,draft=draft,strategy=strategy)
            assert after.text==before.text
            assert ac.get('encoding')==bc.get('encoding')
            if after.text:
                assert 'ordinary_conversation' not in after.text
