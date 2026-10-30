from capability_assessments import assess_english_foundation, assess_research_foundation


def test_english_assessment_uses_current_versioned_implementations():
    report = assess_english_foundation()
    assert report["domain"] == "language_english"
    assert {row["module"] for row in report["results"]} == {"language_components", "discourse", "semantic_topology", "expression_core", "character_language", "self_name_reference", "english_workbench", "language_transfer_limits"}
    assert all(row["version"] in {"V2", "V3", "V6"} for row in report["results"])
    assert report["live_social_expression_assessed"] is False
    assert report["promotion_authorized"] is False


def test_research_assessment_checks_boundaries_but_discloses_missing_live_quality():
    report = assess_research_foundation()
    assert report["all_cases_passed"] is True
    assert report["live_search_quality_assessed"] is False
    assert report["promotion_authorized"] is False


def test_english_profile_does_not_promote_implementation_checks_to_ina_results():
    report=assess_english_foundation()
    profile=report['learning_profile']
    assert profile['subject']=='current_implementation' and profile['global_score'] is None
    assert all(g['ina_performance']=='not_assessed' and g['optional'] for g in profile['goals'])
    assert not report['live_social_expression_assessed']
