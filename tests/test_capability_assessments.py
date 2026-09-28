from capability_assessments import assess_english_foundation, assess_research_foundation


def test_english_assessment_uses_current_versioned_implementations():
    report = assess_english_foundation()
    assert report["domain"] == "language_english"
    assert {row["module"] for row in report["results"]} == {"language_components", "discourse", "semantic_topology", "expression_core"}
    assert all(row["version"] in {"V2", "V3", "V6"} for row in report["results"])
    assert report["live_social_expression_assessed"] is False
    assert report["promotion_authorized"] is False


def test_research_assessment_checks_boundaries_but_discloses_missing_live_quality():
    report = assess_research_foundation()
    assert report["all_cases_passed"] is True
    assert report["live_search_quality_assessed"] is False
    assert report["promotion_authorized"] is False
