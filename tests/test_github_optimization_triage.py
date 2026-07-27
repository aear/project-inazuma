import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model_manager as mm


def test_stale_guard_does_not_replace_current_pressure_evidence():
    context = {
        "trend_pressure": 0.10,
        "current_pressure": 0.0,
        "ina_rss_gb": 2.4,
        "scheduler_max_total_rss_gb": 96.0,
    }

    assert not mm._resource_pressure_warrants_github_report(
        context,
        {"min_resource_trend_pressure": 0.74},
    )


def test_near_scheduler_limit_warrants_report_even_with_low_trend_score():
    context = {
        "trend_pressure": 0.41,
        "current_pressure": 0.0,
        "ina_rss_gb": 80.0,
        "scheduler_max_total_rss_gb": 96.0,
    }

    assert mm._resource_pressure_warrants_github_report(
        context,
        {"min_resource_trend_pressure": 0.74},
    )


def test_optimization_fingerprint_uses_stable_target_identity():
    first = mm._github_optimization_fingerprint(
        continuity_proposal_mode=False,
        largest_module="early_comm.py",
    )
    second = mm._github_optimization_fingerprint(
        continuity_proposal_mode=False,
        largest_module="early_comm.py",
    )

    assert first == second
    assert first != mm._github_optimization_fingerprint(
        continuity_proposal_mode=False,
        largest_module="memory_graph.py",
    )
