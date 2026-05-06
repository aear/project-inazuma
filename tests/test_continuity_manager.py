import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from continuity_manager import ContinuityManager


def test_continuity_manager_proposes_minimum_core_map_with_limits(tmp_path):
    manager = ContinuityManager("TestIna", memory_root=tmp_path / "memory", max_fragments=42)

    proposal = manager.propose_minimum_core_map_integration(
        limit_rules={
            "max_total_rss_gb": 96,
            "max_managed_rss_gb": 90,
            "min_available_gb": 8,
            "memory_estimate_high_gb": 12,
        },
        trigger={"largest_module": "meaning_map.py", "largest_module_ram_gb": 140},
    )

    assert proposal["status"] == "proposal_only"
    assert proposal["review_required"] is True
    assert proposal["limit_rules"]["max_total_rss_gb"] == 96
    assert proposal["minimum_core_bounds"]["max_fragments_sampled"] == 42
    assert len(proposal["options"]) == 3
    assert all("memory_profile" in option for option in proposal["options"])
