import json
import tempfile
import unittest
from pathlib import Path

from algorithm_portfolio import AlgorithmPortfolio, EmpiricalCostModel, GalacticSearch, LocalSearch, SolarSearch
from benchmark_algorithm_portfolio import sample_real_fragments


class AlgorithmPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.records = [("x", [1.0, 0.0]), ("y", [0.0, 1.0]),
                        ("xy", [0.7, 0.7]), ("neg", [-1.0, 0.0])]

    def test_all_tiers_have_identical_exact_results(self):
        expected = LocalSearch().search(self.records, [1.0, 0.0], 3)
        for actual in (SolarSearch().search(self.records, [1.0, 0.0], 3),
                       GalacticSearch(2).search(self.records, [1.0, 0.0], 3)):
            self.assertEqual([item[0] for item in expected], [item[0] for item in actual])
            for left, right in zip(expected, actual):
                self.assertAlmostEqual(left[1], right[1], places=6)

    def test_cost_model_respects_memory_budget(self):
        model = EmpiricalCostModel({
            "local": [{"items": 100, "dimensions": 2, "median_ms": 5, "peak_rss_mb": 10}],
            "solar": [{"items": 100, "dimensions": 2, "median_ms": 1, "peak_rss_mb": 50}],
            "galactic": [{"items": 100, "dimensions": 2, "median_ms": 3, "peak_rss_mb": 20}],
        })
        self.assertEqual(model.choose(100, 2, 100).tier, "solar")
        self.assertEqual(model.choose(100, 2, 25).tier, "galactic")

    def test_solar_preserves_unequal_length_semantics(self):
        records = [("short", [1.0]), ("long", [1.0, 1.0])]
        expected = LocalSearch().search(records, [1.0, 0.0], 2)
        self.assertEqual(expected, SolarSearch().search(records, [1.0, 0.0], 2))

    def test_portfolio_reports_selected_tier(self):
        hits, estimate = AlgorithmPortfolio().search(self.records, [1.0, 0.0], count=4, tier="local")
        self.assertEqual((estimate.tier, hits[0][0]), ("local", "x"))

    def test_real_sampler_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(8):
                (root / f"{index}.json").write_text(json.dumps({"id": str(index), "text": f"memory {index}"}))
            records = sample_real_fragments(root, 3, 4, 1)
            self.assertTrue(records)
            self.assertLessEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
