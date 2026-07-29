import unittest
from pathlib import Path

import numeric_evolution as evolution


class NumericEvolutionTests(unittest.TestCase):
    def test_stable_source_passes_static_capability_gate(self):
        source = Path("homo_silicus_numeric.py").read_text(encoding="utf-8")
        self.assertEqual(evolution.validate_candidate_source(source), [])

    def test_capability_gate_rejects_io_and_process_imports(self):
        source = "import os\nimport subprocess\ndef array(x):\n    return builtins.open('/tmp/x', 'w')\n"
        errors = evolution.validate_candidate_source(source)
        self.assertIn("import not allowed: os", errors)
        self.assertIn("import not allowed: subprocess", errors)
        self.assertIn("call not allowed: open", errors)

    def test_bootstrap_interval_requires_repeatable_improvement(self):
        low, high = evolution._bootstrap_ci([-0.10] * 20, samples=500)
        self.assertLessEqual(low, -0.10)
        self.assertLess(high, 0.0)
        low, high = evolution._bootstrap_ci([-0.1, 0.1] * 10, samples=500)
        self.assertLess(low, 0.0)
        self.assertGreaterEqual(high, 0.0)

    def test_equal_candidate_does_not_win_benchmark_gate(self):
        report = evolution.benchmark_pair(
            Path("homo_silicus_numeric.py"), Path("homo_silicus_numeric.py"),
            workloads=((4, 4),), trials=5,
        )
        self.assertFalse(report["accepted"])
        self.assertEqual(report["workloads"][0]["trials"], 5)

    def test_hardware_provenance_has_reproducibility_fields(self):
        provenance = evolution.hardware_provenance()
        self.assertTrue({"platform", "machine", "processor", "python", "logical_cpus"}.issubset(provenance))


if __name__ == "__main__":
    unittest.main()
