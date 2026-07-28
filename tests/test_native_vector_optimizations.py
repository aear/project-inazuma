import random
import unittest

import memory_graph as mg
import meaning_map as mm


class NativeVectorOptimizationTests(unittest.TestCase):
    def test_cached_cluster_paths_preserve_expected_shape(self):
        fragments = [{"id": "f1", "tags": ["a"]}, {"id": "f2", "tags": ["a"]}]
        clusters = mg.cluster_fragments(fragments, {"f1": [1.0, 0.0], "f2": [1.0, 0.0]}, threshold=0.5, tag_weight=0.0)
        self.assertEqual(len(clusters), 1)
        self.assertNotIn("vector_sum", clusters[0])

        encoded = [{"id": "f1", "vector": [1.0, 0.0]}, {"id": "f2", "vector": [1.0, 0.0]}]
        meaning_clusters = mm._cluster_encoded(encoded, threshold=0.5)
        self.assertEqual(len(meaning_clusters), 1)
        self.assertNotIn("vector", meaning_clusters[0]["members"][0])

    def test_native_synapses_match_python_fallback(self):
        if mg._native_vector is None or not mg._native_vector.available():
            self.skipTest("optional native kernel is not built")
        rng = random.Random(42)
        neurons = [
            {"id": f"n{index}", "vector": [rng.uniform(-1.0, 1.0) for _ in range(16)]}
            for index in range(120)
        ]
        options = dict(
            threshold=0.25,
            max_pairs=5000,
            max_edges=75,
            max_edges_per_neuron=5,
            include_direction=False,
            return_stats=True,
        )
        native_result = mg.build_synaptic_links(neurons, **options)
        accelerator = mg._native_vector
        try:
            mg._native_vector = None
            python_result = mg.build_synaptic_links(neurons, **options)
        finally:
            mg._native_vector = accelerator
        self.assertEqual(native_result, python_result)


if __name__ == "__main__":
    unittest.main()
