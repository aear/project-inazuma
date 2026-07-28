import math
import unittest

from vector_math import cosine_similarity, vector_norm


class VectorMathTests(unittest.TestCase):
    def test_cosine_preserves_legacy_unequal_length_behavior(self):
        expected = 1.0 / (1.0 * math.sqrt(2.0) + 1e-8)
        self.assertAlmostEqual(cosine_similarity([1.0], [1.0, 1.0]), expected)




if __name__ == "__main__":
    unittest.main()
