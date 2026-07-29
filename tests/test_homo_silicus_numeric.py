import math
import unittest

import homo_silicus_numeric as hs


class HomoSilicusNumericTests(unittest.TestCase):
    def test_shape_index_reshape_and_transpose(self):
        matrix = hs.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual((matrix.shape, matrix.ndim, matrix.size), ((2, 3), 2, 6))
        self.assertEqual(matrix[1, 2], 6.0)
        self.assertEqual(matrix[0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(matrix.reshape(3, 2).tolist(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(matrix.T.tolist(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_arithmetic_broadcast_and_reductions(self):
        matrix = hs.array([[1, 2], [3, 4]])
        self.assertEqual((matrix + hs.array([10, 20])).tolist(), [[11.0, 22.0], [13.0, 24.0]])
        self.assertEqual(hs.sum(matrix, axis=0).tolist(), [4.0, 6.0])
        self.assertEqual(hs.mean(matrix, axis=1).tolist(), [1.5, 3.5])
        self.assertEqual(hs.clip(matrix * 2, 3, 7).tolist(), [[3.0, 4.0], [6.0, 7.0]])

    def test_dot_matmul_norm_and_cosine_rows(self):
        matrix = hs.array([[1, 0], [0, 2], [-1, 0]])
        vector = hs.array([1, 0])
        self.assertEqual(hs.dot(matrix, vector).tolist(), [1.0, 0.0, -1.0])
        self.assertAlmostEqual(hs.norm(hs.array([3, 4])), 5.0)
        scores = hs.cosine_rows(matrix, vector).tolist()
        self.assertAlmostEqual(scores[0], 1.0, places=7)
        self.assertEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], -1.0, places=7)
        product = hs.matmul(hs.array([[1, 2]]), hs.array([[3], [4]]))
        self.assertEqual(product.tolist(), [[11.0]])

    def test_fail_loudly_on_ragged_or_misaligned_data(self):
        with self.assertRaises(ValueError):
            hs.array([[1], [2, 3]])
        with self.assertRaises(ValueError):
            hs.dot([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
