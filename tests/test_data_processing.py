# tests/test_data_processing.py

import unittest
from src.train import create_sequences

class TestDataProcessing(unittest.TestCase):
    def test_create_sequences(self):
        X = np.array([[i] for i in range(10)])
        y = np.array([i for i in range(10)])
        seq_length = 3
        X_seq, y_seq = create_sequences(X, y, seq_length)
        expected_X = np.array([
            [[0], [1], [2]],
            [[1], [2], [3]],
            [[2], [3], [4]],
            [[3], [4], [5]],
            [[4], [5], [6]],
            [[5], [6], [7]],
            [[6], [7], [8]],
            [[7], [8], [9]]
        ])
        expected_y = np.array([3,4,5,6,7,8,9])
        self.assertTrue(np.array_equal(X_seq, expected_X))
        self.assertTrue(np.array_equal(y_seq, expected_y))

if __name__ == '__main__':
    unittest.main()
