from utils import load_dataset

import numpy as np
import unittest

class TestLoadDataSet(unittest.TestCase):

    file_csv = 'data/test_sample.csv'
    X, y = load_dataset(path=file_csv)

    def test_if_can_read_sample_X(self):
        self.assertTrue(type(self.X) == np.ndarray)

    def test_if_can_read_sample_y(self):
        self.assertTrue(type(self.y) == np.ndarray)
        self.assertTrue(self.y.ndim == 1)


if __name__ == '__main__':
    unittest.main()
