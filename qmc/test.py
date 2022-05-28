"""
Tests
"""
import sys
import torch
import unittest
import numpy as np
import pt.layers as layers
import pt.models as models


class TestQFeatureMapOneHot(unittest.TestCase):
    def test_one_hot(self):
        self.data_x = torch.tensor([[1, 2], [0, 1], [3, 1], [0, 2]])
        self.fm_x = layers.QFeatureMapOneHot(4)
        self.assertEqual(self.fm_x(self.data_x).shape, (4, 16), "Wrong output shape")
        self.assertTrue(torch.equal(self.fm_x(self.data_x), torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])),"FAILED: QFeatureMapOneHot")


if __name__ == '__main__':
    unittest.main()