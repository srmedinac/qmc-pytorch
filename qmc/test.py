"""
Tests
"""
import sys
import torch
import unittest
import numpy as np
import pt.layers as layers
import pt.models as models


class TestQFeatureMapSmp(unittest.TestCase):

    def test_shape(self):
        """Test shape of output"""
        batch_size = 2
        dim = 2
        beta = 4.0
        qfm = layers.QFeatureMapSmp(dim, beta)
        inputs = torch.rand(batch_size, dim)
        outputs = qfm(inputs)
        self.assertEqual(outputs.size(), (batch_size, dim ** dim))

    def test_smp(self):
        inputs = torch.ones((2, 3))
        qfm = layers.QFeatureMapSmp(dim=2, beta=10)
        out = qfm(inputs)
        self.assertIsNone(torch.testing.assert_close(out, torch.tensor([[3.0588151e-07, 4.5396839e-05, 4.5396839e-05, 6.7374879e-03, 4.5396839e-05,
                                                                         6.7374879e-03, 6.7374874e-03, 9.9993169e-01],
                                                                        [3.0588151e-07, 4.5396839e-05, 4.5396839e-05, 6.7374879e-03, 4.5396839e-05,
                                                                         6.7374879e-03, 6.7374874e-03, 9.9993169e-01]])))


class TestQFeatureMapOneHot(unittest.TestCase):
    def test_one_hot(self):
        self.data_x = torch.tensor([[1, 2], [0, 1], [3, 1], [0, 2]])
        self.fm_x = layers.QFeatureMapOneHot(4)
        self.assertEqual(self.fm_x(self.data_x).shape,
                         (4, 16), "Wrong output shape")
        self.assertTrue(torch.equal(self.fm_x(self.data_x), torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                          [0, 1, 0, 0, 0, 0, 0, 0,
                                                                              0, 0, 0, 0, 0, 0, 0, 0],
                                                                          [0, 0, 0, 0, 0, 0, 0, 0,
                                                                              0, 0, 0, 0, 0, 1, 0, 0],
                                                                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])), "FAILED: QFeatureMapOneHot")


class TestCrossProduct(unittest.TestCase):
    def test_cross_product(self):
        self.out1 = torch.from_numpy(np.fromfile("test_tensors/cp_input1", dtype=np.float32))
        self.out1 = self.out1.reshape(2,8)
        self.out2 = torch.from_numpy(np.fromfile("test_tensors/cp_input2", dtype=np.float32))
        self.out2 = self.out2.reshape(2,3,3)
        self.expected = torch.from_numpy(np.fromfile("test_tensors/cp_expected", dtype=np.float32))
        self.expected = self.expected.reshape(2,8,3,3)
        self.cp = layers.CrossProduct()
        self.output = self.cp([self.out1, self.out2])
        self.assertTrue(torch.equal(self.output, self.expected), "FAILED: CrossProduct")
"""
class TestQFeatureMapRFF(unittest.TestCase):
    def test_rff(self):
        self.data_x = torch.tensor([[0., 0.],[0., 1,],[1., 0,],[1., 1,]])
        self.data_y = torch.tensor([[0], [1], [1], [0]])
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=2, dim=10, gamma=4, random_state=10)
        self.assertEqual(self.fm_x(self.data_x).shape,
                         (4, 4), "Wrong output shape")
        self.assertTrue(torch.equal(self.fm_x(self.data_x), torch.tensor([[0, 0, 0, 0],
                                               [0, 0, 0, 0],
                                               [0, 0, 0, 0],
                                               [0, 0, 0, 0]])),"FAILED: QFeatureMapRFF")
 """


if __name__ == '__main__':
    unittest.main()
