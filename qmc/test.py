"""
Tests
"""
import sys
import torch
import numpy as np
import pt.layers as layers
import pt.models as models

data_x = torch.tensor([[1,2],[0,1],[3,1],[0,2]])
fm_x = layers.QFeatureMapOneHot(4)

print(fm_x(data_x))
print(torch.nn.functional.one_hot(data_x, 4))