"""
Tests
"""
import sys
import torch
import numpy as np
from qmc.torch import layers
from qmc.torch import models

data_x = torch.tensor([[1,2],[0,1],[3,1],[0,2]])
fm_x = layers.QFeatureMapOneHot(4)

#print(fm_x)
print(torch.functional.one_hot(data_x, 4))