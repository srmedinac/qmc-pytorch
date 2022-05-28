"""
Layers implementing quantum feature maps, measurements and
utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_approximation import RBFSampler


class QFeatureMapSmp(nn.Module):
    """Quantum feature map using soft max probabilities.
    input values are asummed to be between 0 and 1.

    Input shape:
        (batch_size, dim1)
    Output shape:
        (batch_size, dim ** dim1)
    Arguments:
        dim: int. Number of dimensions to represent each value
        beta: parameter beta of the softmax function
    """
    def __init__(self, dim: int = 2, beta: float = 4.0, **kwargs):
        super(QFeatureMapSmp, self).__init__(**kwargs)
        self.dim = dim
        self.beta = beta
        

    def forward(self, inputs):
        self.points = torch.reshape(torch.linspace(0, 1, self.dim),len(inputs)*[1] + [self.dim])
        vals = torch.unsqueeze(inputs,-1)
        dists = (self.points - vals) ** 2
        sm = torch.exp(-dists * self.beta)
        sums = torch.sum(sm, dim=-1)
        sm = sm / sums.unsqueeze(-1)
        amp = torch.sqrt(sm)
        b_size = torch.shape(amp)[0]
        t_psi = amp[:,0,:]
        for i in range(1, amp.shape[1]):
            t_psi = torch.einsum('...i,...j->...ij', t_psi, amp[:,i,:])
            t_psi = torch.reshape(t_psi, (b_size - 1))
        return t_psi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim ** input_shape[1])

class QFeatureMapOneHot(nn.Module):
    """Quantum feature map using one-hot encoding.
    input values are indices, with 0<index<num_classes

    Input shape:
        (batch_size, dim)
    Output shape:
        (batch_size, num_classes ** dim)
    Arguments:
        num_classes: int. Number of dimensions to represent each value
    """

    def __init__(self, num_classes: int = 2, **kwargs):
        super(QFeatureMapOneHot, self).__init__(**kwargs)
        self.num_classes = num_classes

    def forward(self, inputs):
        out = nn.functional.one_hot(inputs, self.num_classes)
        b_size = out.size(0)
        t_psi = out[:,0,:]
        for i in range(1, out.size(1)):
            t_psi = torch.einsum('...i,...j->...ij', t_psi, out[:,i,:])
            t_psi = torch.reshape(t_psi, (b_size, - 1))
        return t_psi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes ** input_shape[1])


class QFeatureMapRFF(nn.Module):
    """Quantum feature map using random Fourier Features.
    Uses `RBFSampler` from sklearn to approximate an RBF kernel using
    random Fourier features.

    Input shape:
        (batch_size, dim_in)
    Output shape:
        (batch_size, dim)
    Arguments:
        input_dim: dimension of the input
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim: int, dim: int = 100, gamma: float = 1.0, random_state = None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state

    def forward(self, inputs):
        rbf_sampler = RBFSampler(n_components=self.dim, random_state=self.random_state, gamma=self.gamma)
        x = np.zeros(shape=(1,self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = None #continue here



# Test QFeatureMapSmp
fm_x = QFeatureMapSmp(dim=4, beta=40)
x_in_dim = 2
x_out_dim = fm_x.compute_output_shape((1, x_in_dim))[1]
assert x_out_dim == 16


    
