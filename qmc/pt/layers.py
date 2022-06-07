"""
Layers implementing quantum feature maps, measurements and
utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_approximation import RBFSampler
from . import _RBFSamplerORF


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
        self.points = torch.reshape(torch.linspace(
            0, 1, self.dim), len(inputs)*[1] + [self.dim])
        vals = torch.unsqueeze(inputs, -1)
        dists = (self.points - vals) ** 2
        sm = torch.exp(-dists * self.beta)
        sums = torch.sum(sm, dim=-1)
        sm = sm / sums.unsqueeze(-1)
        amp = torch.sqrt(sm)
        b_size = amp.size(0)
        t_psi = amp[:, 0, :]
        for i in range(1, amp.size(1)):
            t_psi = torch.einsum('...i,...j->...ij', t_psi, amp[:, i, :])
            t_psi = torch.reshape(t_psi, (b_size, - 1))
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
        t_psi = out[:, 0, :]
        for i in range(1, out.size(1)):
            t_psi = torch.einsum('...i,...j->...ij', t_psi, out[:, i, :])
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

    def __init__(self, input_dim: int, dim: int = 100, gamma: float = 1.0, random_state=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state

    def forward(self, inputs):
        rbf_sampler = RBFSampler(
            n_components=self.dim, random_state=self.random_state, gamma=self.gamma)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = torch.tensor(rbf_sampler.random_weights_)
        self.offset = torch.tensor(rbf_sampler.random_offset_)
        vals = torch.matmul(
            inputs.float(), self.rff_weights.float()) + self.offset.float()
        vals = torch.cos(vals)
        # fixme: this is a hack
        vals = vals * torch.sqrt(torch.tensor(2. / self.dim))
        norms = torch.norm(vals, dim=-1)
        psi = vals / torch.unsqueeze(norms, 0)
        return psi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)


class QFeatureMapORF(nn.Module):
    """Quantum feature map using Orthogonal Random Features.
    Uses `ORFSampler` from sklearn to approximate an RBF kernel using
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

    def __init__(self, input_dim: int, dim: int = 100, gamma: float = 1.0, random_state=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state

    def forward(self, inputs):
        rbf_sampler = _RBFSamplerORF.RBFSamplerORF(
            n_components=self.dim, random_state=self.random_state, gamma=self.gamma)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = torch.tensor(rbf_sampler.random_weights_)
        self.offset = torch.tensor(rbf_sampler.random_offset_)
        vals = torch.matmul(
            inputs.float(), self.rff_weights.float()) + self.offset.float()
        vals = torch.cos(vals)
        # fixme: this is a hack
        vals = vals * torch.sqrt(torch.tensor(2. / self.dim))
        norms = torch.norm(vals, dim=1)
        psi = vals / torch.unsqueeze(norms, 0)
        return psi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)


class CrossProduct(nn.Module):
    """Cross product of two two inputs.

    Input shape:
        A list of 2 tensors [t1, t2] with shapes
        (batch_size, n) and (batch_size, m)
    Output shape:
        (batch_size, n, m)

    """

    def __init__(self, **kwargs):
        super(CrossProduct, self).__init__(**kwargs)
        self.idx1 = 'abcdefghij'
        self.idx2 = 'klmnopqrst'

    def forward(self, inputs):
        self.eins_eq = ('...' + self.idx1[:len(inputs[0]) - 1] + ',' +
                        '...' + self.idx2[:len(inputs[1])] + '->' +
                        '...' + self.idx1[:len(inputs[0]) - 1] +
                        self.idx2[:len(inputs[1])])
        cp = torch.einsum(self.eins_eq,
                          inputs[0], inputs[1])
        return cp

    def compute_output_shape(self, input_shape):
        return (input_shape[0][1], input_shape[1][1])


class QuantumDenseLayer(nn.Module):
    """Quantum dense layer for classification.

    Input shape:
        (batch_size, dim_in)
        where dim_in is the dimension of the input state
    Output shape:
        (batch_size, dim_out)
        where dim_out is the dimension of the output state
    Arguments:
        dim_in: int. the dimension of the input state
        dim_out: int. the dimension of the output state
        last_layer: bool. True if the layer is the last layer of a sequential model
    """

    def __init__(self, dim_in: int, dim_out: int, last_layer=True, **kwargs):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.last_layer = last_layer

    def forward(self, inputs):
        self.eig_vec = torch.nn.init.normal_(
            torch.empty(self.dim_out, self.dim_in))
        norms = torch.unsqueeze(torch.norm(self.eig_vec, dim=1), 1)
        eig_veg = self.eig_vec / norms
        psy_out = torch.einsum('ij,...j->...i', eig_veg, inputs)
        norms_psy_out = torch.unsqueeze(torch.norm(psy_out, dim=1), 1)
        psy_out = psy_out / norms_psy_out
        if self.last_layer:
            prob_out = torch.square(psy_out)
            return prob_out
        return psy_out


class QFeatureMapComplexRFF(nn.Module):
    """Quantum feature map including the complex part of random Fourier Features.
    Uses `RBFSampler` from sklearn to approximate an RBF kernel using
    complex random Fourier features.

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

    def __init__(self, input_dim: int, dim: int = 100, gamma: float = 1.0, random_state=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state

    def forward(self, inputs):
        rbf_sampler = RBFSampler(
            n_components=self.dim, random_state=self.random_state, gamma=self.gamma)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = torch.tensor(rbf_sampler.random_weights_)
        self.offset = torch.tensor(rbf_sampler.random_offset_)
        vals = torch.matmul(inputs.float(), self.rff_weights.float())
        vals = torch.complex(torch.cos(vals), torch.sin(vals))
        vals = vals * \
            torch.sqrt(torch.tensor(1. / self.dim)).to(torch.complex64)
        norms = torch.norm(vals, dim=1)
        psi = vals / torch.unsqueeze(norms, 0)
        return psi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)


class Vector2DensityMatrix(nn.Module):
    """
    Represents a state vector as a factorized density matrix.

    Input shape:
        (batch_size, dim)
    Output shape:
        (batch_size, dim + 1, 1)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs):
        ones = torch.full((inputs.size(0), 1), 1.0)
        rho = torch.cat((ones, inputs), 1)
        rho = torch.unsqueeze(rho, -1)
        return rho


class QMeasureClassif(nn.Module):
    """Quantum measurement layer for classification.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_x: int. the dimension of the input  state
        dim_y: int. the dimension of the output state
    """

    def __init__(self, dim_x: int, dim_y: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y

    def forward(self, inputs):
        self.rho = torch.zeros(
            (self.dim_x, self.dim_y, self.dim_x, self.dim_y))
        oper = torch.einsum('...i,...j->...ij',
                            inputs, torch.conj(inputs))
        rho_res = torch.einsum(
            '...ik, klmn, ...mo -> ...ilon', oper, self.rho, oper)
        trace_val = torch.einsum('...ijij->...', rho_res)
        trace_val = torch.unsqueeze(trace_val, -1)
        trace_val = torch.unsqueeze(trace_val, -1)
        trace_val = torch.unsqueeze(trace_val, -1)
        trace_val = torch.unsqueeze(trace_val, -1)
        rho_res = rho_res / trace_val
        rho_y = torch.einsum('...ijik->...jk', rho_res)
        return rho_y


class QMeasureClassifEig(nn.Module):
    """Quantum measurement layer for classification.
    Represents the density matrix using a factorization:

    This representation is ameanable to gradient-based learning.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        num_eig: Number of eigenvectors used to represent the density matrix
    """

    def __init__(self, dim_x: int, dim_y: int = 2, num_eig: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y
        if num_eig < 1:
            num_eig = dim_x * dim_y
        self.num_eig = num_eig

    def forward(self, inputs):

        self.eig_vec = torch.randn(
            self.dim_x * self.dim_y, self.num_eig, requires_grad=True)
        self.eig_val = torch.randn(self.num_eig, requires_grad=True)
        norms = torch.unsqueeze(torch.norm(self.eig_vec, dim=0), 0)
        eig_vec = self.eig_vec / norms
        relu = nn.ReLU()
        eig_val = relu(self.eig_val)
        eig_val = eig_val / torch.sum(eig_val)
        eig_vec = torch.reshape(
            eig_vec, (self.dim_x, self.dim_y, self.num_eig))
        eig_vec_y = torch.einsum('...i,ijk->...jk', inputs, eig_vec)
        eig_val_sr = torch.sqrt(eig_val)
        eig_val_sr = torch.unsqueeze(eig_val_sr, 0)
        eig_val_sr = torch.unsqueeze(eig_val_sr, 0)
        eig_vec_y = eig_vec_y * eig_val_sr
        rho_y = torch.matmul(eig_vec_y, eig_vec_y.transpose(-2, -1).conj())
        trace_val = torch.einsum('...jj->...', rho_y)
        trace_val = torch.unsqueeze(trace_val, -1)
        trace_val = torch.unsqueeze(trace_val, -1)
        rho_y = rho_y / trace_val
        return rho_y

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_y, dim_x, dim_y)
        """
        if (len(list(rho.size())) != 4 or
            rho.shape[0] != self.dim_x or
            rho.shape[2] != self.dim_x or
            rho.shape[1] != self.dim_y or
                rho.shape[3] != self.dim_y):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_y},'
                f' {self.dim_x}, {self.dim_y})')
        rho_prime = torch.reshape(rho,(self.dim_x * self.dim_y, self.dim_x * self.dim_y,))
        e, v = torch.linalg.eigh(rho_prime)
        self.eig_vec = v[:, -self.num_eig:]
        self.eig_val = e[-self.num_eig:]
        return e


