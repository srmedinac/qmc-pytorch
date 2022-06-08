import torch
import torch.nn as nn
from . import layers

class QMClassifier(nn.Module):
    """
    A Quantum Measurement Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y: Quantum feature map layer for outputs
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
    """
    def __init__(self, fm_x, fm_y, dim_x, dim_y):
        super(QMClassifier, self).__init__()
        self.fm_x = fm_x
        self.fm_y = fm_y
        self.qm = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = torch.tensor(0.,requires_grad=False)

    def get_probs(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        probs = self.dm2dist(rho_y)
        return probs

    def forward(self, inputs):
        x, y = inputs
        self.get_probs(x)
        psi_x = self.fm_x(x)
        psi_y = self.fm_y(y)
        psi = self.cp1([psi_x, psi_y])
        rho = self.cp2([psi, torch.conj(psi)])
        self.num_samples = x.size(0)
        rho = torch.sum(rho, dim=0)
        if x.size(1) is not None:
            self.qm.weights[0] = rho
        return rho

    def fit(self, *args, **kwargs):
        result = super(QMClassifier, self).fit(*args, **kwargs)
        self.qm.weights[0] = self.qm.weights[0] / self.num_samples
        return result
    
    def get_rho(self):
        return self.qm.rho




        

    #call_train
    #train_step
    #fit
    #get rho
