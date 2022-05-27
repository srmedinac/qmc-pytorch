import torch
import torch.nn as nn
import layers

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
        self.qm = layers.QMClassif(dim_x=dim_x, dim_y=dim_y)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = torch.tensor(0.,requires_grad=False)

    def forward(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        probs = self.dm2dist(rho_y)
        return probs

    #call_train
    #train_step
    #fit
    #get rho
