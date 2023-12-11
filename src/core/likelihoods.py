from src.misc.constraint_utils import softplus, invsoftplus

import torch
import torch.nn as nn
from torch.nn import init

import numpy as np


class Gaussian(nn.Module):
    """
    Gaussian likelihood
    """

    def __init__(self, ndim=1, init_val=1.0):
        super(Gaussian, self).__init__()
        self.logvariance = nn.Parameter(torch.log(torch.tensor(init_val)), requires_grad=True)
        self.variance = torch.exp(self.logvariance)
        #self._initialize(init_val)

    def _initialize(self, x):
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(x)).item())

    def logdensity(self, x, mu, var):
        return -0.5 * (torch.log(2 * torch.tensor(np.pi)) + torch.log(var) + (mu - x)**2 / var)

    def predict_mean_and_var(self, Fmu, Fvar):
        return torch.clone(Fmu), Fvar + self.variance
    
    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)
