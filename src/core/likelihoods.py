from src.misc.constraint_utils import softplus, invsoftplus

import torch
import torch.nn as nn
from torch.nn import init

import numpy as np


class Gaussian(nn.Module):
    """
    Gaussian likelihood
    """

    def __init__(self, ndim=1, init_val=0.25):
        super(Gaussian, self).__init__()
        self.unconstrained_variance = torch.nn.Parameter(torch.ones(ndim), requires_grad=True)
        self._initialize(init_val)

    def _initialize(self, x):
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(x)).item())

    @property
    def variance(self):
        return softplus(self.unconstrained_variance)

    def logdensity(self, x, mu, var):
        return -0.5 * (torch.log(2 * torch.tensor(np.pi)) + torch.log(var) + (mu - x)**2 / var)

    def predict_mean_and_var(self, Fmu, Fvar):
        return torch.clone(Fmu), Fvar + self.variance
    
    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)
