from src.misc.constraint_utils import softplus, invsoftplus

import numpy as np
import torch
from torch import nn
from torch.nn import init

from torch.distributions import Normal

prior_weights = Normal(0.0, 1.0)


def sample_normal(shape, seed=None):
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    return torch.tensor(rng.normal(size=shape).astype(np.float32))


class RBF(torch.nn.Module):
    """
    Implements squared exponential kernel with kernel computation and weights and frequency sampling for Fourier features
    """

    def __init__(self, D_in, D_out=None, dimwise=False):
        """
        @param D_in: Number of input dimensions
        @param D_out: Number of output dimensions
        @param dimwise: If True, different kernel parameters are given to output dimensions
        """
        super(RBF, self).__init__()
        self.D_in = D_in
        self.D_out = D_in if D_out is None else D_out
        self.dimwise = dimwise
        self.active_dims = slice(self.D_in)
        lengthscales_shape = (self.D_out, self.D_in) if dimwise else (self.D_in,)
        variance_shape = (self.D_out,) if dimwise else (1,)
        self.loglengthscales = nn.Parameter(torch.log(torch.full(size=lengthscales_shape, fill_value=float(self.D_in)**0.5)), requires_grad=True)
        self.logvariance = nn.Parameter(torch.log(torch.full(size=variance_shape, fill_value=float(0.1))), requires_grad=True)
        #self.lengthscales = torch.exp(self.loglengthscales)
        #self.variance = torch.tensor(torch.exp(self.logvariance), requires_grad=True)
        #self._initialize()

    def _initialize(self):
        init.constant_(self.unconstrained_lengthscales, invsoftplus(torch.tensor(self.D_out**0.5)).item())
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(0.1)).item())
    
    @property
    def lengthscales(self):
        return torch.exp(self.loglengthscales)

    @property
    def variance(self):
        return torch.exp(self.logvariance)
    
    def square_dist_dimwise(self, X, X2=None):
        """
        Computes squared euclidean distance (scaled) for dimwise kernel setting
        @param X: Input 1 (N,D_in)
        @param X2: Input 2 (M,D_in)
        @return: Tensor (D_out, N,M)
        """
        X = X.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,N,D_in)
        Xs = torch.sum(torch.pow(X, 2), dim=2)  # (D_out,N)
        if X2 is None:
            return -2 * torch.einsum('dnk, dmk -> dnm', X, X) + \
                Xs.unsqueeze(-1) + Xs.unsqueeze(1)  # (D_out,N,N)
        else:
            X2 = X2.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,M,D_in)
            X2s = torch.sum(torch.pow(X2, 2), dim=2)  # (D_out,N)
            return -2 * torch.einsum('dnk, dmk -> dnm', X, X2) + Xs.unsqueeze(-1) + X2s.unsqueeze(1)  # (D_out,N,M)

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = (X**2).sum(1)

        if X2 is None:
            dist = -2 * torch.matmul(X, X.t())
            dist += Xs.view(-1, 1) + Xs.view(1, -1)
            return dist

        X2 = X2 / self.lengthscales
        X2s = (X2**2).sum(1)
        dist = -2 * torch.matmul(X, X2.t())
        dist += Xs.view(-1, 1) + X2s.view(1, -1)
        return dist

    def K(self, X, X2=None, X_inducing=False, X2_inducing=False, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        res = self.variance * torch.exp(-0.5 * self.square_dist(X, X2))
        return res
    
    def _slice(self, X, X2):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims`.
        :param X: Input 1 (NxD).
        :param X2: Input 2 (MxD), may be None.
        :return: Sliced X, X2, (Nxself.input_dim).
        """
        #if isinstance(self.active_dims, slice):
        X = X[:, self.active_dims]
        if X2 is not None:
            X2 = X2[:, self.active_dims]
        # I think advanced indexing does the right thing also for the second case
        #else:
        assert X.size(1) == self.D_in

        return X, X2
    
    def Kdiag(self, X, presliced=False):
        return self.variance.expand(X.size(0)) # torch.full(X.size()[:-1], self.variance.item(), dtype=torch.float64)
    
    def sample_freq(self, S, seed=None):
        """
        Computes random samples from the spectral density for Squared exponential kernel
        @param S: Number of features
        @param seed: random seed
        @return: Tensor a random sample from standard Normal (D_in, S, D_out) if dimwise else (D_in, S)
        """
        omega_shape = (self.D_in, S, self.D_out) if self.dimwise else (self.D_in, S)
        omega = sample_normal(omega_shape, seed)  # (D_in, S, D_out) or (D_in, S)
        lengthscales = self.lengthscales.T.unsqueeze(1) if self.dimwise else self.lengthscales.unsqueeze(
            1)  # (D_in,1,D_out) or (D_in,1)
        return omega / lengthscales  # (D_in, S, D_out) or (D_in, S)