# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
# Copyright 2017 Thomas Viehmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2

from ..core.conditionals import conditional
#from .. import kullback_leiblers
from ..core import parameter
from ..core.likelihoods import Gaussian
from ..core.kernels import RBF

class VDMGP(nn.Module):
    def __init__(self, X, Y, M, K, minibatch_size=None, **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        """
        # sort out the X, Y into MiniBatch objects if required.
        self.minibatch_size = minibatch_size

        # init the super class, accept args
        super(VDMGP, self).__init__()
        self.num_data = X.size(0)
        self.D = X.size(1)
        self.K = K
        self.M = M
        self.X = X
        self.Y = Y
        
        # init variational parameters
        pca_components = torch.tensor(PCA(self.K).fit(X).components_, device=X.device) # K x D
        input_scales = (10 / np.ptp(torch.mm(X, pca_components.t()), axis=0) ** 2).reshape(-1, 1) # 1 x 1
    
        self.q_mu = parameter.Param(pca_components * input_scales) # K x D
        self.q_cov = parameter.PositiveParam((1 / self.D + (0.001 / self.D) * torch.randn(self.K, self.D))) # K x D

        # init inducing points
        inducing_points_init = torch.tensor(kmeans2(torch.mm(X, self.q_mu.get().t()).detach().numpy(), self.M, minit='points')[0], dtype=torch.float64) # M x K
        self.Z = nn.parameter.Parameter(inducing_points_init.double(), requires_grad=True) # M x K

        # init gaussian likelihood
        self.likelihood = Gaussian(dtype=torch.float64)

        # init RBF-ARD kernel
        self.sigma_f = parameter.PositiveParam(torch.std(Y))
        self.kern = RBF(input_dim=self.K, ARD=True)

    def compute_psi0(self, X):
        N = X.shape[0]
        return N * (self.sigma_f.get() ** 2)
    
    def compute_psi1(self, X):
        q_mu_X = torch.tensordot(self.q_mu.get(), X, [[1], [1]]) # K x N

        top = torch.subtract(
            q_mu_X[:, :, None], self.Z.permute(1, 0)[:, None, :]
        ) ** 2 # K x N x M

        bot = 1 + torch.sum(self.q_cov.get()[:, None, :] * torch.pow(X[None], 2), axis=2) # K x N
        bot = bot[:, :, None]  # K x N x 1

        psi1 = torch.exp(-0.5 * top / bot) / torch.sqrt(bot) # K x N x M

        return torch.multiply(self.sigma_f.get(), torch.prod(psi1, axis=0))
    
    def compute_batched_psi2(self, X):
        q_mu_X = torch.tensordot(self.q_mu.get(), X, [[1], [1]]) # K x N

        Z_diff = torch.subtract(self.Z[:, None, :], self.Z[None, :, :]) # M x M x K

        Z_bar = 0.5 * (self.Z[:, None, :] + self.Z[None, :, :]) # M x M x K

        top = torch.subtract(
            q_mu_X[:, :, None, None], Z_bar.permute(2, 0, 1)[:, None, :, :]
        ) ** 2 # K x N x M x M

        bot = (2 * torch.sum(self.q_cov.get()[:, None, :] * torch.pow(X[None], 2), axis=2) + 1)  # K x N

        bot = bot[:, :, None, None] # K x N x 1 x 1 identity()?

        right = torch.exp(-top / bot) / torch.sqrt(bot) # K x N x M x M
        right = torch.prod(right, axis=0) # N x M x M

        left = torch.multiply(
            torch.pow(self.sigma_f.get(), 2),
            torch.exp(-0.25 * torch.sum(torch.pow(Z_diff, 2), axis=2)),
        ) # M x M

        return torch.multiply(left, right)
    
    def compute_psi2(self, X=None):
        return torch.sum(self.compute_batched_psi2(X), axis=0)
    
    def compute_likelihood(self, X=None, Y=None, jitter_level=1e-6):
        N = X.shape[0]
        D = self.D
        M = self.M
        pi = torch.tensor(np.pi, dtype=X.dtype)
        sigma2 = self.likelihood.variance.get()[None] # 1 x 1

        Kuu = self.kern.K(self.Z, self.Z) + torch.eye(M, dtype=X.dtype, device=X.device) * jitter_level # M x M
        chol_Ku = torch.linalg.cholesky(Kuu, upper=False)

        F1 = (
            - N * torch.log(2 * pi)
            - (N - M) * torch.log(sigma2)
            + torch.logdet(chol_Ku) # use cholesky_logdet() of deep VDMGP
        ) # 1 x 1

        YY = torch.mm(Y.t(), Y) # 1 x 1

        psi2 = self.compute_psi2(X) # M x M

        Ku_Psi2 = sigma2 * Kuu + psi2 + torch.eye(M, dtype=X.dtype, device=X.device) * jitter_level # M x M
        chol_Ku_Psi2 = torch.linalg.cholesky(Ku_Psi2) # M x M

        F1 += -torch.logdet(chol_Ku_Psi2) - YY / sigma2 # use cholesky_logdet() of deep VDMGP

        psi1 = self.compute_psi1(X) # N x M

        Psi1_Y = torch.mm(psi1.t(), Y) # M x 1

        F1 += torch.divide(
            torch.mm(
                Psi1_Y.t(), torch.linalg.solve(chol_Ku_Psi2, Psi1_Y)), # tf.cholesky_solve()
            sigma2,
        )

        psi0 = self.compute_psi0(X)

        KuiPsi2 = torch.linalg.solve(chol_Ku, psi2) # M x M # tf.cholesky_solve()

        F1 += -psi0 / sigma2 + torch.trace(KuiPsi2) / sigma2

        KL = torch.sum(torch.log(self.q_cov.get()), axis=1)
        KL -= D * torch.log(
            torch.sum(self.q_cov.get() + torch.pow(self.q_mu.get(), 2), axis=1)
        )
        KL += D * np.log(D)
        KL = torch.sum(KL)

        return torch.squeeze(0.5 * (F1 + KL))

    def predict_y(self, Xnew, full_cov=False, jitter_level=1e-6):
        N = self.X.shape[0]
        nNew = Xnew.shape[0]
        D = self.D
        M = self.M
        K = self.K

        sigma2 = self.likelihood.variance.get()
        Kuu = self.kern.K(self.Z, self.Z) + torch.eye(M, dtype=self.X.dtype, device=self.X.device) * jitter_level # M x M
        chol_Ku = torch.linalg.cholesky(Kuu, upper=False)

        psi1 = self.compute_psi1(self.X)
        psi2 = self.compute_psi2(self.X)

        Ku_Psi2 = sigma2 * Kuu + psi2 + torch.eye(M, dtype=self.X.dtype, device=self.X.device) * jitter_level # M x M
        chol_Ku_Psi2 = torch.linalg.cholesky(Ku_Psi2) # M x M

        Psi1_Y = torch.mm(psi1.t(), self.Y) # M x 1
        alpha = torch.linalg.solve(chol_Ku_Psi2, Psi1_Y)

        Psi1new = self.compute_psi1(Xnew)

        mean = torch.mm(Psi1new, alpha)

        Psi2new = self.compute_batched_psi2(Xnew)
        var = torch.trace(
            sigma2
            * torch.linalg.solve(torch.tile(chol_Ku_Psi2[None], [nNew, 1, 1]), Psi2new)
            - torch.linalg.solve(torch.tile(chol_Ku[None], [nNew, 1, 1]), Psi2new)
            + torch.tile(torch.mm(alpha, alpha.t())[None], [nNew, 1, 1])
            @ Psi2new
        )
        var = torch.reshape(var, (-1, 1)) + self.sigma_f ** 2 + sigma2 - mean ** 2

        return mean, var