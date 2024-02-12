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
        
        # init variational parameters
        pca_components = torch.tensor(PCA(self.K).fit(X).components_, device=X.device)
        input_scales = (10 / np.ptp(torch.mm(X, pca_components.t()), axis=0) ** 2).reshape(-1, 1)
    
        self.q_mu = parameter.Param(pca_components * input_scales)
        self.q_cov = parameter.PositiveParam((1 / self.D + (0.001 / self.D) * np.random.randn(self.K, self.D)))

        # init inducing points
        inducing_points_init = torch.tensor(kmeans2(X @ self.q_mu.get().t(), self.M, minit='points')[0], dtype=torch.float64)
        self.Z = nn.parameter.Parameter(inducing_points_init.double(), requires_grad=True)

    def prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu.get(), self.q_sqrt.get())
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu.get(), self.q_sqrt.get())
        else:
            K = self.kern.K(self.Z.get()) + torch.eye(self.num_inducing, out=self.Z.new()) * self.jitter_level
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu.get(), self.q_sqrt.get(), K)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu.get(), self.q_sqrt.get(), K)
        return KL

    def compute_log_likelihood(self, X=None, Y=None):
        """
        This gives a variational bound on the model likelihood.
        """

        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        # Get prior KL.
        KL = self.prior_KL()

        # Get conditionals
        fmean, fvar = self.predict_f(X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, Y)

        # re-scale for minibatch size
        scale = float(self.num_data) / X.size(0)

        return var_exp.sum() * scale - KL

    def predict_f(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.Z.get(), self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt.get(), full_cov=full_cov, whiten=self.whiten,
                                           jitter_level=self.jitter_level)
        return mu + self.mean_function(Xnew), var