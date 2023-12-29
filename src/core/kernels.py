# Copyright 2016 James Hensman, alexggmatthews, PabloLeon, Valentine Svensson
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

import torch
import numpy as np
from . import parameter

jitter = 1e-6 

class Kern(torch.nn.Module):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.
        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.
        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        super().__init__()
        self.name = name
        self.input_dim = int(input_dim)
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif isinstance(active_dims, slice):
            self.active_dims = active_dims
            if active_dims.start is not None and active_dims.stop is not None and active_dims.step is not None:
                assert len(range(*active_dims)) == input_dim  # pragma: no cover
        else:
            self.active_dims = np.array(active_dims, dtype=np.int32)
            assert len(active_dims) == input_dim

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
        assert X.size(1) == self.input_dim

        return X, X2
    
class RBF(Kern):
    def __init__(self, input_dim, variance=1.0, init_val=None,
                 active_dims=None, ARD=False, ACD=False, name=None):
        super(RBF, self).__init__(input_dim, active_dims, name=name)
        self.variance = parameter.PositiveParam(variance)
        if ACD:
            low_tri_shape  = (self.input_dim*(self.input_dim+1)//2,) 
            if init_val is None:
                l = torch.ones(low_tri_shape, dtype=torch.float64)
            else:
                l = init_val * torch.ones(low_tri_shape, dtype=torch.float64)
            self.L = parameter.Param(val=l)
            self.ACD = True
            self.rbf_type = 'ACD'
        else:
            if ARD:
                if init_val is None:
                    lengthscales = torch.ones(input_dim)
                else:
                    # accepts float or array:
                    lengthscales = init_val * torch.ones(input_dim)
                self.lengthscales = parameter.PositiveParam(lengthscales)
                self.ARD = True
                self.rbf_type = 'ARD'
            else:
                if init_val is None:
                    lengthscales = 1.0
                self.lengthscales = parameter.PositiveParam(lengthscales)
                self.ARD = False
                self.rbf_type = 'standard'
    
    def square_dist(self, X, X2):
        X = X / (self.lengthscales.get() + jitter)
        Xs = (X**2).sum(1)

        if X2 is None:
            dist = -2 * torch.matmul(X, X.t())
            dist += Xs.view(-1, 1) + Xs.view(1, -1)
            return dist

        X2 = X2 / self.lengthscales.get()
        X2s = (X2**2).sum(1)
        dist = -2 * torch.matmul(X, X2.t())
        dist += Xs.view(-1, 1) + X2s.view(1, -1)
        return dist

    def malhanobis_dist(self, X, X2):
        if X2 is None:
            X2 = X
        N = X.size(0)
        M = X2.size(0)
        precision = self.precision
        # compute z, z2
        z = self._z(X, precision, dim=1) # (N, 1)
        z2 = self._z(X2, precision, dim=1) # (M, 1)
        # compute X(X2Λ)ᵀ
        X2Lambda = torch.matmul(X2, precision) # (M, input_dium)
        XX2LambdaT = torch.matmul(X, X2Lambda.t()) # (N, M)
        # compute z1ᵀ 
        ones_M = torch.ones(M, 1, device=precision.device, dtype=torch.float64) # (M, 1)
        zcol = torch.matmul(z, ones_M.t()) # (N, M)
        # compute 1z2ᵀ 
        ones_N = torch.ones(N, 1, device=precision.device, dtype=torch.float64) # (N, 1)
        zrow = torch.matmul(ones_N, z2.t()) # (N, M)

        dist = zcol - 2*XX2LambdaT + zrow # (N, M)
        return dist
    
    def _z(self, X, Lambda, dim=2):
        XLambda = torch.matmul(X, Lambda) # (N/M, input_dim)
        XLambdaX = torch.mul(XLambda, X) # (M, input_dim)
        return torch.sum(XLambdaX, dim=dim, keepdim=True) # (N/M, 1)
    
    def _fill_triangular(self):
        lower_indices = torch.tril_indices(self.input_dim, self.input_dim) # (2, lsize)
        l_matrix = torch.zeros(self.input_dim, self.input_dim, device=self.L.device, dtype=self.L.dtype) # (input_dim, input_dim)
        l_matrix[lower_indices.tolist()] = self.L.get()
        return l_matrix
    
    def Kdiag(self, X, presliced=False):
        return self.variance.get().expand(X.size(0))
    
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)

        if self.rbf_type != 'ACD': # ARD or standard
            res = self.variance.get() * torch.exp(-0.5 * self.square_dist(X, X2))
        else: # ACD
            res = self.variance.get() * torch.exp(-0.5 * self.malhanobis_dist(X, X2))
        return res
    
    @property
    def precision(self):
        l_matrix = self._fill_triangular()
        precision_matrix = torch.matmul(l_matrix, l_matrix.t())
        return precision_matrix
    
    @property
    def l_matrix(self):
        l_matrix = self._fill_triangular()
        return l_matrix
