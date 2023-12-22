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

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])



class Stationary(Kern):
    """
    Base class for kernels that are stationary, that is, they only depend on
        r = || x - x' ||
    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        super(Stationary, self).__init__(input_dim, active_dims, name=name)
        self.variance = parameter.PositiveParam(variance)
        if ARD:
            if lengthscales is None:
                lengthscales = torch.ones(input_dim)
            else:
                # accepts float or array:
                lengthscales = lengthscales * torch.ones(input_dim)
            self.lengthscales = parameter.PositiveParam(lengthscales)
            self.ARD = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            self.lengthscales = parameter.PositiveParam(lengthscales)
            self.ARD = False

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

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return (r2 + 1e-12)**0.5

    def Kdiag(self, X, X_inducing=False, presliced=False):
        return self.variance.get().expand(X.size(0))


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """

    def K(self, X, X2=None, X_inducing=False, X2_inducing=False, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        res = self.variance.get() * torch.exp(-0.5 * self.square_dist(X, X2))
        return res