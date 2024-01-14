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

import numpy as np
import scipy.special
import torch
import math

jitter = 1e-6 # 1e-19

def bernoulli(p, y):
    return torch.log(y*p+(1-y)*(1-p))

def gaussian(x, mu, var):
    return -0.5 * (float(np.log(2 * np.pi)) + torch.log(var) + (mu-x)**2/var)

def multivariate_normal(x, mu, L):
    """
    L is the Cholesky decomposition of the covariance.
    x and mu are either vectors (ndim=1) or matrices. In the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x - mu
    if d.dim() == 1:
        d = d.unsqueeze(1)
    alpha = torch.linalg.solve_triangular(L, d, upper=False)
    alpha = alpha.squeeze(1)
    num_col = 1 if x.dim() == 1 else x.size(1)
    num_dims = x.size(0)
    ret = - 0.5 * num_dims * num_col * float(np.log(2 * np.pi))
    ret += - num_col * torch.diag(L).log().sum()
    ret += - 0.5 * (alpha**2).sum()
    # ret = - 0.5 * (alpha**2).mean()
    return ret

def lognormal(x, mu, var):
    return -torch.sum(torch.square((x-mu)/var)) / 2.

def loglaplace(x, b):
    return -torch.sum(torch.abs(x) / b)

def logwishart(L, P):
    n = L.size(0)
    diag = torch.clamp(torch.diagonal(torch.abs(L)), min=1e-8)
    return -torch.sum(torch.log(diag)) - torch.trace(n*P) / 2.0

def loginvwishart(L, P):
    n = P.size(0)
    diag = torch.diagonal(torch.clamp(torch.abs(L), min=1e-8))
    return -(2*n + 1)  * torch.sum(torch.log(diag)) - torch.trace(torch.inverse(P)) / 2.0

def loghorseshoe(x, scale):
    # Credits to TensorFlow
    x += 1e-19
    xx = (x / scale)**2 / 2
    g = 0.5614594835668851
    b = 1.0420764938351215
    h_inf = 1.0801359952503342
    q = 20. / 47. * xx**1.0919284281983377
    h = 1. / (1 + xx**(1.5)) + h_inf * q / (1 + q)
    c = -.5 * torch.log(2 * torch.tensor(np.pi)**3) - torch.log(g * scale)
    z = torch.log1p(-torch.tensor(g)) - torch.log(torch.tensor(g))
    return torch.sum(-torch.nn.functional.softplus(z - xx / (1 - g)) + torch.log(
            torch.log1p(g / xx - (1 - g) / (h + b * xx)**2)) + c)
    