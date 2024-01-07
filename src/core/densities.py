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

jitter = 1e-6 # 1e-19

def bernoulli(p, y):
    return torch.log(y*p+(1-y)*(1-p))

def gaussian(x, mu, var):
    return -0.5 * (float(np.log(2 * np.pi)) + torch.log(var) + (mu-x)**2/var)

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
    """
    P += torch.ones_like(P) * jitter # try to avoid cholesky crush
    K = 1 / torch.sqrt(2 * torch.tensor(np.pi)**3)
    A = (scale / P) ** 2
    lb = K / 2 * torch.log(1 + 4 * A)
    ub = K * torch.log(1 + 2 * A)
    return torch.sum(torch.log((lb + ub) / 2.))
    """
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