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


def gaussian(x, mu, var):
    return -0.5 * (float(np.log(2 * np.pi)) + torch.log(var) + (mu-x)**2/var)

def lognormal(x, mu, var):
    return -torch.sum(torch.square((x-mu)/var)) / 2.

def loglaplace(x, b):
    return -torch.sum(torch.abs(x) / b)

def logwishart(L, P):
    n = L.size(0)
    return -torch.sum(torch.log(torch.diagonal(torch.abs(L)))) - torch.trace(n*P) / 2.0

def loginvwishart(L, P):
    n = L.size(0)
    return -(2*n + 1)  * torch.sum(torch.log(torch.diagonal(torch.abs(L)))) - torch.trace(torch.inverse(P)) / 2.0

def loghorseshoe(P, scale):
    K = 1 / torch.sqrt(2 * torch.tensor(np.pi)**3)
    A = (scale / P) ** 2
    lb = K / 2 * torch.log(1 + 4 * A)
    ub = K * torch.log(1 + 2 * A)
    return torch.sum(torch.log((lb + ub) / 2.))