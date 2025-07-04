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


def batch_tril(A):
    B = A.clone()
    ii, jj = np.triu_indices(B.size(-2), k=1, m=B.size(-1))
    B[..., ii, jj] = 0
    return B


def batch_diag(A):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    return A[..., ii, jj]


def conditional(Xnew, X, kern, f, full_cov=False, q_sqrt=None, whiten=False,
                jitter_level=1e-6, return_Lm=False):
    """
    Given F, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about F as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.
    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).
    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

    Args:
        Xnew: data matrix, size N x D.
        X: data points, size M x D.
        kern: kernel.
        f: data matrix, M x K, representing the function values at X,
            for K functions.
        q_sqrt: matrix of standard-deviations or Cholesky matrices,
            size M x K or M x M x K.
        whiten: boolean of whether to whiten the representation as
            described above.

    Returns:
        Two element tuple with conditional mean and variance.
    """

    # compute kernel stuff
    num_data = X.size(0)  # M
    num_func = f.size(1)  # K
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X, X) + torch.eye(num_data, dtype=X.dtype, device=X.device) * jitter_level
    Lm = torch.linalg.cholesky(Kmm, upper=False)

    # Compute the projection matrix A
    A = torch.linalg.solve(Lm, Kmn)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = kern.K(Xnew, Xnew) - torch.matmul(A.t(), A)
        fvar = fvar.unsqueeze(0).expand(num_func, -1, -1) # K x N x N
    else:
        fvar = kern.Kdiag(Xnew) - (A**2).sum(0)
        fvar = fvar.unsqueeze(0).expand(num_func, -1) # K x N
    # fvar is K x N x N or K x N

    # another backsubstitution in the unwhitened case 
    # (complete the inverse of the cholesky decomposition)
    if not whiten:
        A = torch.linalg.solve_triangular(Lm.t(), A, upper=True)

    # construct the conditional mean
    fmean = torch.matmul(A.t(), f)

    if q_sqrt is not None:
        if q_sqrt.dim() == 2:
            LTA = A * q_sqrt.t().unsqueeze(2)  # K x M x N
        elif q_sqrt.dim() == 3:
            L = batch_tril(q_sqrt.permute(2, 0, 1))  # K x M x M
            LTA = torch.matmul(L.transpose(-2, -1), A)  # K x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt :{}".format(q_sqrt.dim()))
        if full_cov:
            fvar = fvar + torch.matmul(LTA.t(), LTA)  # K x N x N
        else:
            fvar = fvar + (LTA**2).sum(1)  # K x N
    fvar = fvar.permute(*range(fvar.dim()-1, -1, -1))  # N x K or N x N x K

    if return_Lm:
        return fmean, fvar, Lm

    return fmean, fvar
    
def bgp_conditional_regression(Xnew, X, Y, kern, likelihood_variance, full_cov=False, jitter_level=1e-6):
    """
        Xnew is a data matrix, point at which we want to predict

        This method computes
            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.
    """
    Kx = kern.K(X, Xnew)
    K = kern.K(X) + torch.eye(X.size(0), dtype=X.dtype, device=X.device) * likelihood_variance
    L = torch.linalg.cholesky(K, upper=False)

    A = torch.linalg.solve(L, Kx)  # could use triangular solve, note gesv has B first, then A in AX=B
    V  = torch.linalg.solve(L, Y) # could use triangular solve

    fmean = torch.mm(A.t(), V)
    if full_cov:
        fvar = kern.K(Xnew) - torch.mm(A.t(), A)
        fvar = fvar.unsqueeze(2).expand(fvar.size(0), fvar.size(1), Y.size(1))
    else:
        fvar = kern.Kdiag(Xnew) - (A**2).sum(0)
        fvar = fvar.view(-1, 1)
        fvar = fvar.expand(fvar.size(0), Y.size(1))

    return fmean, fvar

def bgp_conditional_classification(Xnew, X, Y, kern, full_cov=False, jitter_level=1e-6):
    """
        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so
            v ~ N(0, I)
            f = Lv + m(x)
        with
            L L^T = K
    """
    Kx = kern.K(X, Xnew)
    K = kern.K(X) + torch.eye(X.size(0), dtype=X.dtype, device=X.device) * jitter_level
    L = torch.linalg.cholesky(K, upper=False)

    A = torch.linalg.solve(L, Kx)  # could use triangular solve, note gesv has B first, then A in AX=B
    V  = torch.linalg.solve(L, Y) # could use triangular solve

    fmean = torch.mm(A.t(), V)
    if full_cov:
        fvar = kern.K(Xnew) - torch.mm(A.t(), A)
        fvar = fvar.unsqueeze(2).expand(fvar.size(0), fvar.size(1), Y.size(1))
    else:
        fvar = kern.Kdiag(Xnew) - (A**2).sum(0)
        fvar = fvar.view(-1, 1)
        fvar = fvar.expand(fvar.size(0), Y.size(1))

    return fmean, fvar