import torch

def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False, return_Lm=False):
    """
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: M x N
    :param Kmm: M x M
    :param Knn: N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    num_func = f.size(1)  # R
    Lm = torch.cholesky(Kmm)

    # Compute the projection matrix A
    A = torch.triangular_solve(Kmn.t(), Lm, upper=False)[0]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - A.t().matmul(A)
        fvar = fvar.unsqueeze(0).expand(num_func, -1, -1)  # R x N x N
    else:
        fvar = Knn - torch.sum(A**2, dim=0)
        fvar = fvar.unsqueeze(0).expand(num_func, -1)  # R x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = torch.triangular_solve(A, Lm.t(), upper=True)[0]

    # construct the conditional mean
    fmean = A.t().matmul(f)

    if q_sqrt is not None:
        if q_sqrt.dim() == 2:
            LTA = A * q_sqrt.t().unsqueeze(2)  # R x M x N
        elif q_sqrt.dim() == 3:
            L = q_sqrt
            A_tiled = A.unsqueeze(0).expand(num_func, -1, -1)
            LTA = L.matmul(A_tiled).transpose(1, 2)  # R x M x N
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.dim()))
        if full_cov:
            fvar = fvar + LTA.matmul(LTA.transpose(1, 2))  # R x N x N
        else:
            fvar = fvar + torch.sum(LTA**2, dim=1)  # R x N

    if not full_cov:
        fvar = fvar.t()  # N x R
    if return_Lm:
        return fmean, fvar, Lm

    return fmean, fvar  # N x R, R x N x N or N x R

def conditional(Xnew, X, kern, f, *, full_cov=False, q_sqrt=None, white=False, return_Lm=False):
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.
    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.
    Additionally, the GP may have been centered (whitened) so that
        p(v) = N(0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case `f` represents the values taken by v.
    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).
    We assume R independent GPs, represented by the columns of f (and the
    first dimension of q_sqrt).
    :param Xnew: datasets matrix, size N x D. Evaluate the GP at these new points
    :param X: datasets points, size M x D.
    :param kern: GPflow kernel.
    :param f: datasets matrix, M x R, representing the function values at X,
        for K functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
    :param white: boolean of whether to use the whitened representation as
        described above.
    :return:
        - mean:     N x R
        - variance: N x R (full_cov = False), R x N x N (full_cov = True)
    """
    num_data = X.size(0)  # M
    Kmm = kern.K(X) + torch.eye(num_data, dtype=torch.float64) * 1e-7
    Kmn = kern.K(X, Xnew)
    if full_cov:
        Knn = kern.K(Xnew)
    else:
        Knn = kern.Kdiag(Xnew)

    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white, return_Lm=return_Lm)  # N x R, N x R or R x N x N
