from .models.bsgp import BSGP
from .models.bgp import BGP
from .core.kernels import RBF

from tqdm import tqdm
import torch
import numpy as np

from scipy.stats import norm
from scipy.special import logsumexp
from .core.likelihoods import Gaussian


def build_bsgp_model(X, Y, args):
    """
    Builds a model object of gpode.SequenceModel based on the MoCap experimental setup

    @param data_full_ys: data sequence in observed space (N,T,D_full)
    @param latent2data_projection: an object of misc.mocap_utils.Latent2DataProjector class
    @param args: model setup arguments
    @return: an object of gpode.SequenceModel class
    """
    N, D_in, D_out = X.shape[0], X.shape[1], Y.shape[1]

    # define likelihood
    lik = Gaussian(dtype=torch.float64)

    # define kernel
    kern = RBF(input_dim=D_in, ARD=True)

    mb_size = args.minibatch_size if N > args.minibatch_size else N
    bsgp = BSGP(X=X, Y=Y,
                kernel=kern,
                likelihood=lik,
                prior_type=args.prior_inducing_type,
                inputs=D_in,
                outputs=D_out,
                minibatch_size=mb_size,
                window_size=args.window_size,
                n_data=N,
                n_inducing=args.num_inducing,
                inducing_points_init=None,
                full_cov=args.full_cov)

    return bsgp

def build_bgp_model(X, Y, args):
    N, D_in, D_out = X.shape[0], X.shape[1], Y.shape[1]

    # define likelihood
    lik = Gaussian(dtype=torch.float64)

    # define kernel
    kern = RBF(input_dim=D_in, ARD=True)

    mb_size = args.minibatch_size if N > args.minibatch_size else N
    bgp = BGP(X=X, Y=Y,
                kernel=kern,
                likelihood=lik,
                inputs=D_in,
                outputs=D_out,
                minibatch_size=mb_size,
                n_data=N,
                full_cov=args.full_cov)

    return bgp


def compute_mnll(ms, vs, Y, num_posterior_samples=100, ystd=0.1):
    with torch.no_grad():
        logps = norm.logpdf(np.repeat(Y[None, :, :]*ystd, num_posterior_samples, axis=0), ms*ystd, np.sqrt(vs)*ystd)
        mnll = -np.mean(logsumexp(logps, axis=0) - np.log(num_posterior_samples))
        return mnll

