from .models.bsgp import BSGP
from .models.bgp import BGP
from .core.kernels import RBF

from tqdm import tqdm
import torch
import numpy as np

from scipy.stats import norm
from scipy.special import logsumexp
from .core.likelihoods import Gaussian, Bernoulli


def build_model(X, Y, args, model='BSGP', task='regression'):
    assert model == 'BSGP' or model == 'BGP'
    assert task == 'regression' or 'classification' 

    N, D_in, D_out = X.shape[0], X.shape[1], Y.shape[1]

    # define likelihood
    if task == 'regression':
        lik = Gaussian(dtype=torch.float64)
    elif task == 'classification':
        lik = Bernoulli()

    # define kernel
    if args.kernel_type == 'ACD':
        kern = RBF(input_dim=D_in, ACD=True)
    elif args.kernel_type == 'ARD':
        kern = RBF(input_dim=D_in, ARD=True)

    # define model
    mb_size = args.minibatch_size if N > args.minibatch_size else N
    if model == 'BSGP':
        model = BSGP(X=X, Y=Y,
                    kernel=kern,
                    likelihood=lik,
                    prior_type=args.prior_inducing_type,
                    prior_kernel=args.prior_kernel,
                    inputs=D_in,
                    outputs=D_out,
                    minibatch_size=mb_size,
                    n_data=N,
                    n_inducing=args.num_inducing,
                    inducing_points_init=None,
                    full_cov=args.full_cov)
    elif model == 'BGP':
        model = BGP(X=X, Y=Y,
                kernel=kern,
                likelihood=lik,
                inputs=D_in,
                outputs=D_out,
                minibatch_size=mb_size,
                prior_kernel=args.prior_kernel,
                n_data=N,
                full_cov=args.full_cov)

    return  model

def build_bsgp_model(X, Y, args):
    
    N, D_in, D_out = X.shape[0], X.shape[1], Y.shape[1]

    # define likelihood
    lik = Gaussian(dtype=torch.float64)

    # define kernel
    if args.kernel_type == 'ACD':
        kern = RBF(input_dim=D_in, ACD=True)
    else:
        kern = RBF(input_dim=D_in, ARD=True)

    mb_size = args.minibatch_size if N > args.minibatch_size else N
    bsgp = BSGP(X=X, Y=Y,
                kernel=kern,
                likelihood=lik,
                prior_type=args.prior_inducing_type,
                prior_kernel=args.prior_kernel,
                inputs=D_in,
                outputs=D_out,
                minibatch_size=mb_size,
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
    if args.kernel_type == 'ACD':
        kern = RBF(input_dim=D_in, ACD=True)
    else:
        kern = RBF(input_dim=D_in, ARD=True)

    mb_size = args.minibatch_size if N > args.minibatch_size else N
    bgp = BGP(X=X, Y=Y,
                kernel=kern,
                likelihood=lik,
                inputs=D_in,
                outputs=D_out,
                minibatch_size=mb_size,
                prior_kernel=args.prior_kernel,
                n_data=N,
                full_cov=args.full_cov)

    return bgp


def compute_mnll(ms, vs, Y, num_posterior_samples=100, ystd=0.1):
    with torch.no_grad():
        logps = norm.logpdf(np.repeat(Y[None, :, :]*ystd, num_posterior_samples, axis=0), ms*ystd, np.sqrt(vs)*ystd)
        mnll = -np.mean(logsumexp(logps, axis=0) - np.log(num_posterior_samples))
        return mnll

def compute_accuracy(ms, vs, Y_true, num_posterior_samples=100, ystd=0.1):
    with torch.no_grad():
        Y_pred = (ms >= 0.5).mean(0).reshape(-1)
        accuracy = np.sum(Y_pred == Y_true.numpy().reshape(-1)) / len(Y_pred)
        return accuracy