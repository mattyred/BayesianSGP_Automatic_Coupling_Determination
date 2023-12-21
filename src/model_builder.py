from src.models.bsgp2 import BSGP
from .core.kernels import RBF

from tqdm import tqdm
import torch
import numpy as np

from scipy.stats import norm
from scipy.special import logsumexp
from .core.likelihoods import Gaussian


def build_model(X, Y, args):
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
                n_data=N,
                n_inducing=args.num_inducing,
                inducing_points_init=None,
                full_cov=args.full_cov)

    return bsgp

def train(model, sampler, K):
    for _ in range(K):
        loss = model.train_step(sampler)
    return loss

def predict(model, X, S):
    ms, vs = [], []
    n = max(len(X) / 10000, 1) 
    for xs in np.array_split(X, n):
        m, v = model.predict_y(xs, S)
        ms.append(m)
        vs.append(v)

    return np.concatenate(ms, 1), np.concatenate(vs, 1) 

def compute_mnll(ms, vs, Y_test, num_posterior_samples=100, ystd=0.1):
    logps = norm.logpdf(np.repeat(Y_test[None, :, :]*ystd, num_posterior_samples, axis=0), ms*ystd, np.sqrt(vs)*ystd)
    return logsumexp(logps, axis=0) - np.log(num_posterior_samples)

