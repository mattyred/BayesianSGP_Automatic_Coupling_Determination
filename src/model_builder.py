from src.core.bsgp import BSGP
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
    N, D = X.shape[0], X.shape[1]

    # define likelihood
    lik = Gaussian(ndim = D)

    # define kernels
    kernels = []
    for i in range(args.n_layers):
        D_out = 196 if i >= 1 and D > 700 else D
        kernels.append(RBF(D_in=D, D_out=D_out, dimwise=False))

    gp = BSGP(X=X, Y=Y, 
              num_inducing = args.num_inducing,
              kernels = kernels, 
              lik = lik, 
              minibatch_size = args.minibatch_size, 
              window_size = args.window_size, 
              output_dim= args.output_dim, 
              adam_lr = args.adam_lr, 
              prior_inducing_type = args.prior_inducing_type, 
              full_cov = args.full_cov, 
              epsilon = args.epsilon, 
              mdecay = args.mdecay)

    return gp


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

def compute_inducing_variables_for_plotting(model):
    """
    Uniwhiten the inducing variables for generating plots
    @param model:  a gpode.SequenceModel object
    @return: inducing values, inducing locations
    """
    z = model.flow.odefunc.diffeq.inducing_loc().clone().detach()
    u = model.flow.odefunc.diffeq.Um().clone().detach()
    Ku = model.flow.odefunc.diffeq.kern.K(model.flow.odefunc.diffeq.inducing_loc())  # MxM or DxMxM
    Lu = torch.cholesky(Ku + torch.eye(Ku.shape[1]) * 1e-5)  # MxM or DxMxM
    if model.flow.odefunc.diffeq.dimwise:
        u = torch.einsum('mde, dnm -> nde', u.unsqueeze(2), Lu).squeeze(2)  # DxMx1
    else:
        u = torch.einsum('md, mn -> nd', u, Lu.T)  # NxD
    #u = torch_utils.torch2numpy(u) / 1.5
    #z = torch_utils.torch2numpy(z)
    return u, z