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


def sghmc_sampling(model):
    X_batch, Y_batch = model.get_minibatch()
    model.forward(X_batch)
    model.sghmc_step(Y_batch, burn_in=True)
    for _ in range(10):
        X_batch, Y_batch = model.get_minibatch()
        model.forward(X_batch)
        model.sghmc_step(Y_batch, burn_in=True)
        model.forward(X_batch)
        model.sghmc_step(Y_batch, burn_in=False)

    values = [var.detach().cpu().numpy() for var in model.variables]
    variables_names = ['U', 'Z', 'kernel_lengthscales', 'kernel_variance']
    sample = {}
    for var, value in zip(variables_names, values):
        sample[var] = value
    model.window.append(sample)
    if len(model.window) > model.window_size:
        model.window = model.window[-model.window_size:]
    return sample

def print_sample_performance(model, posterior=False):
    X_batch, Y_batch = model.get_minibatch()
    #if posterior:
    #    feed_dict.update(np.random.choice(self.posterior_samples))
    model.forward(X_batch)
    marginal_ll = -model.compute_nll(Y_batch)
    return marginal_ll  


def collect_samples(model, num, spacing, progress=False):
    model.posterior_samples = []
    r = tqdm(range(num)) if progress else range(num)
    for i in r:
        for j in range(spacing):
            X_batch, Y_batch = model.get_minibatch()
            model.forward(X_batch)
            model.sghmc_step(Y_batch, burn_in=False)

        values = [var.data for var in model.variables]
        variables_names = ['U', 'Z', 'kernel_lengthscales', 'kernel_variance']
        sample = {}
        for var, value in zip(variables_names, values):
            sample[var] = value
        model.posterior_samples.append(sample)

def predict_y(model, X, S, posterior=True):
     # assert S <= len(self.posterior_samples)
    model.eval()
    ms, vs = [], []
    for i in range(S):
        #feed_dict.update(self.posterior_samples[i]) if posterior else feed_dict.update(self.window[-(i+1)])
        model.load_posterior_sample(model.posterior_samples[i])
        model.forward(X)
        m, v = model.y_mean, model.y_var
        ms.append(m.detach().cpu().numpy())
        vs.append(v.detach().cpu().numpy())
    return np.stack(ms, 0), np.stack(vs, 0)

def predict(model, X, S):
    ms, vs = [], []
    n = max(len(X) / 10000, 1) 
    for xs in np.array_split(X, n):
        m, v = predict_y(model, xs, S)
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