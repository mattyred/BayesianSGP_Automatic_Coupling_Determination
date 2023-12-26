import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from scipy.cluster.vq import kmeans2

from ..core.conditionals import conditional, conditional2
from ..misc.utils import get_all_files
from ..core.densities import *

class Strauss(nn.Module):
    
    def __init__(self, gamma=0.5, R=0.5):
        super(Strauss, self).__init__()
        self.gamma = gamma
        self.R = R

    def _euclid_dist(self, X):
        Xs = torch.sum(torch.square(X), dims=-1, keepdims=True)
        dist = -2 * X @ X.T
        dist += Xs + Xs.T

        return torch.sqrt(torch.maximum(dist, 1e-40))
    
    def _get_Sr(self, X):
        """
        Get the # elements in distance matrix dist that are < R
        """
        dist = self._euclid_dist(X)
        val = torch.where(dist <= self.R)
        Sr = val.shape[0] # number of points satisfying the constraint above
        dim = dist.shape[0]
        Sr = (Sr - dim)/2  # discounting diagonal and double counts
        return Sr

    def log_prob(self, X):
        return self._get_Sr(X) * np.log(self.gamma)

def logdet_jacobian(kernel, eps=1e-6):
    l_matrix = kernel.l_matrix
    n = l_matrix.size(0)
    diag_l = torch.diagonal(l_matrix) 
    exps = torch.tensor(np.flip(np.arange(0, n) + 1.).copy(), dtype=l_matrix.dtype)
    return n * np.log(2.) + torch.sum(torch.mul(exps, torch.log(torch.abs(diag_l)))) 


class BSGP(nn.Module):
 
    def __init__(self, X, Y, kernel, likelihood, prior_type, inputs, outputs,
                 minibatch_size=100, window_size=64, n_data=None, n_inducing=None, inducing_points_init=None, full_cov=False, prior_kernel=None, prior_lik_var=0.05):
        super(BSGP, self).__init__()
        self.kern = kernel
        self.likelihood = likelihood
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.inputs = inputs
        self.outputs = outputs
        self.minibatch_size = minibatch_size
        self.data_iter = 0
        self.prior_kernel = prior_kernel
        self.prior_lik_var = prior_lik_var
        self.X, self.Y = X, Y
        # samples window
        self.window = []
        self.window_size = window_size
        
        if n_data is None:
            self.N = X.shape[0]
        else:
            self.N = n_data
        self.M = n_inducing
        if inducing_points_init is not None:
            self.M = inducing_points_init.shape[0]
        else:
            if torch.is_tensor(X):
                X = X.cpu().numpy()

            inducing_points_init = torch.tensor(
                kmeans2(X, self.M, minit='points')[0], dtype=torch.float64)

        if prior_type == "strauss":
            self.pZ = Strauss(R=0.5)
        
        self.Z = nn.parameter.Parameter(
            inducing_points_init.double(),
            requires_grad=True)

        self.U = nn.parameter.Parameter(
            torch.zeros((self.M, self.outputs), dtype=torch.float64),
            requires_grad=True)

        self.Lm = None

    def log_prior_Z(self):
        if self.prior_type == "uniform":
            return 0.
        
        if self.prior_type == "normal":
            return -torch.sum(torch.square(self.Z)) / 2.0
        
        if self.prior_type == "strauss":
            return self.pZ.log_prob(self.Z)

        #if self.Lm is not None: # determinantal;
        if self.prior_type == "determinantal":
            self.Lm = torch.cholesky(self.kern.K(self.Z) + torch.eye(self.M, dtype=torch.float64, device=self.Z.device) * 1e-7)
            log_prob = torch.sum(torch.log(torch.square(torch.diagonal(self.Lm))))
            return log_prob
        
        else:
            raise Exception("Invalid prior type")

    def conditional(self, X):
        mean, var, self.Lm = conditional(X, self.Z, self.kern, self.U,
                                         whiten=True, full_cov=self.full_cov,
                                         return_Lm=True)
        return mean, var

    def predict(self, X):
        f_mean, f_var = self.conditional(X)
        y_mean, y_var = self.likelihood.predict_mean_and_var(f_mean, f_var)
        
        return y_mean, y_var

    def log_prior_hyper(self):
        log_prob = 0.

        # prior on kernel precision
        if self.kern.rbf_type == 'ACD':
            prior_precision_type = self.prior_kernel['type']
            L = self.kern.L.get()
            precision = self.kern.precision # Λ
            diag_precision = torch.diagonal(precision) # diag(Λ)
            offdiag_precision = precision[~torch.eye(precision.size(0), dtype=torch.bool)].view(precision.size(0), -1) # Λ_

            if prior_precision_type == 'laplace':
                # Laplace(Λ_|0,b) + N(diag(Λ)|0,1)
                b = torch.tensor(self.prior_kernel['b'], dtype=precision.dtype)
                log_prob += loglaplace(offdiag_precision, b)
                log_prob += lognormal(diag_precision, mu=0., var=1.)
            elif prior_precision_type == 'wishart':
                # W(Λ|0,)
                log_prob += logwishart(L, precision)
            elif prior_precision_type == 'invwishart':
                # IW(Λ|0,b)
                log_prob += loginvwishart(L, precision)
            elif prior_precision_type == 'normal':
                # N(Λ|m,v) 
                m = torch.tensor(self.prior_kernel['m'], dtype=precision.dtype)
                v = torch.tensor(self.prior_kernel['v'], dtype=precision.dtype)
                log_prob += lognormal(precision, mu=m, var=v)
            elif prior_precision_type == 'horseshoe':
                # HS(Λ_|scale) + N(diag(Λ)|0,1)
                scale = torch.tensor(self.prior_kernel['global_shrinkage'], dtype=precision.dtype)
                log_prob += loghorseshoe(offdiag_precision, scale)
                log_prob += lognormal(diag_precision, mu=0., var=1.)
            
            
            log_prob += logdet_jacobian(self.kern)

        # prior on kernel log-lengthscales
        else:
            log_lengthscales = torch.log(self.kern.lengthscales.get())
            log_prob += -torch.sum(torch.square(log_lengthscales)) / 2.

        # prior on kernel log-variance
        log_variance = torch.log(self.kern.variance.get())
        log_prob += -torch.sum(torch.square(log_variance - np.log(0.05))) / 2.

        return log_prob

    def log_prior_U(self):
        return -torch.sum(torch.square(self.U)) / 2.0

    def log_prior(self):
        return self.log_prior_U() + self.log_prior_Z() + self.log_prior_hyper()

    def log_likelihood(self, X, Y):
        f_mean, f_var = self.conditional(X)
        log_likelihood = torch.sum(self.likelihood.predict_density(f_mean, f_var, Y))

        return log_likelihood

    def log_prob(self, X, Y):
        log_likelihood = self.log_likelihood(X, Y)
        log_prior = self.log_prior()

        batch_size = X.shape[0]

        log_prob = (self.N / batch_size) * log_likelihood + log_prior

        return log_prob

    def train_step(self, device, sampler, K=10):
        for k in range(K):
            X_batch, Y_batch = self.get_minibatch(device)
            log_prob = self.log_prob(X_batch, Y_batch)
            self.zero_grad()
            loss = -log_prob
            loss.backward()
            sampler.step()
        return log_prob

    def optimizer_step(self, device, optimizer):
        X_batch, Y_batch = self.get_minibatch(device)
        log_prob = self.log_prob(X_batch, Y_batch)
        optimizer.zero_grad()
        loss = -log_prob
        loss.backward()
        optimizer.step()
        return log_prob
    
    def predict_y(self, X):
        S = len(self.gp_samples)
        ms, vs = [], []
        for i in range(S):
            gp_params = self.load_samples(i)
            self.gp_params = gp_params
            y_mean, y_var = self.predict(X)
            ms.append(y_mean.detach())
            vs.append(y_var.detach())
        return np.stack(ms, 0), np.stack(vs, 0)
    
    def get_minibatch(self, device):
        assert self.N >= self.minibatch_size
        if self.N == self.minibatch_size:
            return self.X, self.Y

        if self.N < self.data_iter + self.minibatch_size:
            shuffle = np.random.permutation(self.N)
            self.X = self.X[shuffle, :]
            self.Y = self.Y[shuffle, :]
            self.data_iter = 0

        X_batch = self.X[self.data_iter:self.data_iter + self.minibatch_size, :]
        Y_batch = self.Y[self.data_iter:self.data_iter + self.minibatch_size, :]
        self.data_iter += self.minibatch_size
        return X_batch, Y_batch
    
    def save_sample(self, sample_dir, idx):
        torch.save(self.gp_params,
            os.path.join(sample_dir, "gp_{:03d}.pt".format(idx)))
            
    def set_samples(self, sample_dir, cache=False):
        gp_files = get_all_files(os.path.join(sample_dir, "gp*"))

        if cache:
            self.gp_samples = []
            for i in range(len(gp_files)):
                self.gp_samples.append(
                    torch.load(gp_files[i]))
        else:
            self.gp_samples = gp_files
        
        self.loaded_samples = cache

    def load_samples(self, idx):
        gp_params = None

        if self.loaded_samples:
            gp_params = self.gp_samples[idx]
        else:
            gp_params = torch.load(self.gp_samples[idx])

        return gp_params
    
    def __str__(self):
        str = [
            ' BSGP',
            ' Input dim = %d' % self.X.size(0),
            ' Output dim = %d' % self.X.size(1),
            ' Gradient clipping = %d' % False,
            ' Prior kernel type = %s' % self.prior_kernel
            ]
        return '\n'.join(str)
    
    @property
    def sampling_params(self):
        return list(self.parameters())[:-1]  # U, Z, kernel.variance, kernel.lengthscales

    @property
    def optim_params(self):
        return list(self.parameters())[-1] # likelihood.variance
    
    @property
    def gp_params(self):
        return self.state_dict()
    
    @gp_params.setter
    def gp_params(self, params):
        self.load_state_dict(params)
