import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from scipy.cluster.vq import kmeans2

from ..core.conditionals import conditional, conditional2
from ..misc.utils import get_all_files

class BGP(nn.Module):
 
    def __init__(self, X, Y, kernel, likelihood, inputs, outputs,
                 minibatch_size=100, prior_kernel=None, n_data=None, full_cov=False, prior_lik_var=0.05):
        super(BGP, self).__init__()
        self.kern = kernel
        self.likelihood = likelihood
        self.full_cov = full_cov
        self.inputs = inputs
        self.outputs = outputs
        self.minibatch_size = minibatch_size
        self.data_iter = 0
        self.prior_kernel = prior_kernel
        self.prior_lik_var = prior_lik_var
        self.X, self.Y = X, Y
        
        if n_data is None:
            self.N = X.shape[0]
        else:
            self.N = n_data

        self.Lm = None

    def mean_function(self, X):
        return torch.tensor(0, dtype=X.dtype, device=X.device)

    def conditional(self, Xnew):
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + torch.eye(self.X.size(0), dtype=self.X.dtype, device=self.X.device) * self.likelihood.variance.get()
        L = torch.cholesky(K, upper=False)

        A, _ = torch.solve(Kx, L)  # could use triangular solve, note gesv has B first, then A in AX=B
        V, _ = torch.solve(self.Y - self.mean_function(self.X), L) # could use triangular solve

        fmean = torch.mm(A.t(), V) + self.mean_function(Xnew)
        if self.full_cov:
            fvar = self.kern.K(Xnew) - torch.mm(A.t(), A)
            fvar = fvar.unsqueeze(2).expand(fvar.size(0), fvar.size(1), self.Y.size(1))
        else:
            fvar = self.kern.Kdiag(Xnew) - (A**2).sum(0)
            fvar = fvar.view(-1, 1)
            fvar = fvar.expand(fvar.size(0), self.Y.size(1))

        return fmean, fvar

    def predict(self, X):
        f_mean, f_var = self.conditional(X)
        y_mean, y_var = self.likelihood.predict_mean_and_var(f_mean, f_var)
        
        return y_mean, y_var

    def log_prior_hyper(self):
        log_lengthscales = torch.log(self.kern.lengthscales.get())
        log_variance = torch.log(self.kern.variance.get())
        log_lik_var = torch.log(self.likelihood.variance.get())

        log_prob = 0.
        log_prob += -torch.sum(torch.square(log_lengthscales - np.log(self.prior_lengthscale))) / 2.
        log_prob += -torch.sum(torch.square(log_variance - np.log(self.prior_variance))) / 2.
        log_prob += -torch.sum(torch.square(log_lik_var - np.log(self.prior_lik_var))) / 2.

        return log_prob

    def log_prior(self):
        return self.log_prior_hyper()

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
            return torch.tensor(self.X, dtype=torch.float64, device=device), torch.tensor(self.Y, dtype=torch.float64, device=device)

        if self.N < self.data_iter + self.minibatch_size:
            shuffle = np.random.permutation(self.N)
            self.X = self.X[shuffle, :]
            self.Y = self.Y[shuffle, :]
            self.data_iter = 0

        X_batch = self.X[self.data_iter:self.data_iter + self.minibatch_size, :]
        Y_batch = self.Y[self.data_iter:self.data_iter + self.minibatch_size, :]
        self.data_iter += self.minibatch_size
        return torch.tensor(X_batch, dtype=torch.float64, device=device), torch.tensor(Y_batch, dtype=torch.float64, device=device)
    
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

    @property
    def sampling_params(self):
        return list(self.parameters())[:-1]  # kernel.variance, kernel.lengthscales

    @property
    def optim_params(self):
        return list(self.parameters())[-1] # likelihood.variance
    
    @property
    def gp_params(self):
        return self.state_dict()
    
    @gp_params.setter
    def gp_params(self, params):
        self.load_state_dict(params)
