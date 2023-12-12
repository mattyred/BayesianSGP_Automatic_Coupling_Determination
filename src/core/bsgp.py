import torch
from .kernels import RBF
from src.misc import transforms
from .conditionals import conditional
from .priors import Strauss
from src.misc.settings import settings
from scipy.cluster.vq import kmeans2
import numpy as np
from tqdm import tqdm
import torch.nn as nn


jitter = 1e-5

def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_rand(x, full_cov=False):
    mean = x[0]
    var = x[1]
    if full_cov:
        chol = torch.linalg.cholesky(var + torch.eye(mean.size(0), dtype=torch.float64).unsqueeze(0) * 1e-7)
        rnd = torch.transpose(torch.squeeze(torch.matmul(chol, torch.randn(mean.t().size(), dtype=torch.float64).unsqueeze(2))))
        return mean + rnd
    return mean + torch.randn(mean.size(), dtype=torch.float64) * torch.sqrt(var)

    
class BSGP_Layer(torch.nn.Module): 
    def __init__(self, kern, outputs, num_inducing, fixed_mean, X, full_cov, prior_type="uniform"):
        super(BSGP_Layer, self).__init__()
        self.inputs, self.outputs, self.kernel = kern.D_in, outputs, kern
        self.M, self.fixed_mean = num_inducing, fixed_mean
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.X = X
        if prior_type == "strauss":
            self.pZ = Strauss(R=0.5)

        if len(X) > 1000000:
            perm = np.random.permutation(100000)
            X = X[perm]
        
        self.Z = torch.nn.Parameter(torch.tensor(kmeans2(X, self.M, minit='points')[0]), requires_grad=True) 
        
        if self.inputs == self.outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.outputs:
            self.mean = np.concatenate([np.eye(self.inputs), np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.outputs, :].T

        self.U = torch.nn.Parameter(torch.zeros((self.M, self.outputs)), requires_grad=True)
        # self.U = Param(np.random.randn(self.M, self.outputs), dtype=np.float64, requires_grad=False, name='Inducing values U')
        self.Lm = None

    def conditional(self, X):

        mean, var, self.Lm = conditional(X, self.Z, self.kernel, self.U, white=True, full_cov=self.full_cov, return_Lm=True)
        
        if self.fixed_mean:
            mean += torch.matmul(X, torch.cast(self.mean, torch.float64))
        return mean, var
    
    def prior_Z(self):
        if self.prior_type == "uniform":
            return 0.
        
        if self.prior_type == "normal":
            return -torch.reduce_sum(torch.square(self.Z)) / 2.0
            
        if self.prior_type == "strauss":
            return self.pZ.logp(self.Z)

        #if self.Lm is not None: # determinantal;
        if self.prior_type == "determinantal":
            self.Lm = torch.cholesky(self.kernel.K(self.Z) + torch.eye(self.M, dtype=torch.float64) * 1e-7)
            pZ = torch.sum(torch.log(torch.square(torch.diagonal(self.Lm))))
            return pZ
        else: #
            raise Exception("Invalid prior type")
        
    def prior_hyper(self):
        return -torch.sum(torch.square(self.kernel.lengthscales)) / 2.0 - torch.sum(torch.square(self.kernel.variance - torch.log(torch.tensor(0.05)))) / 2.0

    def prior(self):
        return -torch.sum(torch.square(self.U)) / 2.0 + self.prior_hyper() + self.prior_Z()

    

class BSGP(torch.nn.Module):
    def __init__(self, X=None, Y=None, num_inducing=100, kernels=[], lik=None, minibatch_size=100, window_size=64, output_dim=None, adam_lr=0.01, prior_inducing_type="uniform", full_cov=False, epsilon=0.01, mdecay=0.05):
        super(BSGP, self).__init__()
        self.n_inducing = num_inducing
        self.kernels = kernels
        self.likelihood = lik
        self.minibatch_size = minibatch_size
        self.window = []
        self.window_size = window_size
        self.posterior_samples = []
        self.epsilon, self.mdecay = epsilon, mdecay

        self.rand = lambda x: get_rand(x, full_cov)
        self.output_dim = output_dim or Y.shape[1]

        n_layers = len(kernels)
        self.N = X.shape[0]

        self.layers = []
        X_running = X # it should be X.clone()
        for l in range(n_layers):
            outputs = self.kernels[l+1].D_in if l+1 < n_layers else self.output_dim
            self.layers.append(BSGP_Layer(self.kernels[l], 
                                          outputs, 
                                          num_inducing, 
                                          fixed_mean=(l+1 < n_layers),
                                          X=X_running, 
                                          full_cov=full_cov if l+1<n_layers else False, 
                                          prior_type=prior_inducing_type))
            X_running = np.matmul(X_running, self.layers[-1].mean)

        # model parameters (trainable)
        self.bsgp_parameters = nn.ParameterList()
        for l in self.layers:
            self.bsgp_parameters.extend([l.U, l.Z, l.kernel.loglengthscales, l.kernel.logvariance])

        # sampler parameters (non-trainable)
        self.sampler_parameters = []
        for theta in self.parameters():
            self.xi = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            self.g = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            self.g2 = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            self.p = torch.nn.Parameter(torch.zeros_like(theta), requires_grad=False)
            self.sampler_parameters.append({'xi': self.xi, 'g': self.g, 'g2': self.g2, 'p': self.p})

        set_seed()
    
    def parameters(self, recurse=True):
        return iter(self.bsgp_parameters)
    
    def forward(self, X):
        self.f, self.fmeans, self.fvars = self.propagate(X)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])
        return self.y_mean, self.y_var
    
    def fit(self, X, Y):
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        self.X_batch_size = X.size(0)

        self.f, self.fmeans, self.fvars = self.propagate(X)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])
        self.prior = sum([l.prior() for l in self.layers])
        self.log_likelihood = self.likelihood.predict_density(self.fmeans[-1], self.fvars[-1], Y)
        nll = -torch.sum(self.log_likelihood) / self.X_batch_size - (self.prior / self.N)
        return nll
    
    def train_step(self, sampler):
        sampler.zero_grad()
        X_batch, Y_batch = self.get_minibatch()
        nll = self.fit(X_batch, Y_batch)
        nll.backward(retain_graph=True)
        sampler.step(self.sampler_parameters, burn_in=True)   
        for _ in range(10):
            sampler.zero_grad()
            X_batch, Y_batch = self.get_minibatch()
            nll = self.fit(X_batch, Y_batch)
            nll.backward(retain_graph=True)
            sampler.step(self.sampler_parameters, burn_in=True)
            sampler.step(self.sampler_parameters, burn_in=False)  

        with torch.no_grad():
            values = [p.detach().cpu().numpy() for p in self.parameters()]
            variables_names = ['U', 'Z', 'kernel_lengthscales', 'kernel_variance']
            sample = {}
            for var, value in zip(variables_names, values):
                sample[var] = value
            self.window.append(sample)
            if len(self.window) > self.window_size:
                self.window = self.window[-self.window_size:] 

    def print_sample_performance(self, posterior=False):
        X_batch, Y_batch = self.get_minibatch()
        #if posterior:
        #   feed_dict.update(np.random.choice(self.posterior_samples))
        nll = self.fit(X_batch, Y_batch)
        return -nll

    def collect_samples(self, sampler, num=256, spacing=32, progress=False):
        self.posterior_samples = []
        r = tqdm(range(num)) if progress else range(num)
        for i in r:
            for j in range(spacing):
                sampler.zero_grad()
                X_batch, Y_batch = self.get_minibatch()
                nll = self.fit(X_batch, Y_batch)
                nll.backward(retain_graph=True)
                sampler.step(self.sampler_parameters, burn_in=False)  

            with torch.no_grad():
                values = [p.detach().cpu().numpy() for p in self.parameters()]
                variables_names = ['U', 'Z', 'kernel_lengthscales', 'kernel_variance']
                sample = {}
                for var, value in zip(variables_names, values):
                    sample[var] = value
                self.posterior_samples.append(sample)

    def predict_y(self, X, S, posterior=True):
        X = torch.tensor(X, dtype=torch.float32)
        # assert S <= len(self.posterior_samples)
        self.eval()
        ms, vs = [], []
        with torch.no_grad():
            for i in range(S):
                self.load_posterior_sample(self.posterior_samples[i]) if posterior else self.load_posterior_sample(self.window[-(i+1)])
                m, v = self.forward(X)
                ms.append(m.detach().cpu().numpy())
                vs.append(v.detach().cpu().numpy())
        return np.stack(ms, 0), np.stack(vs, 0)

    def load_posterior_sample(self, sample):
        # load model.variables with posterior sample values
        for param, value in zip(self.parameters(), sample.values()):
            param.data = torch.tensor(value) # TODO: insert a torch.no_grad()

    def propagate(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for l, layer in enumerate(self.layers):
            mean, var = layer.conditional(Fs[-1])
            # eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            # F = mean + eps * tf.sqrt(var)
            if l+1 < len(self.layers):
                F = self.rand([mean, var])
            else:
                F = get_rand([mean, var], False)
                
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars
    
    def reset(self, X, Y):
        self.X, self.Y, self.N = X, Y, X.shape[0]
        self.data_iter = 0

    def get_minibatch(self):
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

