import torch
from .kernels import RBF
from src.misc.param import Param
from src.misc import transforms
from .conditionals import conditional
from .priors import Strauss
from scipy.cluster.vq import kmeans2
import numpy as np
import torch.nn as nn


jitter = 1e-5

def get_rand(x, full_cov=False):
    mean = x[0]
    var = x[1]
    if full_cov:
        chol = torch.cholesky(var + torch.eye(mean.size(0), dtype=torch.float64).unsqueeze(0) * 1e-7)
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

        self.Z = Param(kmeans2(X, self.M, minit='points')[0], requires_grad=False, transform=transforms.SoftPlus(), name='Inducing locations Z') 
        
        if self.inputs == self.outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.D_out:
            self.mean = np.concatenate([np.eye(self.inputs), np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.outputs, :].T

        self.U = Param(np.zeros((self.M, self.outputs)), requires_grad=False, name='Inducing values U')
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
        return -torch.sum(torch.square(self.kernel.lengthscales)) / 2.0 - torch.sum(torch.square(self.kernel.variance - torch.log(0.05))) / 2.0

    def prior(self):
        return -torch.sum(torch.square(self.U)) / 2.0 + self.prior_hyper() + self.prior_Z()

    

class BSGP(torch.nn.Module):
    def __init__(self, X=None, Y=None, num_inducing=100, kernels=[], lik=None, minibatch_size=100, window_size=64, output_dim=None, adam_lr=0.01, prior_inducing_type="uniform", full_cov=False, epsilon=0.01, mdecay=0.05):
        super(BSGP, self).__init__()
        self.n_inducing = num_inducing
        self.kernels = kernels
        self.likelihood = lik
        self.minibatch_size = minibatch_size
        self.window_size = window_size

        self.rand = lambda x: get_rand(x, full_cov)
        self.output_dim = output_dim or Y.shape[1]

        n_layers = len(kernels)
        N = X.shape[0]

        self.layers = []
        X_running = X.copy()
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

        variables = []
        for l in self.layers:
            variables += [l.U, l.Z, l.kernel.lengthscales, l.kernel.variance]

        self.f, self.fmeans, self.fvars = self.propagate(X_running)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])

        self.prior = sum([l.prior() for l in self.layers])
        self.log_likelihood = self.likelihood.predict_density(self.fmeans[-1], self.fvars[-1], self.Y)

        self.nll = -torch.sum(self.log_likelihood) / float(self.X_placeholder.size(0)) - (self.prior / self.N)

        self.generate_update_step(self.nll, epsilon, mdecay)


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
    
    def generate_update_step(self, mdecay, epsilon):
        self.epsilon = epsilon
        burn_in_updates = []
        sample_updates = []

        grads = torch.autograd.grad(self.nll, self.vars, create_graph=True)

        for theta, grad in zip(self.vars, grads):
            xi = nn.Parameter(torch.ones_like(theta, dtype=torch.float64), requires_grad=False)
            g = nn.Parameter(torch.ones_like(theta, dtype=torch.float64), requires_grad=False)
            g2 = nn.Parameter(torch.ones_like(theta, dtype=torch.float64), requires_grad=False)
            p = nn.Parameter(torch.zeros_like(theta, dtype=torch.float64), requires_grad=False)

            r_t = 1. / (xi + 1.)
            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad ** 2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (torch.sqrt(g2 + 1e-16) + 1e-16)

            burn_in_updates.extend([(xi, xi_t), (g, g_t), (g2, g2_t)])

            epsilon_scaled = epsilon / torch.sqrt(torch.tensor(self.N, dtype=torch.float64))
            noise_scale = 2. * epsilon_scaled ** 2 * mdecay * Minv
            sigma = torch.sqrt(torch.maximum(noise_scale, torch.tensor(1e-16, dtype=torch.float64)))
            sample_t = torch.distributions.normal.Normal(torch.zeros_like(theta), sigma).sample()
            p_t = p - epsilon ** 2 * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t

            sample_updates.extend([(theta, theta_t), (p, p_t)])

        self.sample_op = sample_updates
        self.burn_in_op = burn_in_updates + sample_updates