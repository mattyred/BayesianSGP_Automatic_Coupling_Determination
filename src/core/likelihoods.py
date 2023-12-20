import torch
import torch.nn as nn
from torch.nn import init
from . import quadrature
from . import densities
import numpy as np


class Likelihood(torch.nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive mean
           \int\int y p(y|f)q(f) df dy
        and the predictive variance
           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        gh_x, gh_w = quadrature.hermgauss(self.num_gauss_hermite_points)
        gh_w /= float(np.pi**0.5)
        gh_w = gh_w.reshape(-1, 1)
        shape = Fmu.size()
        Fmu = Fmu.view(-1,1)
        Fvar = Fvar.view(-1,1)
        X = gh_x[None, :] * (2.0 * Fvar)**0.5 + Fmu

        # here's the quadrature for the mean
        E_y = torch.matmul(self.conditional_mean(X), gh_w).view(shape)

        # here's the quadrature for the variance
        integrand = self.conditional_variance(X) + (self.conditional_mean(X))**2
        V_y = torch.matmul(integrand, gh_w).view(shape) - E_y**2

        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.
        i.e. if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive density
           \int p(y=Y|f)q(f) df
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = quadrature.hermgauss(self.num_gauss_hermite_points, dtype=Fmu.dtype)

        gh_w = gh_w.reshape(-1, 1) / float(np.sqrt(np.pi))
        shape = Fmu.size()
        Fmu, Fvar, Y = [e.view(-1, 1) for e in (Fmu, Fvar, Y)]
        X = gh_x * (2.0 * Fvar)**0.5 + Fmu
        Y = Y.expand(-1, self.num_gauss_hermite_points)  # broadcast Y to match X
        logp = self.logp(X, Y)
        return torch.matmul(logp.exp(), gh_w).view(*shape)

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes
           \int (\log p(y|f)) q(f) df.
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """

        gh_x, gh_w = quadrature.hermgauss(self.num_gauss_hermite_points, dtype=Fmu.dtype)
        gh_x = gh_x.view(1, -1)
        gh_w = gh_w.view(-1, 1) / float(np.pi)**0.5
        shape = Fmu.size()
        Fmu, Fvar, Y = [e.view(-1, 1) for e in (Fmu, Fvar, Y)]
        X = gh_x * (2.0 * Fvar)**0.5 + Fmu
        Y = Y.expand(-1, self.num_gauss_hermite_points)  # broadcast Y to match X
        logp = self.logp(X, Y)
        return torch.matmul(logp, gh_w).view(*shape)

    def _check_targets(self, Y_np):  # pylint: disable=R0201
        """
        Check that the Y values are valid for the likelihood.
        Y_np is a np array.
        The base class check is that the array has two dimensions
        and consists only of floats. The float requirement is so that AutoFlow
        can work with Model.predict_density.
        """
        if not Y.dim() == 2:
            raise ValueError('targets must be shape N x D')
        #if np.array(list(Y_np)).dtype != settings.np_float:
        #    raise ValueError('use {}, even for discrete variables'.format(settings.np_float))

class Gaussian(Likelihood):
    def __init__(self, variance=1.0, dtype=torch.float32):
        Likelihood.__init__(self)
        self.variance = nn.Parameter(torch.tensor([variance], dtype=dtype), requires_grad=True)

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance.get())

    def conditional_mean(self, F):
        return F

    def conditional_variance(self, F):
        return self.variance.get().expand_as(F)

    def predict_mean_and_var(self, Fmu, Fvar):
        return Fmu, Fvar + self.variance.get()

    def predict_density(self, Fmu, Fvar, Y):
        return densities.gaussian(Fmu, Y, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return (-0.5 * np.log(2 * np.pi) - 0.5 * torch.log(self.variance.get())
                - 0.5 * ((Y - Fmu)**2 + Fvar) / self.variance.get())
