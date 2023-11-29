import torch
import torch.nn.functional as F

class Strauss(object):

    def __init__(self, gamma=0.5, R=0.5):
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.R = torch.tensor(R, dtype=torch.float64)

    def _euclid_dist(self, X):
        Xs = torch.sum(X**2, dim=-1, keepdim=True)
        dist = -2 * torch.matmul(X, X.transpose(-2, -1))
        dist += Xs + Xs.transpose(-2, -1)
        return torch.sqrt(torch.clamp(dist, min=1e-40))

    def _get_Sr(self, X):
        """
        Get the # elements in distance matrix dist that are < R
        """
        dist = self._euclid_dist(X)
        val = torch.where(dist <= self.R)
        Sr = val[0].shape[0]  # number of points satisfying the constraint above
        dim = dist.shape[0]
        Sr = (Sr - dim) / 2  # discounting diagonal and double counts
        return Sr

    def logp(self, X):
        return self._get_Sr(X) * torch.log(self.gamma)
