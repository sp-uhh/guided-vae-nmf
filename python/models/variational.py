from itertools import repeat

import torch
from torch import nn
import torch.nn.functional as F

from python.models.utils import log_sum_exp, enumerate_discrete
from python.models.distributions import log_standard_categorical

class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in [Sønderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler, eps=1e-8):
        """
        Initialises a new SVI optimizer for semi-supervised learning.

        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.beta = beta
        self.sampler = sampler
        self.eps = eps

    def forward(self, x, y=None):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)

        # # Increase sampling dimension
        # xs = self.sampler.resample(xs)
        # ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -self.likelihood(reconstruction, xs, self.eps)

        # p(y)
        # not needed if y follow binary distribution
        prior = -log_standard_categorical(ys, self.eps)

        # Equivalent to -L(x, y)
        #TODO: what is the beta (repeat)?
        #elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        elbo = likelihood + prior - self.model.kl_divergence
        # elbo = likelihood - self.model.kl_divergence

        
        # L = self.sampler(elbo)
        L = elbo

        if is_labelled:
            #return torch.mean(L)
            return [-torch.mean(L), -torch.mean(likelihood),\
                 -torch.mean(prior), torch.mean(self.model.kl_divergence)]

        logits = self.model.classify(x)

        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        return torch.mean(U)

class SVI_M1(nn.Module):
    """
    Stochastic variational inference (SVI) for model M1.
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler, eps=1e-8):
        """
        Initialises a new SVI optimizer for semi-supervised learning.

        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI_M1, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.beta = beta
        self.sampler = sampler
        self.eps = eps

    def forward(self, x):

        # Prepare for sampling
        #xs = torch.sqrt(x)
        xs = x

        # # Increase sampling dimension
        # xs = self.sampler.resample(xs)
        # ys = self.sampler.resample(ys)

        reconstruction = self.model(xs)

        # p(x|y,z)
        likelihood = -self.likelihood(reconstruction, xs, self.eps)

        # Equivalent to -L(x, y)
        elbo = likelihood - self.model.kl_divergence
        
        # L = self.sampler(elbo)
        L = elbo

        return [-torch.mean(L), -torch.mean(likelihood), torch.mean(self.model.kl_divergence)]

