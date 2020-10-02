import math
import torch
import torch.nn.functional as F

def prior_categorical(batch_size, y_dim, device):
    # Uniform prior over y
    prior = torch.ones((batch_size, y_dim)).to(device)
    prior = F.softmax(prior, dim=1)
    prior.requires_grad = False
    return prior


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_standard_categorical(p, eps):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.

    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = 0.5 * torch.ones_like(p).to(p.get_device())
    #prior = F.softmax(prior, dim=1)
    # prior = F.sigmoid(prior)
    prior.requires_grad = False

    #cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    cross_entropy = -torch.sum((p * torch.log(prior + eps) + (1-p) * torch.log(1 - prior + eps)), dim=1)

    return cross_entropy