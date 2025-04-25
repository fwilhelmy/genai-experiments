"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch

torch.manual_seed(42)

def log_likelihood_bernoulli(mu, target): # Implemented
    """
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    eps = 1e-8
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1) # flatten the mu
    mu = torch.clamp(mu, eps, 1-eps) # for numerical stability
    target = target.view(batch_size, -1) # flatten the target

    # Compute log likelihood of Bernoulli
    ll_bernoulli = target * torch.log(mu) + (1 - target) * torch.log(1 - mu)

    return ll_bernoulli.sum(dim=1) # sum over the input size

def log_likelihood_normal(mu, logvar, z): # Implemented
    """
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # Term 1: D x log(2π)
    D = mu.view(batch_size, -1).size(1)
    c = D * math.log(2 * math.pi)

    # Term 2: log(var)
    variance = torch.sum(logvar, dim=1)

    # Term 3: (z - mu)^2 / (var)
    sq_term  = torch.sum(((z - mu) ** 2) * torch.exp(-logvar), dim=1)

    # Log likelihood of normal distribution
    ll_normal = -0.5 * (c + variance + sq_term )
    
    return ll_normal

def log_mean_exp(y): # Implemented
    """
    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    a = y.max(dim=1, keepdim=True)[0] # Row-wise max
    shifted_mean = torch.exp(y - a).mean(dim=1, keepdim=True)
    lme = a + torch.log(shifted_mean)

    return lme.squeeze(1) # (batch_size,)


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p): # Implemented
    """ 
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)

    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    t1 = logvar_p - logvar_q
    t2 = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)

    # Sum over the input size
    kl_gg = 0.5 * (t1 + t2 - 1).sum(dim=1)

    return kl_gg

def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1): # Implemented
    """ 
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    B, D = mu_q.size(0), mu_q.view(mu_q.size(0), -1).size(1)

    # Expand the tensors to (B, num_samples, D)
    mu_q = mu_q.view(B, 1, D).expand(B, num_samples, D)
    logvar_q = logvar_q.view(B, 1, D).expand(B, num_samples, D)
    mu_p = mu_p.view(B, 1, D).expand(B, num_samples, D)
    logvar_p = logvar_p.view(B, 1, D).expand(B, num_samples, D)

    # Sampling from q
    eps = torch.randn(B, num_samples, D) # (B, num_samples, D)
    z = mu_q + torch.exp(0.5 * logvar_q) * eps # (B, num_samples, D)

    # Flatten the tensors to (B * num_samples, D) for easier computation
    flat_mu_q = mu_q.flatten(0,1)
    flat_logvar_q = logvar_q.flatten(0,1)
    flat_mu_p = mu_p.flatten(0,1)
    flat_logvar_p = logvar_p.flatten(0,1)
    flat_z = z.flatten(0,1)

    # Compute log likelihoods
    flat_ll_q = log_likelihood_normal(flat_mu_q, flat_logvar_q, flat_z) # Log likelihood of q(z)
    flat_ll_p = log_likelihood_normal(flat_mu_p, flat_logvar_p, flat_z) # Log likelihood of p(z)

    # Unflatten the log likelihoods to (B, num_samples)
    ll_q = flat_ll_q.view(B, num_samples)
    ll_p = flat_ll_p.view(B, num_samples)

    # Monte Carlo estimate of KL
    kl_mc = (ll_q - ll_p).mean(dim=1) # (batch_size,)

    return kl_mc
