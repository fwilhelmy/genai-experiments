# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20
    
    def get_lambda(self, t: torch.Tensor): 
        u = t / (self.n_steps - 1) # u in [0,1]
        lambda_t = self.lambda_min + u * (self.lambda_max - self.lambda_min) # lambda_t in [lambda_min, lambda_max]
        return lambda_t.view(-1, 1, 1, 1) 
    
    def alpha_lambda(self, lambda_t: torch.Tensor):
        var = 1 / (1 + torch.exp(-lambda_t))
        return var.sqrt()

    def sigma_lambda(self, lambda_t: torch.Tensor):
        var = 1 - self.alpha_lambda(lambda_t).pow(2)
        return var.sqrt()
    
    # Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        alpha = self.alpha_lambda(lambda_t)
        sigma = self.sigma_lambda(lambda_t)

        z_lambda_t = alpha * x + sigma * noise

        return z_lambda_t

    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        var_q = (1 - torch.exp(lambda_t - lambda_t_prim)) * self.sigma_lambda(lambda_t).pow(2)
        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        var_q_x = (1 - torch.exp(lambda_t - lambda_t_prim)) * self.sigma_lambda(lambda_t_prim).pow(2)
        return var_q_x.sqrt()

    # Reverse sampling
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        alpha = self.alpha_lambda(lambda_t)
        alpha_prim = self.alpha_lambda(lambda_t_prim)

        t1 = torch.exp(lambda_t - lambda_t_prim) * (alpha_prim / alpha) * z_lambda_t
        t2 = (1 - torch.exp(lambda_t - lambda_t_prim)) * alpha_prim * x
        mu = t1 + t2

        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        t1 = self.sigma_q_x(lambda_t, lambda_t_prim).pow(1 - v)
        t2 = self.sigma_q(lambda_t, lambda_t_prim).pow(v)
        var = t1 * t2   
        return var.pow(2)
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        
        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)
        eps = torch.randn_like(z_lambda_t)

        sample = mu + var.sqrt() * eps

        return sample

    # Loss
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        lambda_t = self.get_lambda(t)
        
        if noise is None:
            noise = torch.randn_like(x0)
        
        z_lambda = self.q_sample(x0, lambda_t, noise)
        
        estimated_noise = self.eps_model(z_lambda, labels)

        loss = (estimated_noise - noise).pow(2).mean()
    
        return loss