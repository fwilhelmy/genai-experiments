import torch 
from torch import nn 
from typing import Optional, Tuple


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def gather(self, c: torch.Tensor, t: torch.Tensor):
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    # Forward distribution q(x_t | x_0)
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar_t = self.gather(self.alpha_bar, t)

        mean = x0 * alpha_bar_t.sqrt()
        var = 1.0 - alpha_bar_t

        return mean, var

    # Sampling from q(x_t | x_0)
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        
        mean, var = self.q_xt_x0(x0, t)
        sample = mean + var.sqrt() * eps

        return sample

    # Reverse distribution p(x_{t-1} | x_t)
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        # Unrolling the parameters (B,1,1,1)
        beta_t = self.gather(self.beta, t)
        alpha_t = self.gather(self.alpha, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)

        # Predicting the noise
        estimated_noise = self.eps_model(xt, t)

        # Reversing the diffusion process (atleast trying to)
        estimated_x0 = xt - (beta_t / (1 - alpha_bar_t).sqrt()) * estimated_noise

        # Posterior mean
        mu_theta = 1/alpha_t.sqrt() * estimated_x0

        # Posterior variance
        var = beta_t
        
        return mu_theta, var

    # Sampling from p(x_{t-1} | x_t)
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        
        mu_theta, var = self.p_xt_prev_xt(xt, t)
        eps = torch.randn_like(xt)
        sample = mu_theta + var.sqrt() * eps

        return sample

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Sample forward diffusion and estimate noise
        xt = self.q_sample(x0, t, eps=noise)
        estimated_noise = self.eps_model(xt, t)

        # Compute the loss
        loss = (noise - estimated_noise).pow(2).sum(dim=dim).mean()

        return loss