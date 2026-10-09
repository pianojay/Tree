"""Autoregressive model, RNADE-style: continuous values, one pixel at a time.

    p(x1, x2) = p(x1) * p(x2 | x1)

p(x1) is a free mixture of Gaussians. p(x2 | x1) is a mixture of Gaussians
whose parameters come from a small MLP of x1. The crux map is

    x1  |->  p(x2 | x1)
"""
import math

import torch
import torch.nn as nn


def mog_log_prob(x, logits, mu, log_sigma):
    """log of sum_k pi_k N(x; mu_k, sigma_k^2). x: (B,), params: (B, K) or (K,)."""
    x = x.unsqueeze(-1)
    log_pi = torch.log_softmax(logits, -1)
    log_n = -0.5 * ((x - mu) / log_sigma.exp()) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(log_pi + log_n, -1)


def mog_sample(logits, mu, log_sigma):
    k = torch.distributions.Categorical(logits=logits).sample()
    pick = lambda t: t.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    return pick(mu) + pick(log_sigma).exp() * torch.randn_like(pick(mu))


class AR(nn.Module):
    def __init__(self, k=8, hidden=64):
        super().__init__()
        self.k = k
        self.p1_logits = nn.Parameter(torch.zeros(k))
        self.p1_mu = nn.Parameter(torch.linspace(-2, 2, k))
        self.p1_log_sigma = nn.Parameter(torch.full((k,), -0.5))
        self.cond = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3 * k),
        )

    # --- the two factors ---------------------------------------------------

    def p1_params(self, n):
        return (self.p1_logits.expand(n, -1), self.p1_mu.expand(n, -1),
                self.p1_log_sigma.clamp(-5, 2).expand(n, -1))

    def p2_params(self, x1):
        logits, mu, log_sigma = self.cond(x1.unsqueeze(-1)).chunk(3, -1)
        return logits, mu, log_sigma.clamp(-5, 2)

    def log_prob_parts(self, x):
        lp1 = mog_log_prob(x[:, 0], *self.p1_params(len(x)))
        lp2 = mog_log_prob(x[:, 1], *self.p2_params(x[:, 0]))
        return lp1, lp2

    def loss(self, x):
        lp1, lp2 = self.log_prob_parts(x)
        return -(lp1 + lp2).mean()

    # --- generation --------------------------------------------------------

    @torch.no_grad()
    def sample(self, n):
        x1 = mog_sample(*self.p1_params(n))
        x2 = mog_sample(*self.p2_params(x1))
        return torch.stack([x1, x2], 1)

    # --- views -------------------------------------------------------------

    @torch.no_grad()
    def marginal1(self, grid):
        return mog_log_prob(grid, *self.p1_params(len(grid))).exp()

    @torch.no_grad()
    def conditional(self, x1, grid):
        """p(x2 | x1) evaluated on a grid of x2 values."""
        params = self.p2_params(torch.full((1,), float(x1)))
        return mog_log_prob(grid, *(p.expand(len(grid), -1) for p in params)).exp()

    @torch.no_grad()
    def density_grid(self, axis):
        g1, g2 = torch.meshgrid(axis, axis, indexing="xy")
        pts = torch.stack([g1.reshape(-1), g2.reshape(-1)], 1)
        lp1, lp2 = self.log_prob_parts(pts)
        return (lp1 + lp2).exp().reshape(len(axis), len(axis))  # [row = x2][col = x1]
