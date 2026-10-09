"""Variational autoencoder with a 2D latent. Two crux maps:

    encoder  x |-> q(z | x) = N(mu(x), diag sigma(x)^2)
    decoder  z |-> p(x | z) = N(m(z), diag sigma_x^2)
"""
import math

import torch
import torch.nn as nn


def mlp(n_in, n_out, hidden):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_out),
    )


class VAE(nn.Module):
    def __init__(self, zdim=2, hidden=64):
        super().__init__()
        self.zdim = zdim
        self.enc = mlp(2, 2 * zdim, hidden)
        self.dec = mlp(zdim, 2, hidden)
        self.log_sigma_x = nn.Parameter(torch.full((2,), math.log(0.3)))

    def encode(self, x):
        mu, log_var = self.enc(x).chunk(2, -1)
        return mu, log_var.clamp(-12, 6)

    def decode(self, z):
        return self.dec(z)

    def sigma_x(self):
        return self.log_sigma_x.clamp(math.log(0.02), 2).exp()

    def loss(self, x, beta=1.0):
        mu, log_var = self.encode(x)
        z = mu + (0.5 * log_var).exp() * torch.randn_like(mu)  # reparameterization
        m = self.decode(z)
        s = self.sigma_x()
        rec = (0.5 * ((x - m) / s) ** 2 + s.log() + 0.5 * math.log(2 * math.pi)).sum(-1).mean()
        kl_per_point = 0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum(-1)
        kl = kl_per_point.mean()
        return rec + beta * kl, rec, kl

    @torch.no_grad()
    def sample(self, n, noise=True):
        m = self.decode(torch.randn(n, self.zdim))
        return m + self.sigma_x() * torch.randn_like(m) if noise else m
