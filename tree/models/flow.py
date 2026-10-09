"""Normalizing flow: a chain of 2D affine autoregressive layers.

In two dimensions, an affine autoregressive layer (MAF-style) and an affine
coupling layer are the same thing. One layer with order (i -> j):

    x_i = a + e^b * z_i                    (elementwise affine)
    x_j = t(x_i) + e^{s(x_i)} * z_j        (affine, conditioned on x_i)

Its Jacobian is triangular, so log|det J| = b + s(x_i). The chain

    z = y_0  --f_1-->  y_1  --f_2-->  ...  --f_L-->  y_L = x

pushes the Gaussian forward stage by stage. If every layer keeps the same
order, x_1 stays Gaussian and every p(x_2 | x_1) stays Gaussian, however many
layers are stacked; alternating the order (a permutation) is what lets the
chain bend.
"""
import math

import torch
import torch.nn as nn


class AffineAR2D(nn.Module):
    def __init__(self, first, hidden=64):
        super().__init__()
        self.i, self.j = (0, 1) if first == 0 else (1, 0)
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)   # start at the identity map
        nn.init.zeros_(self.net[-1].bias)

    def _ts(self, xi):
        t, s = self.net(xi.unsqueeze(-1)).unbind(-1)
        return t, 2.0 * torch.tanh(s / 2.0)    # bounded log-scale

    def _join(self, ui, uj):
        return torch.stack([ui, uj], 1) if self.i == 0 else torch.stack([uj, ui], 1)

    def forward(self, z):
        """z -> x, with log|det dx/dz|."""
        xi = self.a + self.b.exp() * z[:, self.i]
        t, s = self._ts(xi)
        xj = t + s.exp() * z[:, self.j]
        return self._join(xi, xj), self.b + s

    def inverse(self, x):
        """x -> z, with log|det dz/dx|."""
        xi = x[:, self.i]
        t, s = self._ts(xi)
        zi = (xi - self.a) * (-self.b).exp()
        zj = (x[:, self.j] - t) * (-s).exp()
        return self._join(zi, zj), -(self.b + s)


class Flow(nn.Module):
    def __init__(self, layers=3, permute=True, hidden=64):
        super().__init__()
        firsts = [(k % 2) if permute else 0 for k in range(layers)]
        self.layers = nn.ModuleList(AffineAR2D(f, hidden) for f in firsts)
        self.orders = [("x₁→x₂" if f == 0 else "x₂→x₁") for f in firsts]

    @property
    def n(self):
        return len(self.layers)

    def push(self, y, start, end):
        """Forward from stage `start` to stage `end` (start <= end)."""
        for layer in self.layers[start:end]:
            y, _ = layer(y)
        return y

    def pull(self, y, start, end=0):
        """Inverse from stage `start` down to stage `end`, with summed log|det|."""
        ld = torch.zeros(len(y))
        for layer in reversed(self.layers[end:start]):
            y, l = layer.inverse(y)
            ld = ld + l
        return y, ld

    def log_prob_at(self, y, stage):
        """log density of the distribution at `stage` (the pushforward of N(0, I))."""
        z, ld = self.pull(y, stage, 0)
        return -0.5 * (z ** 2).sum(-1) - math.log(2 * math.pi) + ld

    def log_prob(self, x):
        return self.log_prob_at(x, self.n)

    def loss(self, x):
        return -self.log_prob(x).mean()

    @torch.no_grad()
    def sample(self, n):
        return self.push(torch.randn(n, 2), 0, self.n)
