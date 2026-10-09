"""GAN: an implicit generator G: z -> x played against a discriminator D: x -> logit.

No density and no inverse: G is a direct map that is free to fold. The only
signal G ever receives is the gradient of D at its own samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(n_in, n_out, hidden):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.LeakyReLU(0.2),
        nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
        nn.Linear(hidden, n_out),
    )


class GAN(nn.Module):
    def __init__(self, zdim=2, hidden=64):
        super().__init__()
        self.zdim = zdim
        self.G = mlp(zdim, 2, hidden)
        self.D = mlp(2, 1, hidden)

    def d_loss(self, real, fake):
        lr, lf = self.D(real), self.D(fake.detach())
        return (F.binary_cross_entropy_with_logits(lr, torch.ones_like(lr))
                + F.binary_cross_entropy_with_logits(lf, torch.zeros_like(lf)))

    def g_loss(self, fake):
        """Non-saturating: maximize log D(G(z))."""
        lf = self.D(fake)
        return F.binary_cross_entropy_with_logits(lf, torch.ones_like(lf))

    @torch.no_grad()
    def p_real(self, x):
        return torch.sigmoid(self.D(x)).squeeze(-1)
