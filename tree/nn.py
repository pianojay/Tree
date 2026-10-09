"""Minimal networks and optimizer on top of engine.Value. Pure Python."""
import math
import random

from tree.engine import Value

ACTIVATIONS = {
    "tanh": lambda v: v.tanh(),
    "relu": lambda v: v.relu(),
    "linear": lambda v: v,
}


class Linear:
    """y = W x + b, for one input vector (a list of Values or floats)."""

    def __init__(self, n_in, n_out, rng):
        scale = 1.0 / math.sqrt(n_in)
        self.W = [[Value(rng.uniform(-scale, scale)) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [Value(0.0) for _ in range(n_out)]

    def __call__(self, x):
        return [sum((w * xi for w, xi in zip(row, x)), b) for row, b in zip(self.W, self.b)]

    def parameters(self):
        return [w for row in self.W for w in row] + self.b


class MLP:
    """Stack of Linear layers; `act` between layers, none after the last.

    MLP([2, 1]) is a single linear layer: the perceptron.
    """

    def __init__(self, sizes, act="tanh", seed=0):
        rng = random.Random(seed)
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.act = ACTIVATIONS[act]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = [self.act(v) for v in x]
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Adam:
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * p.grad ** 2
            p.data -= self.lr * (self.m[i] / c1) / (math.sqrt(self.v[i] / c2) + self.eps)
