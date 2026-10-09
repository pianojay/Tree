"""Does it learn? A perceptron vs. a small MLP on an XOR-shaped 2D problem.

Four blobs at (+-1, +-1). Label 1 where x1*x2 > 0, else 0. No straight line
separates the classes, so a perceptron (one linear layer) cannot solve it;
an MLP with a hidden layer can.

Run:  python -m tests.test_learns
"""
import math
import random
import time

from tree.engine import Value
from tree.nn import MLP, Adam


def make_xor(n, rng, std=0.3):
    data = []
    for _ in range(n):
        sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
        x = (sx + rng.gauss(0, std), sy + rng.gauss(0, std))
        data.append((x, 1.0 if sx * sy > 0 else 0.0))
    return data


def bce_with_logit(z, y):
    # log(1 + e^z) - y*z, written two ways for numerical stability
    if z.data >= 0:
        return z + (1.0 + (-z).exp()).log() - y * z
    return (1.0 + z.exp()).log() - y * z


def accuracy(model, data):
    return sum((model(x)[0].data > 0) == (y > 0.5) for x, y in data) / len(data)


def ascii_map(model, lo=-2.0, hi=2.0, rows=17, cols=41):
    """Predicted class over the square: '#' = 1, '.' = 0."""
    lines = []
    for r in range(rows):
        x2 = hi - (hi - lo) * r / (rows - 1)
        line = ""
        for c in range(cols):
            x1 = lo + (hi - lo) * c / (cols - 1)
            line += "#" if model((x1, x2))[0].data > 0 else "."
        lines.append("    " + line)
    return "\n".join(lines)


def train(name, sizes, train_set, test_set, steps=400, batch=32, lr=0.03, seed=0):
    model = MLP(sizes, act="tanh", seed=seed)
    opt = Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    n_params = len(model.parameters())
    print(f"\n== {name}: layers {sizes}, {n_params} parameters ==")
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        mb = rng.sample(train_set, batch)
        loss = sum(bce_with_logit(model(x)[0], y) for x, y in mb) / batch
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            print(f"  step {step:4d}  loss {loss.data:.4f}")
    secs = time.perf_counter() - t0
    acc = accuracy(model, test_set)
    print(f"  test accuracy {acc:.1%}   ({secs:.1f}s, {1000 * secs / steps:.1f} ms/step)")
    print(ascii_map(model))
    return acc


if __name__ == "__main__":
    rng = random.Random(42)
    train_set, test_set = make_xor(256, rng), make_xor(512, rng)

    acc_p = train("perceptron", [2, 1], train_set, test_set)
    acc_m = train("MLP", [2, 16, 16, 1], train_set, test_set)

    print(f"\nperceptron {acc_p:.1%}  vs  MLP {acc_m:.1%}")
    ok = acc_p < 0.8 and acc_m > 0.95
    print("LEARNS TEST", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)
