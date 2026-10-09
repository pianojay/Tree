"""Gradient check: autodiff gradients vs. central finite differences.

Run:  python -m tests.test_engine
"""
import random

from tree.engine import Value


def f(a, b, c):
    """A composite expression touching every op in the engine."""
    u = (a * b + c ** 2).tanh()
    v = (-a / (1.0 + c.relu())).exp()
    w = (b.sigmoid() + 1.0).log()
    return u * v - w + 3.0 * a - b / 2.0 + (c - a).relu()


def check(point, h=1e-6, tol=1e-6):
    xs = [Value(p) for p in point]
    out = f(*xs)
    out.backward()
    ok = True
    for i, x in enumerate(xs):
        up = list(point); up[i] += h
        dn = list(point); dn[i] -= h
        fd = (f(*map(Value, up)).data - f(*map(Value, dn)).data) / (2 * h)
        err = abs(fd - x.grad) / max(1.0, abs(fd))
        ok &= err < tol
        print(f"  d/dx{i}: autodiff={x.grad:+.8f}  finite-diff={fd:+.8f}  rel.err={err:.1e}")
    return ok


if __name__ == "__main__":
    rng = random.Random(0)
    all_ok = True
    for trial in range(5):
        point = [rng.uniform(-2, 2) for _ in range(3)]
        print(f"point {trial}: {[round(p, 3) for p in point]}")
        all_ok &= check(point)
    print("GRADCHECK", "PASSED" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)
