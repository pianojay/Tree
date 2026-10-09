"""Toy ground truths (provisional debug ladder, not the curated Tree).

Each dataset is a sampling program over R^2. Coordinates are unbounded reals;
the display maps each coordinate to a hue (see web/src/colors.ts).
"""
import numpy as np

WINDOW = (-3.0, 3.0)  # plotting window for both coordinates


def _logit(p):
    return float(np.log(p / (1 - p)))


def _gauss(rng, n, mean, cov):
    return rng.multivariate_normal(mean, cov, size=n)


def _mixture(rng, n, comps):
    """comps: list of (weight, sampler(rng, m))."""
    w = np.array([c[0] for c in comps], dtype=float)
    k = rng.choice(len(comps), size=n, p=w / w.sum())
    out = np.empty((n, 2))
    for i, (_, f) in enumerate(comps):
        m = k == i
        out[m] = f(rng, int(m.sum()))
    return out


def _rotated(rng, m, angle_deg, std_major, std_minor, center=(0.0, 0.0)):
    a = np.deg2rad(angle_deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cov = r @ np.diag([std_major ** 2, std_minor ** 2]) @ r.T
    return _gauss(rng, m, center, cov)


def simple(rng, n):
    return _gauss(rng, n, [0.5, -0.3], [[0.36, 0.15], [0.15, 0.25]])


def multimodal(rng, n):
    blob = lambda mean, std: (lambda r, m: _gauss(r, m, mean, np.eye(2) * std ** 2))
    return _mixture(rng, n, [
        (0.40, blob((-1.6, -1.2), 0.30)),
        (0.30, blob((1.4, -1.0), 0.25)),
        (0.25, blob((0.0, 1.5), 0.30)),
        (0.05, blob((1.9, 1.9), 0.12)),  # rare and narrow
    ])


def tree(rng, n):
    """Pixel 1 = crown, pixel 2 = trunk. Coordinates sit where the display hue map
    puts green / yellow (crown) and orange (trunk; rendered desaturated, i.e. brown)."""
    to_v = lambda hue_deg: _logit(hue_deg / 300.0)   # inverse of the display map
    summer = (to_v(120), to_v(30))                  # green crown, brown trunk
    autumn = (to_v(55), to_v(26))                   # yellow crown, brown trunk
    return _mixture(rng, n, [
        (0.7, lambda r, m: _gauss(r, m, summer, [[0.08, 0.03], [0.03, 0.04]])),
        (0.3, lambda r, m: _gauss(r, m, autumn, [[0.06, 0.02], [0.02, 0.04]])),
    ])


def x_shape(rng, n):
    """Two crossing elongated Gaussians: p(x2 | x1) is bimodal away from the center."""
    return _mixture(rng, n, [
        (0.5, lambda r, m: _rotated(r, m, 45, 1.3, 0.18)),
        (0.5, lambda r, m: _rotated(r, m, -45, 1.3, 0.18)),
    ])


def sine(rng, n):
    """Periodic: a noisy band along x2 = 1.3 sin(2 x1)."""
    x1 = rng.uniform(-2.8, 2.8, n)
    x2 = 1.3 * np.sin(2.0 * x1) + rng.normal(0, 0.15, n)
    return np.stack([x1, x2], 1)


def two_moons(rng, n):
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    a = np.stack([np.cos(t1), np.sin(t1)], 1)
    b = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], 1)
    out = (np.concatenate([a, b]) - [0.5, 0.25]) * 1.6
    out = out + rng.normal(0, 0.08, out.shape)
    return out[rng.permutation(n)]   # any prefix must be a fair sample


def ring8(rng, n):
    k = rng.integers(0, 8, n)
    ang = 2 * np.pi * k / 8
    return 2.0 * np.stack([np.cos(ang), np.sin(ang)], 1) + rng.normal(0, 0.15, (n, 2))


def rings(rng, n):
    r = rng.choice([1.0, 2.2], n)
    t = rng.uniform(0, 2 * np.pi, n)
    return (r + rng.normal(0, 0.08, n))[:, None] * np.stack([np.cos(t), np.sin(t)], 1)


def spiral(rng, n):
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    pts = np.stack([t * np.cos(t), t * np.sin(t)], 1) / (3 * np.pi) * 2.6
    return pts + rng.normal(0, 0.08, pts.shape)


def checkerboard(rng, n):
    out = []
    while len(out) < n:
        p = rng.uniform(-2, 2, (2 * n, 2))
        keep = (np.floor(p[:, 0]) + np.floor(p[:, 1])) % 2 == 0
        out.extend(p[keep])
    return np.array(out[:n])


def crescent(rng, n):
    n_arc = int(0.65 * n)
    theta = rng.uniform(np.deg2rad(20), np.deg2rad(160), size=n_arc)
    arc = np.stack([1.7 * np.cos(theta), 1.7 * np.sin(theta) - 0.6], 1)
    arc += rng.normal(0, 0.12, size=arc.shape)
    blob = _gauss(rng, n - n_arc, [0.0, -1.7], np.eye(2) * 0.2 ** 2)
    out = np.concatenate([arc, blob])
    return out[rng.permutation(n)]


DATASETS = {f.__name__: f for f in (
    simple, multimodal, tree, x_shape, sine, two_moons, ring8, rings, spiral, checkerboard, crescent)}


def sample(name, n, seed=0):
    rng = np.random.default_rng(seed)
    return DATASETS[name](rng, n).astype(np.float32)
