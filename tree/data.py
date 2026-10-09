"""Toy ground truths (provisional debug ladder, not the curated Tree).

Each dataset is a sampling program over R^2. Coordinates are unbounded reals;
the display maps each coordinate to a hue (see web/src/colors.ts).
"""
import numpy as np

WINDOW = (-3.0, 3.0)  # plotting window for both coordinates


def _gauss(rng, n, mean, cov):
    return rng.multivariate_normal(mean, cov, size=n)


def _simple(rng, n):
    return _gauss(rng, n, [0.5, -0.3], [[0.36, 0.15], [0.15, 0.25]])


def _multimodal(rng, n):
    comps = [  # weight, mean, std (isotropic)
        (0.40, (-1.6, -1.2), 0.30),
        (0.30, (1.4, -1.0), 0.25),
        (0.25, (0.0, 1.5), 0.30),
        (0.05, (1.9, 1.9), 0.12),  # rare and narrow
    ]
    w = np.array([c[0] for c in comps])
    k = rng.choice(len(comps), size=n, p=w / w.sum())
    out = np.empty((n, 2))
    for i, (_, mean, std) in enumerate(comps):
        m = k == i
        out[m] = _gauss(rng, m.sum(), mean, np.eye(2) * std ** 2)
    return out


def _crescent(rng, n):
    n_arc = int(0.65 * n)
    theta = rng.uniform(np.deg2rad(20), np.deg2rad(160), size=n_arc)
    arc = np.stack([1.7 * np.cos(theta), 1.7 * np.sin(theta) - 0.6], 1)
    arc += rng.normal(0, 0.12, size=arc.shape)
    blob = _gauss(rng, n - n_arc, [0.0, -1.7], np.eye(2) * 0.2 ** 2)
    out = np.concatenate([arc, blob])
    return out[rng.permutation(n)]


DATASETS = {"simple": _simple, "multimodal": _multimodal, "crescent": _crescent}


def sample(name, n, seed=0):
    rng = np.random.default_rng(seed)
    return DATASETS[name](rng, n).astype(np.float32)
