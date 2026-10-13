"""Local demo server: trains the miniature models and answers the page's queries.

Run:  .venv/Scripts/python -m uvicorn server:app --port 8000
Then open http://localhost:8000 (serves the built page from web/dist).
"""
import threading
import time
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tree.data import DATASETS, WINDOW, sample
from tree.models.ar import AR
from tree.models.flow import Flow
from tree.models.gan import GAN
from tree.models.vae import VAE

torch.set_num_threads(2)

N_TRAIN = 2000        # training set: a finite subset of p_data
N_SHOW = 500          # points sent to the page
SNAP_SECONDS = float(os.environ.get('TREE_SNAPSHOT_SECONDS', '0.35'))
MAX_STEPS = 20000

AXIS = torch.linspace(WINDOW[0], WINDOW[1], 64)
AXIS_SMALL = torch.linspace(WINDOW[0], WINDOW[1], 48)
FINE = torch.linspace(WINDOW[0], WINDOW[1], 161)


def r(t, nd=3):
    """Tensor/array -> rounded nested lists (small JSON)."""
    a = t.detach().numpy() if torch.is_tensor(t) else np.asarray(t)
    return np.round(a.astype(float), nd).tolist()


def grid_points(axis):
    g1, g2 = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack([g1.reshape(-1), g2.reshape(-1)], 1)  # row-major: rows = x2


def latent_grid_lines(lo=-2.5, hi=2.5, n_lines=11, n_pts=61, span=3.0):
    along = torch.linspace(-span, span, n_pts)
    lines = []
    for t in torch.linspace(lo, hi, n_lines):
        lines.append(torch.stack([torch.full_like(along, t), along], 1))
        lines.append(torch.stack([along, torch.full_like(along, t)], 1))
    return lines


class Session:
    """One model family: owns the model, trains it in a thread, serves snapshots."""
    kind = ""
    default_cfg: dict = {}

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cfg = dict(self.default_cfg)
        self.reset("multimodal")

    # --- lifecycle -------------------------------------------------------------

    def reset(self, dataset, cfg=None):
        self.pause()
        with self.lock:
            if cfg:
                self.cfg.update(cfg)
            torch.manual_seed(0)
            self.dataset = dataset
            self.generation = getattr(self, "generation", 0) + 1
            self.data = torch.tensor(sample(dataset, N_TRAIN, seed=0))
            self.heldout = torch.tensor(sample(dataset, 512, seed=1))
            self.step = 0
            self.history = []
            self.snapshot_count = 0
            self.snapshot_total_ms = 0.0
            self.build()
            self.snapshot = self._snapshot()

    def configure(self, cfg):
        """Live changes that do not need a rebuild."""
        with self.lock:
            self.cfg.update(cfg)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def pause(self):
        self.running = False
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        self.thread = None

    def _loop(self):
        last, cost = 0.0, 0.0
        while self.running and self.step < MAX_STEPS:
            with self.lock:
                for _ in range(5):
                    vals = self.train_step()
                    self.step += 1
                    if self.step % 5 == 0:
                        self.history.append([self.step] + [None if v is None else round(float(v), 4) for v in vals])
                # snapshot often, but never spend more than about a third of the time on it
                if time.perf_counter() - last > max(SNAP_SECONDS, 2 * cost):
                    t0 = time.perf_counter()
                    self.snapshot = self._snapshot()
                    last = time.perf_counter()
                    cost = last - t0
            time.sleep(0.001)  # let queries in
        with self.lock:
            self.snapshot = self._snapshot()
        self.running = False

    def batch(self, n=256):
        return self.data[torch.randint(0, len(self.data), (n,))]

    def _snapshot(self):
        started = time.perf_counter()
        self.model.eval()
        snap = {"kind": self.kind, "dataset": self.dataset, "step": self.step,
                "generation": self.generation,
                "history": self.history[-3000:], "window": WINDOW, "cfg": dict(self.cfg),
                "data": r(self.data[:N_SHOW])}
        with torch.no_grad():
            snap.update(self.extra())
            if self.kind in ("ar", "flow"):
                snap["heldout_nll"] = self.model.loss(self.heldout).item()
        self.model.train()
        self.snapshot_last_ms = (time.perf_counter() - started) * 1000
        self.snapshot_count += 1
        self.snapshot_total_ms += self.snapshot_last_ms
        return snap

    # --- per family ------------------------------------------------------------

    def build(self): raise NotImplementedError
    def train_step(self): raise NotImplementedError
    def extra(self): return {}


class ARSession(Session):
    kind = "ar"
    default_cfg = {"k": 8}

    def build(self):
        self.model = AR(k=int(self.cfg["k"]))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)

    def train_step(self):
        loss = self.model.loss(self.batch())
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return [loss.item()]

    def extra(self):
        return {"axis": r(AXIS), "density": r(self.model.density_grid(AXIS), 4),
                "fine": r(FINE), "marginal1": r(self.model.marginal1(FINE), 4),
                "samples": r(self.model.sample(N_SHOW))}


class VAESession(Session):
    kind = "vae"
    default_cfg = {"beta": 1.0}

    def build(self):
        self.model = VAE()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)

    def train_step(self):
        loss, rec, kl = self.model.loss(self.batch(), float(self.cfg["beta"]))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return [loss.item(), rec.item(), kl.item()]

    def extra(self):
        mu, log_var = self.model.encode(self.data[:N_SHOW])
        return {
            "sigma_x": r(self.model.sigma_x()),
            "samples": r(self.model.sample(N_SHOW)),
            "latent_mu": r(mu), "latent_sigma": r((0.5 * log_var).exp()),
            "kl_points": r(0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum(-1)),
            "grid_lines": [r(self.model.decode(z)) for z in latent_grid_lines()],
        }


class FlowSession(Session):
    kind = "flow"
    default_cfg = {"layers": 3, "permute": True}

    def build(self):
        self.model = Flow(int(self.cfg["layers"]), bool(self.cfg["permute"]))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=2e-3)

    def train_step(self):
        loss = self.model.loss(self.batch())
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return [loss.item()]

    def extra(self):
        f, n = self.model, self.model.n
        pts = grid_points(AXIS_SMALL)
        lines = latent_grid_lines(n_pts=41)
        n_lines, n_pts = len(lines), len(lines[0])

        grids = [torch.cat(lines)]                    # base grid pushed forward, stage by stage
        for layer in f.layers:
            grids.append(layer(grids[-1])[0])
        datas = [self.data[:300]]                     # data mapped back, stage by stage
        for layer in reversed(f.layers):
            datas.append(layer.inverse(datas[-1])[0])
        datas.reverse()

        stages = [{
            "density": r(f.log_prob_at(pts, k).exp().reshape(len(AXIS_SMALL), -1), 4),
            "data": r(datas[k]),
            "grid": r(grids[k].reshape(n_lines, n_pts, 2)),
        } for k in range(n + 1)]
        return {"axis": r(AXIS_SMALL), "stages": stages, "orders": f.orders,
                "samples": r(f.sample(N_SHOW))}

    def trace(self, stage, point):
        """One point seen at every stage of the chain."""
        f = self.model
        with self.lock, torch.no_grad():
            y = torch.tensor([point], dtype=torch.float32)
            z, _ = f.pull(y, stage, 0)
            return [r(f.push(z, 0, k)[0]) for k in range(f.n + 1)]


class GANSession(Session):
    kind = "gan"
    default_cfg = {"d_steps": 1, "freeze_d": False, "freeze_g": False}

    def build(self):
        self.model = GAN()
        betas = (0.5, 0.999)
        self.opt_d = torch.optim.Adam(self.model.D.parameters(), lr=1e-3, betas=betas)
        self.opt_g = torch.optim.Adam(self.model.G.parameters(), lr=1e-3, betas=betas)
        g = torch.Generator().manual_seed(1)
        self.z_show = torch.randn(N_SHOW, 2, generator=g)   # fixed latent points shown on the page

    def train_step(self):
        m, d_loss, g_loss = self.model, None, None   # a frozen player logs a gap, not 0
        if not self.cfg["freeze_d"]:
            for _ in range(int(self.cfg["d_steps"])):
                d_loss = m.d_loss(self.batch(), m.G(torch.randn(256, 2)))
                self.opt_d.zero_grad(); d_loss.backward(); self.opt_d.step()
        if not self.cfg["freeze_g"]:
            g_loss = m.g_loss(m.G(torch.randn(256, 2)))
            self.opt_g.zero_grad(); g_loss.backward(); self.opt_g.step()
        with torch.no_grad():
            p_real = m.p_real(self.batch()).mean()
            p_fake = m.p_real(m.G(torch.randn(256, 2))).mean()
        item = lambda t: None if t is None else t.item()
        return [item(d_loss), item(g_loss), p_real.item(), p_fake.item()]

    def extra(self):
        m = self.model
        fake = m.G(self.z_show)
        with torch.enable_grad():   # the generator's only signal: grad of log D at its samples
            x = fake[:120].clone().requires_grad_(True)
            torch.nn.functional.logsigmoid(m.D(x)).sum().backward()
            grad = x.grad
        for p in m.D.parameters():
            p.grad = None
        return {"axis": r(AXIS), "d_field": r(m.p_real(grid_points(AXIS)).reshape(len(AXIS), -1), 3),
                "z": r(self.z_show), "fake": r(fake), "grad": r(grad, 4),
                "grid_lines": [r(m.G(z)) for z in latent_grid_lines()]}


sessions = {s.kind: s for s in (ARSession(), VAESession(), FlowSession(), GANSession())}
app = FastAPI()


@app.middleware("http")
async def revalidate_pages(request, call_next):
    """Hashed assets may be cached; everything else is revalidated so rebuilds show up."""
    response = await call_next(request)
    if not request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class Control(BaseModel):
    action: str                  # start | pause | reset | config
    dataset: str | None = None
    config: dict | None = None


@app.get("/api/datasets")
def datasets():
    return list(DATASETS)


@app.get("/api/_runtime")
def runtime():
    """Persistent model, data and optimizer tensor ownership for verification."""
    tensors = []
    for s in sessions.values():
        with s.lock:
            tensors.extend(s.model.parameters())
            tensors.extend([s.data, s.heldout])
            if hasattr(s, 'z_show'):
                tensors.append(s.z_show)
            for name in ('opt', 'opt_d', 'opt_g'):
                opt = getattr(s, name, None)
                if opt:
                    for state in opt.state.values():
                        tensors.extend(v for v in state.values() if torch.is_tensor(v))
    return {'backend': 'pytorch-cpu', 'version': torch.__version__,
            'threads': torch.get_num_threads(), 'tensors': len(tensors),
            'bytes': sum(t.numel() * t.element_size() for t in tensors)}


@app.get("/api/{kind}/metrics")
def metrics(kind: str):
    s = sessions[kind]
    with s.lock:
        return {'snapshot_count': s.snapshot_count, 'snapshot_total_ms': s.snapshot_total_ms,
                'snapshot_last_ms': s.snapshot_last_ms, 'snapshot_interval_ms': SNAP_SECONDS * 1000}


@app.get("/api/{kind}/state")
def state(kind: str):
    s = sessions[kind]
    with s.lock:
        snap = dict(s.snapshot)
    snap["running"] = s.running
    return snap


@app.post("/api/{kind}/control")
def control(kind: str, c: Control):
    s = sessions[kind]
    if c.action == "start":
        s.start()
    elif c.action == "pause":
        s.pause()
    elif c.action == "reset":
        s.reset(c.dataset or s.dataset, c.config)
    elif c.action == "config" and c.config:
        s.configure(c.config)
    return {"ok": True}


# --- AR queries --------------------------------------------------------------

@app.get("/api/ar/conditional")
def ar_conditional(x1: float):
    s = sessions["ar"]
    with s.lock:
        p = s.model.conditional(x1, FINE)
        p1 = s.model.marginal1(torch.tensor([x1]))
    return {"x1": x1, "fine": r(FINE), "p": r(p, 4), "p1": float(p1[0])}


@app.get("/api/ar/point")
def ar_point(x1: float, x2: float):
    s = sessions["ar"]
    with s.lock, torch.no_grad():
        lp1, lp2 = s.model.log_prob_parts(torch.tensor([[x1, x2]]))
    return {"lp1": float(lp1[0]), "lp2": float(lp2[0])}


@app.get("/api/ar/sample_one")
def ar_sample_one():
    s = sessions["ar"]
    with s.lock:
        x = s.model.sample(1)[0]
    return {"x1": float(x[0]), "x2": float(x[1])}


# --- VAE queries -------------------------------------------------------------

@app.get("/api/vae/encode")
def vae_encode(x1: float, x2: float):
    s = sessions["vae"]
    with s.lock, torch.no_grad():
        mu, log_var = s.model.encode(torch.tensor([[x1, x2]]))
        kl = 0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum(-1)
        rec = s.model.decode(mu)
        sx = s.model.sigma_x()
        eps = torch.randn(40, 2, generator=torch.Generator().manual_seed(7))
        z = mu + (0.5 * log_var).exp() * eps            # reparameterization
        xhat = s.model.decode(z)
    return {"mu": r(mu[0]), "sigma": r((0.5 * log_var[0]).exp()),
            "kl": float(kl[0]), "recon": r(rec[0]), "sigma_x": r(sx),
            "eps": r(eps), "z": r(z), "xhat": r(xhat)}


@app.get("/api/vae/decode")
def vae_decode(z1: float, z2: float):
    s = sessions["vae"]
    with s.lock, torch.no_grad():
        m = s.model.decode(torch.tensor([[z1, z2]]))
        sx = s.model.sigma_x()
    return {"mean": r(m[0]), "sigma": r(sx)}


@app.get("/api/vae/inspect")
def vae_inspect(x1: float, x2: float, resolution: int = 64):
    """Finite-domain quadrature and a deterministic Monte Carlo ELBO estimate."""
    import math
    n = max(24, min(120, resolution))
    span, samples = 4.0, 2048
    delta = 2 * span / n
    zaxis = -span + (torch.arange(n) + 0.5) * delta
    s = sessions['vae']
    with s.lock, torch.no_grad():
        m = s.model
        x = torch.tensor([[x1, x2]])
        z = grid_points(zaxis)
        sigma = m.sigma_x()
        normal_logp = lambda residual, std: -(0.5 * (residual / std) ** 2 + std.log() + 0.5 * math.log(2 * math.pi)).sum(-1)
        ll = normal_logp(x - m.decode(z), sigma)
        joint = ll - 0.5 * (z ** 2).sum(-1) - math.log(2 * math.pi)
        logp = torch.logsumexp(joint, 0) + 2 * math.log(delta)
        mu, lv = m.encode(x)
        zs = mu + (0.5 * lv).exp() * torch.randn(samples, 2, generator=torch.Generator().manual_seed(91))
        draws = normal_logp(x - m.decode(zs), sigma)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum()
        elbo = draws.mean() - kl
        se = draws.std() / math.sqrt(samples)
        q = normal_logp(z - mu, (0.5 * lv).exp()).exp()
        return {'step': s.step, 'x': [x1, x2], 'axis': r(zaxis, 6),
                'likelihood': r((ll - ll.max()).exp().reshape(n, n), 7),
                'posterior': r((joint - logp).exp().reshape(n, n), 7),
                'q': r(q.reshape(n, n), 7), 'logp': float(logp), 'elbo': float(elbo),
                'mu': r(mu[0], 7), 'sigma': r((0.5 * lv[0]).exp(), 7),
                'gap': float(logp - elbo), 'mc_se': float(se), 'samples': samples,
                'domain': [-span, span], 'resolution': n}


# --- flow and GAN queries ----------------------------------------------------

@app.get("/api/flow/trace")
def flow_trace(stage: int, x1: float, x2: float):
    return {"points": sessions["flow"].trace(stage, [x1, x2])}


@app.get("/api/gan/generate")
def gan_generate(z1: float, z2: float):
    s = sessions["gan"]
    with s.lock, torch.no_grad():
        x = s.model.G(torch.tensor([[z1, z2]]))
        p = s.model.p_real(x)
    return {"x": r(x[0]), "p_real": float(p[0])}


@app.get("/api/gan/discriminate")
def gan_discriminate(x1: float, x2: float):
    s = sessions["gan"]
    with s.lock:
        p = s.model.p_real(torch.tensor([[x1, x2]]))
    return {"p_real": float(p[0])}


dist = Path(__file__).parent / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
