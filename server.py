"""Local demo server: trains the miniature models and answers the page's queries.

Run:  .venv/Scripts/python -m uvicorn server:app --port 8000
Then open http://localhost:8000 (serves the built page from web/dist).
"""
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tree.data import DATASETS, WINDOW, sample
from tree.models.ar import AR
from tree.models.vae import VAE

torch.set_num_threads(2)

N_TRAIN = 2000       # training set: a finite subset of p_data
N_SHOW = 600         # points sent to the page
GRID = 64            # density grid resolution
SNAP_EVERY = 20      # training steps between snapshots
MAX_STEPS = 20000

AXIS = torch.linspace(WINDOW[0], WINDOW[1], GRID)
FINE = torch.linspace(WINDOW[0], WINDOW[1], 161)


def r(t, nd=3):
    """Tensor/array -> rounded nested lists (small JSON)."""
    a = t.detach().numpy() if torch.is_tensor(t) else np.asarray(t)
    return np.round(a.astype(float), nd).tolist()


class Trainer:
    """Owns one model; trains it in a background thread; serves snapshots."""

    def __init__(self, kind):
        self.kind = kind
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.reset("multimodal")

    # --- lifecycle ---------------------------------------------------------

    def reset(self, dataset, beta=None):
        with self.lock:
            torch.manual_seed(0)
            self.dataset = dataset
            self.data = torch.tensor(sample(dataset, N_TRAIN, seed=0))
            self.model = AR() if self.kind == "ar" else VAE()
            self.opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
            self.beta = 1.0 if beta is None else beta
            self.step = 0
            self.history = []
            self.snapshot = self._make_snapshot()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def pause(self):
        self.running = False

    def _loop(self):
        while self.running and self.step < MAX_STEPS:
            with self.lock:
                for _ in range(SNAP_EVERY):
                    self._train_step()
                self.snapshot = self._make_snapshot()
            time.sleep(0.01)  # let queries in
        self.running = False

    def _train_step(self):
        batch = self.data[torch.randint(0, len(self.data), (256,))]
        if self.kind == "ar":
            loss = self.model.loss(batch)
            rec = kl = None
        else:
            loss, rec, kl = self.model.loss(batch, self.beta)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.step += 1
        if self.step % 5 == 0:
            row = [self.step, round(loss.item(), 4)]
            if rec is not None:
                row += [round(rec.item(), 4), round(kl.item(), 4)]
            self.history.append(row)

    # --- snapshots ---------------------------------------------------------

    def _make_snapshot(self):
        self.model.eval()
        snap = {"kind": self.kind, "dataset": self.dataset, "step": self.step,
                "running": self.running, "history": self.history[-2000:],
                "window": WINDOW, "data": r(self.data[:N_SHOW])}
        if self.kind == "ar":
            snap.update(
                axis=r(AXIS), density=r(self.model.density_grid(AXIS), 4),
                fine=r(FINE), marginal1=r(self.model.marginal1(FINE), 4),
                samples=r(self.model.sample(N_SHOW)),
            )
        else:
            with torch.no_grad():
                mu, log_var = self.model.encode(self.data[:N_SHOW])
                snap.update(
                    beta=self.beta, sigma_x=r(self.model.sigma_x()),
                    samples=r(self.model.sample(N_SHOW)),
                    latent_mu=r(mu), latent_sigma=r((0.5 * log_var).exp()),
                    kl_points=r(0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum(-1)),
                    grid_lines=self._decoder_grid(),
                )
        self.model.train()
        return snap

    @torch.no_grad()
    def _decoder_grid(self):
        """Pushforward of latent grid lines through the decoder mean."""
        ticks = torch.linspace(-2.5, 2.5, 11)
        along = torch.linspace(-3, 3, 61)
        lines = []
        for t in ticks:
            for z in (torch.stack([torch.full_like(along, t), along], 1),
                      torch.stack([along, torch.full_like(along, t)], 1)):
                lines.append(r(self.model.decode(z)))
        return lines


trainers = {"ar": Trainer("ar"), "vae": Trainer("vae")}
app = FastAPI()


@app.middleware("http")
async def revalidate_pages(request, call_next):
    """Hashed assets may be cached; everything else is revalidated so rebuilds show up."""
    response = await call_next(request)
    if not request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class Control(BaseModel):
    action: str                 # start | pause | reset
    dataset: str | None = None
    beta: float | None = None


@app.get("/api/datasets")
def datasets():
    return list(DATASETS)


@app.get("/api/{kind}/state")
def state(kind: str):
    t = trainers[kind]
    with t.lock:
        snap = dict(t.snapshot)
    snap["running"] = t.running
    return snap


@app.post("/api/{kind}/control")
def control(kind: str, c: Control):
    t = trainers[kind]
    if c.action == "start":
        t.start()
    elif c.action == "pause":
        t.pause()
    elif c.action == "reset":
        t.pause()
        if t.thread:
            t.thread.join(timeout=2)
        t.reset(c.dataset or t.dataset, c.beta)
    if c.beta is not None and c.action != "reset":
        t.beta = c.beta          # live: change the tug of war while training
    return {"ok": True}


# --- AR queries --------------------------------------------------------------

@app.get("/api/ar/conditional")
def ar_conditional(x1: float):
    t = trainers["ar"]
    with t.lock:
        p = t.model.conditional(x1, FINE)
        p1 = t.model.marginal1(torch.tensor([x1]))
    return {"x1": x1, "fine": r(FINE), "p": r(p, 4), "p1": float(p1[0])}


@app.get("/api/ar/point")
def ar_point(x1: float, x2: float):
    t = trainers["ar"]
    with t.lock, torch.no_grad():
        lp1, lp2 = t.model.log_prob_parts(torch.tensor([[x1, x2]]))
    return {"lp1": float(lp1[0]), "lp2": float(lp2[0])}


@app.get("/api/ar/sample_one")
def ar_sample_one():
    t = trainers["ar"]
    with t.lock:
        x = t.model.sample(1)[0]
    return {"x1": float(x[0]), "x2": float(x[1])}


# --- VAE queries -------------------------------------------------------------

@app.get("/api/vae/encode")
def vae_encode(x1: float, x2: float):
    t = trainers["vae"]
    with t.lock, torch.no_grad():
        mu, log_var = t.model.encode(torch.tensor([[x1, x2]]))
        kl = 0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum(-1)
        rec = t.model.decode(mu)
    return {"mu": r(mu[0]), "sigma": r((0.5 * log_var[0]).exp()),
            "kl": float(kl[0]), "recon": r(rec[0])}


@app.get("/api/vae/decode")
def vae_decode(z1: float, z2: float):
    t = trainers["vae"]
    with t.lock, torch.no_grad():
        m = t.model.decode(torch.tensor([[z1, z2]]))
        s = t.model.sigma_x()
    return {"mean": r(m[0]), "sigma": r(s)}


dist = Path(__file__).parent / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
