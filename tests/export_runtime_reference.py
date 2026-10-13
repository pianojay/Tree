"""Export main's actual models/data, plus matched runtime comparisons.

Run from the root: .venv/Scripts/python -m tests.export_runtime_reference
The ordinary demo does not require Python after these assets have been built.
"""
import base64
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from tree.data import DATASETS, sample
from tree.models.ar import AR
from tree.models.vae import VAE
from tree.models.flow import Flow
from tree.models.gan import GAN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'web' / 'public'


def pack(t):
    a = np.ascontiguousarray(t.detach().numpy(), dtype='<f4')
    return {'shape': list(a.shape), 'b64': base64.b64encode(a.tobytes()).decode()}


def named(model, gradients=False):
    out = {}
    for name, value in model.named_parameters():
        if gradients:
            value = value.grad if value.grad is not None else torch.zeros_like(value)
        m = re.fullmatch(r'(.*)\.(\d+)\.(weight|bias)', name)
        if m:
            name = f'{m[1]}.{"W" if m[3] == "weight" else "b"}{int(m[2]) // 2}'
            if m[3] == 'weight':
                value = value.T
        name = re.sub(r'^layers\.(\d+)\.', r'l\1.', name)
        out[name] = pack(value)
    return out


def make(kind):
    torch.manual_seed(0)
    return {'ar': AR, 'vae': VAE, 'flow': Flow, 'gan': GAN}[kind]()


def loss(model, kind, x, nz, group='all'):
    if kind == 'vae':
        original = torch.randn_like
        torch.randn_like = lambda _: nz['eps']
        try:
            return model.loss(x)[0]
        finally:
            torch.randn_like = original
    if kind == 'gan':
        return (model.d_loss(x, model.G(nz['z_d'])) if group == 'D'
                else model.g_loss(model.G(nz['z_g'])))
    return model.loss(x)


def export():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    models = {k: make(k) for k in ('ar', 'vae', 'flow', 'gan')}
    baseline = {'seed': 0, 'params': {k: named(m) for k, m in models.items()},
                'datasets': {k: pack(torch.tensor(sample(k, 2000, seed=0))) for k in DATASETS},
                'heldout': {k: pack(torch.tensor(sample(k, 512, seed=1))) for k in DATASETS}}
    (OUT / 'runtime-baseline.json').write_text(json.dumps(baseline, separators=(',', ':')))
    data = torch.tensor(sample('multimodal', 2000, seed=0))
    gen = torch.Generator().manual_seed(378)
    batches = [data[torch.randint(0, len(data), (256,), generator=gen)] for _ in range(12)]
    noise = [{k: torch.randn(256, 2, generator=gen) for k in ('eps', 'z_d', 'z_g')}
             for _ in batches]
    records = []
    for kind, model in models.items():
        # Warm up the actual main model so flow checks exercise non-identity paths.
        lr, betas = (0.002 if kind == 'flow' else 0.001 if kind == 'gan' else 0.003), ((0.5, 0.999) if kind == 'gan' else (0.9, 0.999))
        def optimizers():
            return ([torch.optim.Adam(model.D.parameters(), lr=lr, betas=betas),
                     torch.optim.Adam(model.G.parameters(), lr=lr, betas=betas)]
                    if kind == 'gan' else [torch.optim.Adam(model.parameters(), lr=lr, betas=betas)])
        opts = optimizers()
        def step(i):
            total = 0.0
            for group, opt in zip(('D', 'G') if kind == 'gan' else ('all',), opts):
                opt.zero_grad(); model.zero_grad()
                value = loss(model, kind, batches[i % 12], noise[i % 12], group)
                value.backward(); opt.step(); total += value.item()
            return total
        for i in range(20):
            step(i)
        initial = named(model)
        checks = {}
        for group in ('D', 'G') if kind == 'gan' else ('all',):
            model.zero_grad()
            value = loss(model, kind, batches[0], noise[0], group)
            value.backward()
            checks[group] = {'loss': value.item(), 'grads': {k: v for k, v in named(model, True).items() if group == 'all' or k.startswith(group + '.')}}
        opts = optimizers()
        curve = [step(i) for i in range(12)]
        final = named(model)
        # Timing includes a scalar readback, just like browser timings below.
        timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            for i in range(60):
                step(i)
            timings.append((time.perf_counter() - t0) * 1000 / 60)
        records.append({'kind': kind, 'config': {'k': 8, 'layers': 3, 'permute': True, 'beta': 1}, 'initial': initial,
                        'batches': [pack(x) for x in batches], 'noise': [{k: pack(v) for k, v in n.items()} for n in noise],
                        'checks': checks, 'curve': curve, 'final': final,
                        'torch': {'version': torch.__version__, 'threads': 2, 'ms_step': timings}})
        print(kind, 'torch ms/step', timings)
    (OUT / 'runtime-fixtures.json').write_text(json.dumps(records, separators=(',', ':')))


if __name__ == '__main__':
    export()
