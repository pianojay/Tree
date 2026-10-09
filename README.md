# Tree

A miniature for making the abstract mathematics of generative models explicit and visible. See [whitepaper.md](whitepaper.md).

## Requirements

- Python 3.12+ (tested with 3.14) and Node.js 20+ (tested with 24); CPU only, no GPU needed.
- First setup takes about 4 minutes, mostly downloading PyTorch.

## Setup

```bash
gh repo clone pianojay/Tree && cd Tree      # or: git clone https://github.com/pianojay/Tree.git
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # macOS/Linux: .venv/bin/python
cd web && npm install && npm run build && cd ..
```

## Demo (autoregressive, VAE, normalizing flow, GAN)

```bash
.venv/Scripts/python -m uvicorn server:app --port 8000      # macOS/Linux: .venv/bin/python
```

Open http://localhost:8000. Python trains the models on the CPU in the background; the page polls snapshots and queries the models on hover and drag. Restart the server after pulling changes. For page development with hot reload, run `npm run dev` in `web/` alongside the server and open http://localhost:5173.

Toy datasets (`tree/data.py`): simple, multimodal, tree, x_shape, sine, two_moons, ring8, rings, spiral, checkerboard, crescent.

## Tests

```bash
.venv/Scripts/python -m tests.test_engine   # gradient check of the pure-Python reference engine
.venv/Scripts/python -m tests.test_learns   # perceptron vs. MLP on XOR (pure Python, ~2 min)
```

## License

MIT
