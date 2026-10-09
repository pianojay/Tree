# Tree

A miniature for making the abstract mathematics of generative models explicit and visible. See [whitepaper.md](whitepaper.md).

## Setup (Windows, CPU)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Demo (autoregressive, VAE, normalizing flow, GAN)

```bash
cd web && npm install && npm run build && cd ..
.venv/Scripts/python -m uvicorn server:app --port 8000
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
