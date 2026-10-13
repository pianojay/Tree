# Tree

A miniature for making the abstract mathematics of generative models explicit and visible. See [whitepaper.md](docs/whitepaper.md).

## jax-js experiment branch

This `jax-js` branch translates the current TF.js application to jax-js while preserving the equations, GUI, and optional PyTorch server. The browser defaults to jax-js WASM; use `?engine=tfjs` for the retained TF.js worker or `?engine=server` for PyTorch. The experiment checks numerical equivalence, lifecycle behavior, and practical performance before any decision to change `main`. Its contract, findings, and remaining acceptance gates live in [docs/jax-js/](docs/jax-js/README.md).

The rebuild through Phases 1–5 runs four model families in a TensorFlow.js WASM worker, with Canvas views and optional PyTorch server mode. See the [roadmap](docs/roadmap.md), [implementation contract](docs/implementation-contract.md), and [runtime comparison](docs/runtime-comparison.md). The historical TF.js and jax-js branches were deleted; their findings remain documented. See the [documentation index](docs/README.md) for reading material and development records.

## Reference VAE lab

The separate [VAE interactive lab](docs/references/vae_2d_lab.html) is by **Professor Minhyuk Sung, KAIST**, [mhsung@kaist.ac.kr](mailto:mhsung@kaist.ac.kr), created with Claude. See its [attribution and provenance](docs/references/README.md). Tree was independently devised by its owner.

## Requirements

- Node.js 24 is tested and supports the verification scripts. No dedicated GPU is needed.
- Python 3.12+ (tested with 3.14) is optional: use it for the PyTorch server or regenerating reference assets.

## Setup

```bash
gh repo clone pianojay/Tree && cd Tree      # or: git clone https://github.com/pianojay/Tree.git
cd web
npm install
npm run build
npm run preview
```

Open http://127.0.0.1:4173. Training happens in a worker; WASM and reference data are served locally from the build. Python and a CDN are not required. For development, run `npm run dev` in `web/` and open http://127.0.0.1:5173.

The built `web/dist` directory can also be served by an ordinary static server. WASM uses one CPU thread and does not require cross-origin isolation headers.

## Optional PyTorch server

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # macOS/Linux: .venv/bin/python
.venv/Scripts/python -m uvicorn server:app --port 8000      # macOS/Linux: .venv/bin/python
```

Open http://127.0.0.1:8000/?engine=server. Python trains in background threads; the same views query it over HTTP. Restart the server after changes. With the development server, use http://127.0.0.1:5173/?engine=server.

Toy datasets (`tree/data.py`): simple, multimodal, tree, x_shape, sine, two_moons, ring8, rings, spiral, checkerboard, crescent.

## Tests

```bash
cd web && npm test
```

`npm test` checks normalization, flow inverses and Jacobians, VAE KL and evidence against analytic examples, GAN update isolation, and tensor ownership. Open `/verify.html` on the built application for worker training, lifecycle, query, memory, held-out NLL, and drawing checks; add `?engine=server` when serving it through PyTorch.

Regenerate the common data and matched fixtures from the root:

```bash
.venv/Scripts/python -m tests.export_runtime_reference
```

Open `/compare.html?lib=tfjs&backend=wasm` or `/compare.html?lib=jax-js&backend=wasm` on the development server to compare current runtimes using the same model equations. The commands `npm run compare:tfjs` and `npm run compare:jax` run CPU and WASM comparisons in Node. See the [jax-js bug memo](docs/jax-js-bugs.md) for the fixed historical WebGL bug and reproduction routes.

The original pure-Python checks are retained:

```bash
.venv/Scripts/python -m tests.test_engine   # gradient check of the pure-Python reference engine
.venv/Scripts/python -m tests.test_learns   # perceptron vs. MLP on XOR (pure Python, ~2 min)
```

## License

Tree code: MIT. The separate VAE reference lab has its authorship recorded in [references/README.md](docs/references/README.md).
