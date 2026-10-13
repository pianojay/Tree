# Substrate Audit: Lighter Installs and a Monolithic Build

**Status:** Audit (2026-10-09). Decided in Review 6 (D16): keep the current substrate; tininess is an optional later effort.
**Parent:** [whitepaper.md](../whitepaper.md) · **Asked in:** owner's request after Review 5

**Historical record:** these measurements and Review 6 describe the original pilot. The 2026-10-13 [development roadmap](roadmap.md) reopens runtime selection for the fresh implementation; the earlier measurements remain reference evidence.

The question: can Tree run with a much smaller install than PyTorch, and could it ship as one self-contained program, like a game downloaded from itch.io?

## 1. What we use now (measured on this machine)

| Piece | Size | Notes |
|---|---|---|
| `.venv` (Python) | **754 MB** | torch alone 543 MB, numpy 34 MB |
| `web/node_modules` (build only) | 71 MB | not needed at run time |
| Built page `web/dist` | **4.7 MB** | almost all of it Plotly (about 4.6 MB) |
| Training speed (2→64→64→1 MLP, batch 256) | 1.2 ms/step | torch CPU, 4 threads |

The page is already a small static bundle. All the weight is on the Python side, and the models are tiny: two inputs and hidden layers of 64 units.

## 2. Measurements

The same network and task (2→64→64→1 tanh MLP, batch 256, Adam, XOR) on each candidate:

| Substrate | Install | ms/step | Works here? |
|---|---|---|---|
| PyTorch CPU (current) | 543 MB | **1.2** | yes |
| Our pure-Python scalar engine (`tree/engine.py`) | 0 | ~251 | yes, too slow for live views |
| tinygrad 0.14, default device (OpenCL) | 42 MB, no deps | ~235, after a **41 s** first-step compile | yes, but slow |
| tinygrad 0.14, `DEV=CPU` | 42 MB | — | **no**: needs a `clang` binary on PATH |
| tinygrad 0.14, `DEV=PYTHON` | 42 MB | — | **no**: permission error on Windows |
| Plain JavaScript in Node (Float32Array, hand-written backprop, no libraries) | **0** | **11** | yes, converges (loss 0.017) |

11 ms/step in plain JavaScript, without SIMD or Web Workers, matches the throughput the current server actually delivers once snapshots are included (about 60–100 steps/s).

## 3. The peer's work: knok (github.com/gmmyung)

Gyungmin Myung's public repositories contain a directly relevant stack:

- **knok:** "compile-time Rust tensor graphs backed by IREE". Graphs are written in `build.rs`, lowered through MLIR, compiled AOT by IREE during `cargo build`, and called as typed Rust functions. Static shapes, ranks 0–6. It has **experimental build-time value-and-gradient generation** for scalar losses.
- **knok-mnist-training:** trains an MLP with knok's static autograd. Host-side Rust does SGD and shuffling, and it ships an **egui app** that trains and runs inference. That is very close to an itch.io-style monolithic trainer.
- **knok-demo:** an egui app running compiled graphs (Mandelbrot, diffusion, waves, Life, particles).
- **eerie:** his Rust bindings to the IREE compiler and runtime. The IREE runtime can be ~30 KB on bare metal.

**Fit for Tree:**
- **Good:** static shapes suit us (fixed batch, fixed widths). The training loop stays on the host, so samplers like Langevin or ODE integration are plain Rust. egui gives a native window, and the compiled runtime is small.
- **Risks:**
  - eerie lists **Windows as unsupported**, and this is a Windows machine.
  - Building needs the `iree-compile` toolchain, pinned via a Nix shell.
  - Autograd is marked experimental, and the API is at v0.3.
  - Every graph shape is fixed at build time, so changing a width, the number of layers, or K means a rebuild.

It's a strong option if the peer is interested and development moves to macOS or Linux.

## 4. Options

| Option | Install for a user | Monolithic? | Effort | Fit |
|---|---|---|---|---|
| **A. Status quo:** PyTorch + FastAPI + web page | ~800 MB, Python and Node | no | done | Best for research iteration; heavy to share |
| **B. Lighter Python:** NumPy + hand-written or small autograd | ~40 MB | no | medium | Smaller, still needs Python; little gain for users |
| **C. Browser-native:** training in TypeScript inside the page | **0** (a ~1–5 MB page) | **yes, as a web app**: itch.io HTML5 upload, GitHub Pages, or a Tauri desktop wrapper | medium | **Best for "download and play"**; our models are small enough |
| **D. Rust native:** burn or candle (autodiff, CPU) + egui; burn and candle can also target WASM | one ~10–30 MB executable | **yes** | high | A genuine desktop app; Rust learning curve |
| **E. knok (peer):** Rust + IREE AOT graphs + egui | one small executable | **yes** | high, plus toolchain | Elegant and small; Windows unsupported, static shapes |
| tinygrad | 42 MB | no | low | Not usable on this machine as-is (see §2) |
| JAX CPU | ~100+ MB | no | low | No real saving over torch |

Notes:
- **C can use a library or not.** TensorFlow.js has full autograd in the browser, but its maintenance signals are mixed. A small hand-written tensor + autograd module in TypeScript (matmul, elementwise ops, sum, a tape) is a few hundred lines, in the same spirit as `tree/engine.py` but vectorized. Plotly (4.6 MB) could later be replaced by plain canvas drawing to bring the page under 1 MB.
- **D and E both produce the itch.io-style executable.** burn and candle can also compile to WASM, so one codebase could serve both desktop and web, though in-browser training with burn is less documented than inference.

## 5. Recommendation (for discussion)

1. **Keep A** as the research workbench. It's what we iterate on quickly.
2. **Prototype C** for sharing: move training into the page with a small TypeScript autograd. That keeps all the existing views, needs nothing installed, and publishes as an HTML5 project on itch.io. A Tauri wrapper turns the same page into a desktop download of a few MB if a "real app" is wanted.
3. **Talk to the peer about E.** knok already has the pieces (static autograd plus an egui trainer), and a Tree port would be a good real-world test for knok. It needs macOS or Linux, or Windows support in eerie.

## Sources

- gmmyung repositories: knok, knok-mnist-training, knok-demo, eerie (READMEs read 2026-10-09 via the GitHub API).
- tinygrad, PyTorch, and the plain-JS benchmark: measured locally (see §2).
- Burn WASM support: [Burn MNIST web example](https://github.com/WorldSEnder/burn/blob/main/examples/mnist-inference-web) (inference only; in-browser training not documented there).
- TensorFlow.js maintenance signals: [Snyk advisor, @tensorflow/tfjs](https://dev.snyk.io/advisor/npm-package/@tensorflow/tfjs), [@tensorflow/tfjs-core](https://snyk.io/advisor/npm-package/@tensorflow/tfjs-core) (third-party, dates unclear).
- tinygrad packaging notes: [T2 Linux package page](https://t2linux.com/packages/tinygrad).
