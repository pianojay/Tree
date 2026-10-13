# Three-runtime comparison with GUI rendering

**Measured:** 2026-10-13. **Scope:** this branch's built application, all four existing model families. This supersedes the separate exploratory GUI runs for the comparison below; it does not resolve the historical jax-js WASM timing reversal.

## Result

Median training updates per second, with the real Canvas views and their queries active:

| Family | PyTorch server | TF.js WASM | jax-js WASM | jax-js / TF.js |
|---|---:|---:|---:|---:|
| AR | 212.6 | 36.5 | 64.5 | 1.77× |
| VAE | 196.7 | 31.0 | 54.9 | 1.77× |
| Flow | 107.3 | 17.9 | 31.4 | 1.75× |
| GAN | 128.3 | 23.8 | 55.0 | 2.31× |

The advantage survives application overhead, but jax-js is not near PyTorch throughput in this test: PyTorch is about 2.3–3.6× faster. PyTorch uses two CPU threads; the browser WASM runtimes use one. This is a comparison of the delivered application paths, not identical low-level execution or equal thread budgets.

The three-run throughput ranges are:

| Family | PyTorch | TF.js | jax-js |
|---|---:|---:|---:|
| AR | 200.1–235.6 | 34.5–37.4 | 62.3–64.7 |
| VAE | 177.2–199.9 | 30.5–32.7 | 54.9–57.2 |
| Flow | 102.4–111.5 | 17.3–19.2 | 30.8–32.2 |
| GAN | 118.0–139.3 | 23.4–24.1 | 52.2–55.2 |

## Interaction and rendering

Median paused query latency includes worker messaging or HTTP and response materialization:

| Query | PyTorch ms | TF.js ms | jax-js ms |
|---|---:|---:|---:|
| AR conditional | 9.2 | 1.0 | 3.6 |
| VAE encode | 9.4 | 1.4 | 2.5 |
| Flow trace | 11.3 | 2.3 | 2.9 |
| GAN generate | 6.8 | 0.5 | 1.2 |
| VAE inspect, 64² grid and 2048 q samples | 28.6 | 51.6 | 40.7 |

TF.js is faster on these small interactive queries. jax-js is faster on the larger VAE inspection workload. Neither observation establishes latency during continuous dragging while training; the paused probes intentionally avoid that contention.

Frame callbacks are around 54–56 FPS across the runtimes. This is `requestAnimationFrame` cadence, **not 54–56 model redraws per second**. The real view redraws about eleven times during each six-second measurement, consistent with the 500 ms polling target. Whole-view timing includes Canvas draws and any awaited query (notably AR's conditional), so it must not be described as pure rendering time.

Median of each run's mean snapshot construction time:

| Family | PyTorch ms | TF.js ms | jax-js ms |
|---|---:|---:|---:|
| AR | 10.3 | 54.4 | 54.5 |
| VAE | 8.7 | 19.1 | 26.4 |
| Flow | 20.8 | 171.7 | 127.2 |
| GAN | 9.5 | 16.3 | 31.7 |

Snapshot cost materially narrows the training-only advantage and varies by family. It remains a useful optimization target for both browser adapters.

## Protocol

- One production build, served by the worktree's PyTorch server at the same origin for all three engines. The worker paths still perform browser training; they do not call Python model endpoints.
- Intel i7-1065G7 / Iris Plus system, Windows, Chromium 155. Each record retains the viewport, device-pixel ratio, user agent and visibility. All runs were visible.
- Three runtime rounds: **PyTorch → TF.js → jax-js**, **TF.js → jax-js → PyTorch**, **jax-js → PyTorch → TF.js**. Family order also rotates by round. Only one benchmark runs at a time; completed tabs/workers are closed. No Node tests/builds ran during the timed sequences.
- Every family resets to the default model and `multimodal` dataset: 2,000 training examples, batch 256, hidden width 64. Two seconds of warm-up, pause, then six seconds of training with ordinary view polling/rendering. Elapsed measurement includes start/pause request completion and the final snapshot; update counts come from paused states.
- Frontend polling and both backend snapshot targets are 500 ms. The existing adaptive backoff to at least twice snapshot cost remains active. `TREE_SNAPSHOT_SECONDS=0.5` overrides PyTorch's ordinary 0.35 s default for this test; it does not change the default server configuration.
- Twelve warm paused query probes after one initial probe; VAE inspection warms once and is timed three times per run. These probe warm-ups are separate from training warm-up.

Native backend differences remain: PyTorch trains in five-step thread chunks and reads a scalar every update; workers use two-step event-loop chunks and log every five steps. PyTorch rounds snapshots and transports JSON; workers use structured cloning with full float values. The benchmark preserves these application paths. The reported `snapshotJsonBytes` is a common JSON-size proxy, not actual worker wire bytes.

Wall-clock warm-up produces different trained step counts. Browser runtimes share data, initialization and RNG code; PyTorch uses its original RNG/sampling stream. These measurements rank throughput and responsiveness, not training quality or matched parameter trajectories. The windows are short, three repetitions provide limited uncertainty information, and sustained memory/browser coverage remain open gates.

## Reproduce and inspect

From `web/`, run `npm ci`, `npm run build`. From the root, start the optional Python server using the installed environment with `TREE_SNAPSHOT_SECONDS=0.5`. Then open `/gui-benchmark.html?engine=server&round=1`, `/gui-benchmark.html?engine=tfjs&round=1`, and `/gui-benchmark.html?engine=jax-js&round=1`, sequentially. Repeat the counterbalanced orders above with round 2 and 3. Each page runs all four families and exposes its JSON result in `#results`; close it before the next run.

[Raw nine-run results](results/gui-three-way.json) retain frame intervals, render/poll/query samples, snapshot statistics, step counts and environment metadata. [Summary JSON](results/gui-summary.json) records aggregates. After saving the nine results, `node scripts/summarize-gui-benchmark.mjs` validates their count/visibility/cadence and recomputes the summary.

The benchmark adds snapshot timing counters and a read-only `/api/{kind}/metrics` response to each runtime. `npm test` and the production build pass with this instrumentation. No model equations or training objectives were changed.
