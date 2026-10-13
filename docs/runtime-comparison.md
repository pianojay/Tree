# Tree runtime comparison and verification

**Date:** 2026-10-13. **Scope:** development through Phase 5 using the mathematical setup of `main`. The historical experiment branches were references; the new production and comparison adapters use `web/src/runtime/math.ts`.

## Working runtime

Tree uses TensorFlow.js core 4.22.0 with registered gradients and CPU/WASM backends. Training and mathematical queries run in a Web Worker; Canvas draws the existing family views. The default application requires neither Python nor a CDN. The original PyTorch server remains an explicit option through `?engine=server`.

Current jax-js 0.1.26 is competitive and faster than TF.js WASM in these matched training tests. TF.js is selected as the working runtime in line with the owner's stated preference, not because the measurements establish it as fastest. jax-js remains a tested comparison candidate.

## Matched model tests

The fixtures are exported from the actual `main` PyTorch models after twenty warm-up steps. Both browser adapters receive the same float32 weights, batches, noise, and optimizer settings. Checks compare loss and gradients, then twelve optimizer updates and final parameters. Timing uses three repeats of sixty updates, including a scalar readback; first-step compilation is reported separately.

The models have two hidden layers of width 64, batch size 256, and the original AR/VAE/flow/GAN settings. PyTorch uses two CPU threads. TF.js WASM uses one thread. jax-js uses its JIT training step on WASM. No architecture search or generative-quality ranking is implied by these backbone measurements.

**Machine:** Intel Core i7-1065G7, Intel Iris Plus Graphics, Windows, Chromium 155, eight reported logical processors. Backend initialization times exclude module download. The results are local measurements, not cross-device guarantees.

| Family | PyTorch CPU median ms/update | TF.js WASM median ms/update | jax-js WASM median ms/update |
|---|---:|---:|---:|
| AR | 2.10 | 6.76 | 2.06 |
| VAE | 2.25 | 8.94 | 2.86 |
| Normalizing flow | 3.84 | 14.04 | 4.61 |
| GAN | 2.89 | 13.66 | 3.23 |

All four families pass in both WASM adapters. The loss/gradient tolerance is 1e-4 relative to the reference scale; the twelve-step curve/parameter tolerance is 1e-3. Observed worst gradient relative error is about 1.6e-5 and worst TF.js final-parameter relative error about 5.8e-5. These checks validate the port and short optimizer trajectories; they do not require bit-identical long training runs.

A separate jax-js worker check passes the same four model fixtures and verifies the input gradient of a known quadratic. This establishes worker feasibility without making jax-js a second GUI runtime in the delivered application.

The fixed historical jax-js WebGL GAN failure is recorded in the [bug memo](jax-js-bugs.md), with its existing upstream issue and merged fix. Current WebGL activation and complete GAN checks pass. WebGL is much slower than WASM on this machine and is not selected for Tree.

## Production GUI checks

The production directory was served by an ordinary static server with no `node_modules` available and no isolation headers. All four families trained, paused without further step advancement, reset to a new generation, and answered repeated queries without increasing their retained tensor count. Repeated resets also kept tensor counts stable. AR and flow held-out NLL improved in the short training checks.

The same verification page passed against the PyTorch server over HTTP. Persistent tensor counts measure owned model/data/optimizer state rather than operating-system memory; counts between the two libraries are not directly comparable.

| Five-second AR drawing check | TF.js worker | PyTorch server |
|---|---:|---:|
| Animation frame rate | 54.3 FPS | 55.5 FPS |
| Training throughput with snapshots and drawing | 102.7 updates/s | 369.9 updates/s |
| Median AR conditional query while paused | about 0.7 ms | about 6.1 ms |
| Last measured data-plane drawing call | 1.8 ms | 3.9 ms |

The drawing values describe a short local check with the new renderer. They are not a rendering-duration distribution or sustained performance guarantee. Raw model timings above exclude HTTP and snapshots; this table measures an application path that includes them.

The built directory is approximately 3.9 MB uncompressed, including comparison fixtures, verification code, all three WASM variants and shared baseline data. The normal GUI loads approximately 2.0 MB uncompressed when using the SIMD WASM variant. Compression by a hosting server can reduce transferred bytes; the local static-server measurement does not assume it. The runtime worker bundle is approximately 1.06 MB, reduced from the initial rebuild's 1.92 MB by using TF.js core rather than its unused Layers/converter packages.

## Mathematical and lifecycle checks

`npm test` passes:

- AR marginal and conditional probability normalization on a broad integration domain.
- Flow round trips, finite-difference Jacobians and change-of-variables determinants.
- Analytic Gaussian KL and the relationship between beta-weighted loss and ordinary ELBO.
- VAE evidence quadrature convergence at multiple resolutions against a decoder with analytic independent likelihood.
- GAN player update isolation, fresh inputs for additional discriminator updates, and discriminator input gradients against finite differences.
- Repeated reset and query ownership for all four families, ending with zero live TF tensors after disposal.

The VAE GUI also exposes likelihood, the estimated model posterior and encoder approximation at a selected data point. It distinguishes ordinary ELBO from the weighted training objective and labels finite-domain quadrature, the number of encoder samples, and approximate Monte Carlo uncertainty. GAN's existing explanatory text is unchanged, as requested; its mechanics are verified without claiming to complete that description.

## Reproduction and records

From the root, regenerate assets with `.venv/Scripts/python -m tests.export_runtime_reference`. The committed baseline and fixtures make the ordinary browser build independent of Python. From `web/`, run `npm test` and `npm run build`.

Use the development page `/compare.html?lib=tfjs&backend=wasm` or `/compare.html?lib=jax-js&backend=wasm` for matched browser timings. `probe=leaky` and `probe=readback` isolate WebGL operations. `probe=worker` verifies all four jax-js model fixtures and an input-gradient operation in a worker. Keep GPU timing pages visible.

Serve `web/dist` and open `/verify.html` for the worker path, or serve it through PyTorch and open `/verify.html?engine=server`. Machine-readable results are preserved in [validation/runtime-results.json](validation/runtime-results.json). The main benchmark fixtures live in `web/public/runtime-fixtures.json`; no historical-branch result substitutes for these checks.

Phases 6–7 remain pending. EBM, diffusion and flow matching are not implemented by this rebuild, and broader dataset curation, browser coverage and extended delivery testing remain future work.
