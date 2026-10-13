# Application parity contract

**Scope:** translate the existing AR, VAE, normalizing-flow and GAN application. Keep model equations, default architecture, datasets, color mapping, controls, numerical labels and rendering unchanged. EBM, diffusion, flow matching and expanded GAN explanations are outside this experiment.

## Shared implementation

`runtime/math.ts` remains the single mathematical implementation. `runtime/initialization.ts` supplies the same seeded initializer and RNG streams. `runtime/session-core.ts` supplies common sampling, snapshots, diagnostic grids, VAE evidence inspection, configuration, and serialized training/control behavior. TF.js and jax-js inject tensor operations and a model adapter; the UI and message protocol are shared.

Every array passed into jax-js is either consumed or explicitly retained. The tensor facade owns one reference per wrapper, borrows inputs to operations and releases unreturned intermediates on scope exit. Reads borrow references. Adam state belongs to each model, with separate step counters for GAN's two players. Frozen players do not advance optimizer state; extra discriminator steps use fresh batches/noise.

JIT training receives parameters, optimizer state, batch, noise and bias-correction scalars explicitly. Inference compilation receives all arrays, including parameters, explicitly; model weights must never become captured constants. Inference graph variants are bounded to 64 entries. A changed mathematical configuration replaces that player's training graph. Shape changes must either select a valid graph or trigger compilation, never reuse an incompatible graph.

## Equivalence

- Actual main PyTorch fixtures: relative scaled loss/gradient error below `1e-4`; twelve-step trajectory/final-parameter error below `1e-3`, as in main's comparison.
- Initial and twelve-update TF.js/jax-js snapshots and queries: equal schema; finite numerical results; per-element error at most `2e-4 + 1e-4 * max(abs(a), abs(b))`. These allowances include float32 reduction/Adam differences. Stochastic queries use identical deterministic streams.
- Mathematical checks: AR normalization, flow inverse and finite-difference Jacobian, VAE evidence at three resolutions against an analytic decoder, GAN update isolation and finite-difference input gradients.
- Configurations: alternate AR mixture size, flow layer count/order, VAE beta changes and GAN freeze/multiple-D controls.

GAN's extended control parity uses the common warmed PyTorch fixture to isolate update semantics. Input-gradient fields are piecewise smooth: after independent training, small weight differences can move a point across a leaky-ReLU boundary and produce a larger derivative difference. Compare derivatives at matched weights/inputs before diagnosing a backend bug; do not loosen all numerical tolerances to conceal that case. The retained investigation is described in [findings](findings.md).

Training and inspection use separate random streams. Equivalent long training means comparable behavior across seeds; bit-identical stochastic trajectories are not required.

## Lifecycle and delivery

Both runtimes must train outside the UI thread, stop after pause, reset to a new generation, ignore stale UI responses, and expose errors/timeouts. Repeated query/reset tests must preserve owned-array counts. Also check model parameter reference counts so wrapper counts cannot conceal borrowed-reference leaks.

Runtime metadata reports owned array and Adam state counts/bytes, plus compiled graph counts. It excludes backend allocator reservations and JIT constants; it is not proof of flat browser/process memory. Sustained browser memory evaluation remains a separate gate.

Production checks must use locally served built assets, with no Python requirement for browser mode. The normal jax-js worker must not import TensorFlow. The comparison path may load both libraries.

## Adoption gates

Passing parity establishes a usable experimental port. Adoption on main additionally requires matched full-application performance tests, sustained training/reset/configuration stress and memory observations, intended-browser coverage, and investigation of the old/new timing reversal. Record cold compilation separately from warm queries and training; include snapshot/diagnostic costs and asset transfer sizes. A faster training microbenchmark alone does not satisfy these gates.
