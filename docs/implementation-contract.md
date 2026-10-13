# Tree implementation contract

**Phase 1, 2026-10-13:** use the setup of `main`, as requested by the owner. Other mathematical and presentation choices can be iterated later.

The model world is R², with plotting window [-3, 3]². Coordinates remain unbounded; `web/src/colors.ts` renders each coordinate using the existing logistic map to a hue arc. Pixel 2 remains darker and desaturated. The existing eleven dataset sampling programs, finite training set of 2,000 points, batch size 256, and small two-hidden-layer networks of width 64 remain the starting setup.

| Family | Initial model | Initial optimizer |
|---|---|---|
| Autoregressive | Eight Gaussian mixture components; tanh conditional MLP; continuous factorization | Adam, lr 0.003, beta1 0.9, beta2 0.999 |
| VAE | Two-dimensional Gaussian encoder and diagonal Gaussian decoder; tanh MLPs; beta 1 | Adam, lr 0.003, beta1 0.9, beta2 0.999 |
| Normalizing flow | Three affine autoregressive layers with alternating order and identity initialization | Adam, lr 0.002, beta1 0.9, beta2 0.999 |
| GAN | Existing leaky-ReLU generator and discriminator; non-saturating objective; one D update per G update | Separate Adam optimizers, lr 0.001, beta1 0.5, beta2 0.999 |

Model clamps, sampling equations, and display conventions follow the original Python models. GAN's description remains as it is; Phase 5 does not expand its interpretation.

Comparison fixtures use identical exported parameters, batches, and noise from the original `main` models. The production TF.js model operations themselves are tested against those fixtures. PyTorch is a supported comparison and optional server runtime. TF.js and jax-js are evaluated afresh; historical branch code is reference material only.

Training runs outside the browser UI thread. A reset stops training, waits for queued operations, disposes the model's owned state, and establishes a new generation. Each snapshot identifies its generation and step; obsolete state polls are ignored. Display queries use their own deterministic random stream so hovering does not change subsequent training noise. Pausing waits for the current bounded chunk; changing tabs pauses the previously displayed model.

Each model exposes training, sampling, mathematical queries, and snapshot construction separately from rendering. The UI retains equal-size planes and the existing family interactions. Queries can be coalesced, rendering updates are serialized, and initialization errors are visible. Worker requests have timeouts and reject when the worker fails.

Baseline data and initial weights are exported from `main` for reproducibility across runtimes. Resetting at the default configuration reproduces those weights and data; other supported K or layer counts use a seeded initializer with the same initialization distribution. Training randomness and diagnostic randomness have separate seeds.

Numerical displays distinguish learned quantities from reference quantities. The VAE's ordinary negative ELBO is reconstruction plus KL; its training objective is reconstruction plus beta times KL. Evidence and posterior displays use stated quadrature domains and resolutions; sampled expectations report Monte Carlo uncertainty. Their difference estimates total variational inference error, not amortization error alone.

Completion through Phase 5 requires fixture parity, normalization and flow invariants, bounded tensor ownership across reset and queries, browser training/control checks, VAE diagnostics, and a production build served without `node_modules`. CPU/WASM remains required; a dedicated GPU is not required.

**Validated implementation:** TensorFlow.js core 4.22.0 with registered gradients, CPU/WASM backends, a serialized worker service, and Canvas views. PyTorch server mode remains supported. Current jax-js is a viable and faster WASM comparison candidate; the owner's preference selects TF.js for the working application. See [runtime comparison](runtime-comparison.md).
