# Translation findings

**Date:** 2026-10-13. **Status:** usable experimental four-family port; adoption gates remain open.

## Implementation

The application defaults to a jax-js WASM worker. `?engine=tfjs` selects the retained TF.js worker; `?engine=server` preserves HTTP/PyTorch mode. Both browser adapters use common equations, seeded initialization and session behavior. The existing Canvas views and explanatory text, including GAN's unfinished description, are unchanged.

The production jax-js adapter supports loss/gradient checks, jitted Adam updates, changing VAE beta, separate GAN optimizer counters, freeze controls, multiple discriminator updates with fresh inputs, all snapshots/queries and the VAE inspector. Inference kernels are compiled with explicit array arguments so parameter changes cannot leave a cached query using stale weights. The inference cache retains at most 64 graph variants.

The comparison page now tests the production adapter, rather than relying solely on the earlier comparison-only implementation. Older standalone probes remain available to reproduce upstream operations.

## Validation

`npm test` runs retained TF.js mathematical checks, the production jax-js fixture/invariant/lifecycle checks and TF.js/jax-js full-session parity. The checks cover:

- All four actual-main PyTorch loss/gradient fixtures and twelve Adam updates/final parameters.
- Full initial and twelve-update snapshots plus AR, VAE, flow and GAN queries, including VAE posterior/evidence inspection.
- Alternate AR K, flow layer count/order/datasets, VAE beta changes and GAN freeze/multiple-D controls.
- AR normalization, flow round trips/Jacobians, analytic VAE evidence at 32/64/96 resolutions and GAN finite-difference input gradients.
- Repeated resets/queries, unchanged parameter reference counts, separate diagnostic/training RNG, training/pause and zero facade-owned tensors after disposal.

The production browser verification passes on Chromium 155 with WASM for both runtimes. AR/flow held-out NLL decreases; all four families train, pause, query and reset. A separate GUI interaction trained GAN to step 3132 with live real/fake verdicts, then opened the VAE inspector successfully. These are finite local checks, not a sustained-memory or cross-browser guarantee.

The [final built-application check](results/browser-final.json) also passes after the GAN diagnostic fields were completed. Its five-second AR drawing check reaches about 56 FPS and 63 updates/s, with 43 cached inference graphs and stable owned-array counts across reset/query repetitions. The different values from the exploratory runs reinforce the need for repeated controlled measurements.

## Performance observations

The initial translation compiled training but executed snapshots/queries eagerly. Its short AR drawing check reached about 38 updates/s and 61 FPS. Compiling inference and dataset gathering raised the next short check to about 72 updates/s at 61 FPS. The TF.js control run in this worktree reached about 37 updates/s at 61 FPS. AR query median fell from about 16 ms to about 3 ms; first-query compilation still creates an observable cold cost.

These exploratory runs are saved in [browser-initial.json](results/browser-initial.json). They were collected during development, with other verification work occurring nearby, and are not a controlled final performance ranking. They show why training-only timing cannot predict the full application. Main's earlier 103 updates/s TF.js result and this branch's 37 updates/s result also demonstrate substantial run-to-run/environment sensitivity.

The first production-adapter training comparison passes all four fixtures. Its warm timings are several milliseconds per update; cold first updates take roughly 0.13–0.18 seconds in that run. This differs from the earlier comparison-only jax-js numbers and should be measured again under controlled conditions before adoption.

The built jax-js worker is approximately 357 KB uncompressed; the retained TF.js worker is approximately 1.06 MB. The jax-js worker embeds its runtime WASM. Those worker sizes exclude the application, baseline data and optional comparison assets, and are not total compressed transfer sizes. TF.js comparison/fallback assets remain in the build but are not imported by the jax-js worker.

## GAN derivative-field discrepancy

An extended comparison from default initialization passed initial and twelve-update snapshots, then found a gradient-field difference after freeze/multiple-D changes. Maximum parameter difference was about `2.56e-4`. At generated point 24, one discriminator activation crossed zero: approximately `-2.18e-7` in TF.js versus `1.88e-6` in jax-js. The input gradients consequently differed by roughly `0.003`.

After assigning the TF.js weights to the jax-js model and evaluating the identical input, maximum gradient difference was `1.49e-8`. This case is independent float32 trajectory divergence across a leaky-ReLU boundary, rather than evidence of an incorrect input-gradient port. Extended control parity therefore uses the matched warmed fixture; the default-initialization case remains preserved instead of being silently removed or addressed by weakening all tolerances.

Reproduce with `node scripts/diagnose-gan-parity.mjs` from `web/`. The retained [diagnostic record](results/gan-controls.json) includes the point, both gradients, parameter difference, crossing activation and aligned-weight result. No new upstream issue was filed for this case.

## Remaining adoption gates

1. Reconcile the historical 0.1.25 experiment with the current implementation under one timing harness, varying implementation and library version separately. The WebGL bug fix does not explain the WASM timing reversal.
2. Measure cold/warm training, snapshots, VAE inspection and hover queries under controlled, repeated conditions. Report ordinary application transfer sizes separately from the full build.
3. Exercise sustained training, repeated configuration/reset churn and browser/process memory. Owned-array counts exclude backend reservations and compiled constants.
4. Validate intended browsers/devices and failure/recovery behavior. WASM is the tested backend; GPU support is not an adoption claim.

Main has not been switched to jax-js. The experiment remains independently reviewable on this branch.
