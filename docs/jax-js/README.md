# jax-js application experiment

This branch starts from main commit `7a504a3` and ports the four existing families to jax-js 0.1.26. Main's mathematical setup and family explanations are retained. TF.js remains available for comparison and PyTorch server code is unchanged.

**Working direction (2026-10-13):** the owner selected PyTorch + jax-js. jax-js is the working browser runtime; TF.js is a retained comparison baseline. See the [upstream assessment and maintenance policy](upstream-assessment.md). Main integration is a separate operation.

- [Contract](contract.md): scope, ownership, numerical equivalence, and acceptance gates.
- [Findings](findings.md): implementation changes, measured results, and unresolved questions.
- [GUI comparison](gui-comparison.md): repeated three-runtime measurements with the actual views.

## Run

Follow the repository setup: `cd web`, `npm ci`, `npm run build`, then `npm run preview -- --port 4174 --strictPort`. The separate port avoids interfering with a main-worktree preview.

Open http://127.0.0.1:4174 for jax-js, http://127.0.0.1:4174/?engine=tfjs for TF.js. PyTorch still uses `?engine=server` through its server or the development proxy.

`npm test` runs the retained TF.js tests, production jax-js tests, and full session parity tests. `npm run compare:jax` now benchmarks the production jax-js adapter on WASM. The built `/compare.html?lib=jax-js&backend=wasm` compares the same production adapter in the browser; `/verify.html` checks the default worker, and `/verify.html?engine=tfjs` checks TF.js.

All new experimental documentation and saved results belong in this directory. Earlier comparison documents describe main's baseline and have not been rewritten to claim completion of this experiment.
