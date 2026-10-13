# jax-js bug memo

**Checked:** 2026-10-13. **Current comparison version:** `@jax-js/jax` 0.1.26.

## Historical WebGL GAN discrepancy

The old runtime branch used jax-js 0.1.25. Its WebGL GAN had a large loss and gradient discrepancy while WASM passed. An existing upstream report describes the matching failure: comparisons such as `less(x, 0)` returned false on WebGL, so `nn.leakyRelu` returned negative inputs unchanged instead of applying the requested slope.

- Report: [jax-js issue 227](https://github.com/ekzhang/jax-js/issues/227).
- Fix: [jax-js PR 229](https://github.com/ekzhang/jax-js/pull/229), replacing floating-point textures used to hold raw words with integer textures. Boolean/integer bit patterns could otherwise become subnormal floats and be flushed to zero by the graphics implementation.
- Both are closed. A duplicate issue is unnecessary.

The fresh 0.1.26 WebGL activation check uses inputs `[-5, -1, -0.1, 0.1, 1, 5]`, slope 0.2, and a differentiated sum. Outputs and gradients match their expected values. The current four-family WASM comparison also passes matched losses, gradients, twelve optimizer updates, and final parameters.

The complete current WebGL GAN fixture also passes: maximum gradient relative error about 2.3e-6, loss-curve relative error about 3.3e-7, and final-parameter relative error about 7.4e-7. The old GAN discrepancy is not reproduced on 0.1.26. Its WebGL throughput here is much slower than WASM, which is a performance result rather than a correctness bug.

## Reproduction and reporting policy

Use `web/compare.html` with `lib=jax-js`, a backend, and an optional `families` filter. The `probe=leaky` route isolates activation values and gradients; `probe=readback` distinguishes synchronous from asynchronous GPU readback. Keep GPU pages visible during timing and record visibility; a throttled background page is not sufficient evidence of a numerical library bug.

For any new reproducible failure, retain a minimal example, expected and actual values, library version, browser, graphics environment, and fixture or seed. Check existing issues first, then file an upstream GitHub issue as authorized by the owner. Keep Tree implementation mistakes separate from upstream defects.
