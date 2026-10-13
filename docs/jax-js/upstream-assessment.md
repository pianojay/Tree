# jax-js maintenance and adoption assessment

**Checked:** 2026-10-13. **Decision:** the owner selected PyTorch + jax-js as Tree's working pair. jax-js is the browser runtime; PyTorch remains the numerical reference and optional server. TF.js is retained as comparison evidence rather than a co-equal development target. This records the direction on this branch; it does not merge this work into main.

## Upstream evidence

The maintainer is **Eric Zhang, `@ekzhang`**. The [repository](https://github.com/ekzhang/jax-js) was created on 2025-02-04 and remains active, with its latest push and `jax/v0.1.26` release on 2026-10-13. Its version is still 0.1.x. Repository age, star counts and rapid responses cannot establish a probability of future bugs.

GitHub REST metadata over 2026-07-15 through the check date shows:

- **80 main-branch commits**, attributed to seven GitHub accounts. `dtran24` authored 41 and `ekzhang` 32; five other accounts authored seven collectively. Activity is not limited to the owner.
- **Eight core jax releases**, from v0.1.19 through v0.1.26. Release intervals vary; this is evidence of continuing delivery rather than a support guarantee.
- **18 reported contributors**. Eric accounts for 865 of 938 summed GitHub contributor counts, about **92.2%** historically. These API counts are not an exact census of all repository commits. The substantial second contributor in the recent window reduces, but does not eliminate, maintainer concentration risk.
- **29 successful and one failed workflow run** among the latest 30 main-branch runs fetched. This is workflow-level status, not test coverage percentage or proof of absence of bugs.

The [CI workflow](https://github.com/ekzhang/jax-js/blob/main/.github/workflows/ci.yaml) builds, type-checks, lints and tests. Platform jobs use Ubuntu/macOS, Chromium/Firefox/WebKit, Node and Deno. The [test configuration](https://github.com/ekzhang/jax-js/blob/main/vitest.config.ts) disables Firefox WebGPU on GitHub Actions and excludes one tokenizer test on WebKit. Coverage instrumentation is disabled by default. Upstream platform testing is a positive signal, but it cannot replace Tree's testing on intended browsers/drivers.

The dated [metadata snapshot](results/upstream-metadata.json) was collected with `gh api`: repository, contributors, releases, paginated main commits, issues updated since the cutoff, and recent main workflow runs. The issue selection uses **updated since**, so older issues appear too; it is not a sample of only newly opened issues. The endpoints were read sequentially and are not a transactional snapshot. Public GitHub counters may differ slightly as caches/activity change.

## Handling of defects relevant to Tree

Your [WebGL report #227](https://github.com/ekzhang/jax-js/issues/227) was opened at 2026-10-12 13:15:31 UTC. Eric replied at 2026-10-13 05:31:38 UTC; [fix #229](https://github.com/ekzhang/jax-js/pull/229) merged at 05:44:44 UTC and v0.1.26 released at 05:47:15 UTC. That is about **16.5 hours from report to release**, followed by Tree's successful activation/gradient verification. This is substantive responsiveness, not merely issue closure.

Your [Optax/JIT report #228](https://github.com/ekzhang/jax-js/issues/228) remains open. Eric acknowledged it at 2026-10-13 05:33:30 UTC. Acknowledgement is not a fix. Tree's current production adapter uses its own matched, fixture-tested Adam update and does not depend on `@jax-js/optax`, so this issue does not block the tested path. Replacing that update with Optax would require a new check.

Other recent closed numerical/performance reports include scalar division (#225), infinite-input logaddexp (#223), power edge cases (#221), uint32 subtraction (#216), matmul-gradient allocation (#150), and run-ahead synchronization (#151). Among seven closed issues created in the window, closure delays range from about ten hours to 6.4 days, with a median about 2.4 days. This small **closed-only** sample excludes unresolved reports and must not be treated as a response SLA or a forecast of resolution speed.

The older [leak-checker request #81](https://github.com/ekzhang/jax-js/issues/81) remains open. That reinforces why Tree needs its own ownership/reference checks and browser memory observations. It does not establish that the current Tree implementation leaks.

## Interpretation and working policy

The evidence supports **an active, technically engaged, small upstream project**, not an abandoned dependency. It also supports less redundancy and a shorter compatibility track record than a large established framework. Trust responsiveness enough to collaborate and report reproducible defects; make correctness and version adoption depend on Tree's tests.

The owner's selection is reasonable for Tree: the four-family port passes its sanity checks, and the [repeated GUI comparison](gui-comparison.md) finds a 1.75–2.31× training advantage over TF.js. PyTorch remains materially faster, small TF.js queries were faster, and longer-term stability is still unknown. This is a deliberate working-runtime choice with recorded uncertainties, not a claim of no tradeoffs.

Keep jax-js **pinned at 0.1.26** until a proposed upgrade passes the fixture, query, ownership, production-browser and representative GUI checks. Keep the shared equations/runtime boundary and PyTorch oracle. The retained TF.js code need not track every new feature; label its supported scope when that diverges. No new upstream issue was filed during this metadata assessment.

## Additional data that should change the decision

1. **Sustained operation:** 30–60 minute browser sessions per family, repeated resets/configuration churn, ownership counts plus actual browser/process memory and recovery from worker errors. Warm allocator/cache growth must be distinguished from continued growth after repeated identical cycles.
2. **Platform coverage:** Tree's parity and interactive controls on intended Chromium/Firefox/WebKit environments. Upstream CI does not validate Tree or all hardware backends. Keep WASM as the working backend; promote GPU paths only after separate checks.
3. **Next-family operators:** before UI development, check the exact autodiff and array primitives needed by each new family against analytic examples/PyTorch. In particular, EBM sampling and gradient-through-gradient training may exercise capabilities not covered by the current four families.
4. **Upgrade history:** record whether actual dependency upgrades preserve the application, how often adapter fixes are needed, and time spent debugging confirmed upstream regressions. This provides a project-specific maintenance-cost estimate.

Continue with the pinned version while these checks pass. Treat a new-version regression as a reason to reject that upgrade, not automatically abandon the runtime. Reconsider jax-js if required operations fail on the pinned working version, sustained ownership/memory problems cannot be bounded, target browsers cannot be supported, or repeated upstream workarounds dominate model/UI development. A period of maintainer inactivity alone is insufficient reason to replace a pinned version that still works.
