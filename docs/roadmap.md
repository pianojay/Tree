# Tree development roadmap

**Updated:** 2026-10-13. **Direction:** fresh development on `main`, requested by the owner. **Status:** Phases 1–5 implemented and checked within the agreed scope; Phases 6–7 remain pending. The initial setup follows `main`. TensorFlow.js WASM is the working browser runtime; PyTorch remains supported and current jax-js is retained for comparison. See the [contract](implementation-contract.md), [comparison](runtime-comparison.md), and [bug memo](jax-js-bugs.md).

| Phase | Current result |
|---|---|
| 1 | Main's world, model shapes, datasets and optimizer settings adopted in the implementation contract |
| 2 | Fresh matched PyTorch, TF.js and jax-js comparison; working packaged TF.js worker; upstream jax-js bug recorded with its existing fix |
| 3 | AR training, conditional/likelihood queries, sequential sampling, reproducible resets, held-out NLL and Canvas views |
| 4 | VAE maps and beta control, separate ordinary ELBO and weighted objective, likelihood/posterior inspection with labeled numerical estimates |
| 5 | Flow stages, inverse traces, composition and permutation; existing GAN mechanics and interactions preserved, with its description left unchanged |
| 6–7 | Pending; additional families and broader curation/delivery work are not claimed complete |

Tree will make the mathematics of generative models concrete and interactive in a small Euclidean world. The rebuild should preserve that purpose while making the demo easy to obtain and run. A reader should be able to inspect what a model represents, how its objective updates it, and how it samples, without needing to understand the implementation.

The existing Python and Vite demo is the historical working baseline. The `bench/tfjs-vs-jaxjs` and `tfjs-ui` branches are referential only. Development starts from the requirements below; their implementations will not be merged as the foundation. Professor Minhyuk Sung's [VAE lab](references/README.md) is credited separately as an interaction reference.

## Purpose and scope

Two continuous coordinates are the primary case. Keep model geometry separate from its colorful rendering. Circular sample spaces, one- and three-dimensional extensions, larger architectures, and generative-quality leaderboards remain outside the initial rebuild.

The demo must run without a dedicated GPU. Require a usable CPU or WASM path; browser GPU acceleration may be an optional backend. Keep networks small enough for the mathematical objects and their behavior to remain inspectable.

Use small networks and a finite training set from a reproducible sampling program. Start with simple and multimodal data, then curate challenging distributions when there is working machinery to inspect. Success and failure are both useful when their causes are visible. Do not build datasets around predetermined verdicts about particular families.

The seven intended families remain autoregressive, VAE, normalizing flow, GAN, energy-based, diffusion, and flow matching. Each view should answer the same questions:

1. What does this model currently believe the data distribution is?
2. Where is that belief represented?
3. How does it produce a sample?
4. What changes during one training update?
5. Which information does its objective supply, and which information does it omit?

## Development sequence

Each phase produces something usable and inspectable. The completion conditions are checks for development progress, not a fixed schedule. Adjust the sequence when a working demo reveals a more useful next experiment.

### Phase 1 Define the new foundation

The owner specified that Phase 1 follow `main`. The [contract](implementation-contract.md) adopts its real-valued two-dimensional setup, color mapping, datasets, model sizes, and optimizers. Other choices can be iterated later.

Specify how a model exposes training, sampling, density or fields where available, and interactive queries. Separate these operations from rendering and lifecycle control so a renderer can inspect a model without implementing a second version of its mathematics. Define tensor ownership and cancellation when resetting, changing datasets, or switching views.

**Completion:** the contract identifies the chosen world, preserves the five questions, and distinguishes settled requirements from open implementation choices. The reference materials and historical baseline remain accessible.

### Phase 2 Select a runtime with a small working experiment

Compare the PyTorch server backbone, TensorFlow.js and jax-js afresh against the operations Tree needs through Phase 5: matrix operations, parameter gradients, random sampling, input-gradient queries and worker execution. Keep a browser application without a Python install as the initial delivery target. PyTorch remains an optional server and development oracle. Additional EBM/diffusion/flow-matching operations belong to Phase 6.

Use the same saved initial parameters, batches, noise, and optimizer settings for numerical comparisons. Check losses, gradients, several updates, and model-specific invariants. Set tolerances from the calculation's scale and precision requirements, and investigate failures rather than widening thresholds merely to obtain a pass.

Measure worker startup, steady training, query latency, memory behavior, and a representative plot update. Record backend, library version, hardware, browser, synchronization, repeats, and first-run compilation separately. Measure the complete downloaded runtime, including WASM and worker assets, rather than only the HTML file.

Build and serve the production artifact from a clean directory with no `node_modules`. Test all required assets and the chosen hosting headers. A development-server success is insufficient. Record fallback behavior and show a visible error if initialization fails.

**Completion:** a recorded runtime choice with reproducible numerical evidence, a working worker, and a production artifact that trains without Python or missing assets. If neither candidate meets the contract, record the limitation and assess another substrate before expanding the models.

### Phase 3 Build the shared world and autoregressive demo

Implement reproducible datasets, training-set and held-out sampling, the display map, and common equal-size planes. Begin with the simplest useful renderer; choose Canvas, Plotly, or another renderer through measured interaction and drawing cost.

Implement a continuous autoregressive model with an expressive conditional density. Show the marginal, the conditional distribution as a coordinate moves, the joint density, likelihood decomposition, and sampling one coordinate at a time. Make the number of mixture components an inspectable model choice.

Train in a worker. Bound the rate of snapshots and queries, coalesce obsolete hover requests, and prevent overlapping plot updates. Reset and pause must have clear behavior, and snapshots should identify the model state they describe.

**Completion:** one complete vertical slice covering dataset selection, training, pause, reset, conditional inspection, and sequential sampling. Check probability normalization, the chain rule, held-out behavior, repeated lifecycle operations, and the production artifact. Record interaction latency and memory stabilization on a named machine.

### Phase 4 Build the VAE inspection tools

Implement the encoder, stochastic decoder, reparameterized update, and ordinary ELBO. Add a beta control while distinguishing the ordinary ELBO from the weighted training objective when beta differs from one.

Show equal-level data, latent, and generated-data planes, with the maps between them. Selecting a data point should reveal its encoder distribution and reconstruction; selecting a latent point should reveal the decoder distribution. Preserve color identity across corresponding points.

Use Professor Sung's lab as a reference for inspecting likelihood, model posterior, encoder approximation, and evidence. Compute these from the same model that trains. Label quadrature and Monte Carlo approximations, their domains, and relevant uncertainty. Distinguish total variational inference error from amortization error; an ellipse mismatch alone does not separate them.

**Completion:** encoder and decoder interactions, reconstruction and prior-KL terms, beta behavior, and posterior comparison are inspectable. Validate analytic Gaussian KL, the reparameterized gradient, decoder sampling, and numerical convergence of evidence estimates on controlled examples. Keep richer diagnostics optional if they would block responsive basic interaction.

### Phase 5 Add normalizing flow and GAN

For normalizing flow, show the base, intermediate stages, and data plane, with a grid pushed through each map. Support forward and inverse inspection, density bookkeeping, composition, and permutation. Check round trips, Jacobian determinants, and change of variables.

For GAN, show latent points, generated points, the discriminator field, and the generator's signal at generated points. Expose alternating updates and freezing either player. Keep adversarial losses distinct from measures of sample quality. Verify which variables each update changes and that input-gradient queries release their tensors.

The owner requested that GAN's incomplete description be left as it is. This phase preserves the current text and verifies the existing mechanics; it does not claim to finish the GAN explanation.

**Completion:** both families can be traced through training and sampling, and can be compared on the same fixed datasets. Passing mechanics tests and responsive interactions count as implementation progress; perfect fitting is not required.

### Phase 6 Add energy diffusion and flow matching

Add one family at a time, with a complete sampling path and view before proceeding to the next.

| Family | Objects to inspect | Mechanics to verify |
|---|---|---|
| Energy-based | Energy, relative density, positive and negative training samples, Langevin paths | Input gradients, update signs, noise and step-size conventions, confinement or stated domain, and mixing on a controlled distribution |
| Diffusion | Forward noising, time-conditioned denoising or score, reverse sampling paths | Noise schedule, parameterization conversions, endpoint treatment, and reverse-sampler convergence |
| Flow matching | Training pairs, interpolation, marginal velocity, integrated trajectories | Target velocity, time conditioning, ODE integration, and step-count convergence |

Historical benchmark stand-ins can suggest operation-level tests, but do not count as these implementations or as validation of sampling.

**Completion:** all seven families have working mechanics, sampling, and inspectable mathematical objects on the simple and multimodal cases. Distinguish learned fields from any oracle fields drawn for comparison.

### Phase 7 Curate experiments and deliver Tree

Iterate challenging ground truths, retaining version, seed, and construction. Develop distribution-level observation tools as questions arise: mode coverage, rare modes, density mismatch, transport, or mass in selected regions. Compare behavior without assuming equal numerical losses mean equal model quality.

Review every mathematical caption and approximation. Document the model choices that affect interpretation. Test the actual distribution artifact on a clean browser setup, including missing assets, initialization failures, pause and reset, sustained training, and repeated interaction. Bundle dependencies locally if offline use becomes a requirement; a CDN-backed HTML file is not an offline artifact.

**Completion:** a documented way to obtain and run Tree, credited references, reproducible examples, and measured download size, startup, responsiveness, and memory behavior. Retire the Python runtime requirement from the user setup instructions only when the new artifact meets this condition.

## How to resume work

The next development phase is Phase 6, subject to a new request to proceed beyond Phase 5. Keep using the same model operations in validation and in the GUI. Consult the contract and comparison record before changing runtime or mathematical conventions.

Keep the current `main` demo runnable until the replacement slice exists. Record new decisions in [decisions.md](decisions.md), preserving the historical review log. Backend choices and benchmark results should carry dates and tested versions; a branch name or passing build is not a completion claim.
