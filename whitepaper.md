# Tree: A Generative-Model Microscope

**Status:** Draft v0.3 · 2026-10-07 (after Reviews 1–2)
**Origin:** Brainstorm of 2026-10-07 ([record](docs/brainstorm-log.md), [raw export](ChatGPT-Circle%20Representation%20Priors-20261007-2051.md)) · **Decisions:** [docs/decisions.md](docs/decisions.md)

**Development restart (2026-10-13):** the owner requested a fresh implementation effort on `main`. The [development roadmap](docs/roadmap.md) governs that work; the TF.js and jax-js branches are referential. Professor Minhyuk Sung's separate VAE lab is [credited here](references/README.md).

**Current implementation:** Phases 1–5 use the setup of `main`, a TensorFlow.js WASM worker and Canvas views, with the original PyTorch server supported. AR, VAE, flow and existing GAN mechanics are implemented; the other three families remain pending. [Comparison and verification](docs/runtime-comparison.md).

> Make $p_{\text{data}}$ simple enough to see, but complicated enough that learning it is nontrivial.

> Everything is explicit, and a miniature is what allows it. If a failure occurs, it must be sensibly and intuitively traceable.

---

## Abstract

**Tree** is a miniature that makes the abstract mathematics of generative models explicit and visible.

- **The data.** Each sample is an "image" of one, two, or three pixels. Each pixel holds a single scalar, shown as a hue. A *Tree* is a sample from $p_{\text{data}}$: a hand-curated, deliberately non-trivial distribution over those few pixels.
- **The models.** Miniature versions of seven families learn it: autoregressive, VAE, normalizing flow, GAN, energy-based, diffusion, and flow matching. Each is reduced to its crux map, with a small network.
- **The aim.** To show the objects these principles are made of, live: pushforward measures, maps from a sample to a distribution and from one distribution to another, latent spaces, scalar energy fields, score and velocity fields, Langevin dynamics. Whatever a model does, success or failure, can then be traced to its principle by eye. The input space is small enough to draw, and nothing else is in the way.

## 1. Premise

Generative modeling is, at heart, abstract mathematics:

- pushforward measures
- maps from a sample to a distribution, and from one distribution to another
- energy landscapes and Langevin dynamics
- score and velocity fields

We tend to forget this once we are running DiT-XL/2 on 512×512×3 inputs for hours on specialized GPUs. The abstract objects are still there, but they live in a space no one can look at, buried under architectures, conditioning, normalization, optimizer details, and compute.

Tree **makes the abstract explicit.** It shrinks the input space until it can be traced visually. Every abstract object then becomes something you can draw and watch: a measure is a cloud of points, a map is a warped grid, a field is a set of arrows, a dynamic is a moving particle.

Four commitments:

1. **Explicitness through miniature.** The miniature is not a simplification for its own sake. It is what *allows* everything to be explicit.
2. **Traceability, not an outcome.** Tree does not require models to succeed or to fail. It requires that what happens can be traced, sensibly and intuitively, to the principle that produced it.
3. **Abstract nonsense, not substrate.** The phrase is borrowed as a parody of, or homage to, the mathematicians' expression. It doesn't refer to category theory. The pixels and hues give just enough meaning that `(green, orange)` feels like an object rather than a pair of numbers. Everything else is the mathematics of each principle, taken as it is.
4. **Non-trivial data.** Low dimensionality is not the same as a simple distribution. $p_{\text{data}}$ is curated, raw-data-like, and not pretty, so that the principles have something non-trivial to act on, and their behavior is worth tracing.

## 2. The Tree world

- **Pixels.** 2 pixels is the primary case, so every model can be drawn on one square. 1 and 3 pixels come later; how to show them is deliberately left open.
- **Sample space.** Euclidean. Circular hue (a torus) is **out of scope**, because its geometry would contaminate the comparison ([background](docs/background-circular-variables.md)). Whether it is $\mathbb R^d$ or $[0,1]^d$, and what then plays the role of "the Gaussian", is still **under scrutiny** ([sample space and priors](docs/sample-space-and-priors.md)). The current leaning is $\mathbb R^d$, so every family's math stays textbook.
- **Display.** A scalar-to-hue map onto an arc that doesn't close on itself (from $[0,1]$ directly, or from $\mathbb R$ through a squashing map). It is purely a rendering, can be adjusted freely, and never enters any model.
- **Ground truth.** $p_{\text{data}}$ is a *sampling program*. It is curated to be challenging, versioned, and iterated, never fixed early. Training data is a finite subset of it. There is a ladder of versions: **simple** (to check that models learn at all), **multimodal**, and **challenging** (the actual object of study). Details: [ground truth](docs/ground-truth.md).

Since we write the program, we know $p_{\text{data}}$ far better than any real dataset allows: unlimited samples, and in 2D a density estimate as precise as we want. That is the "microscope" luxury.

## 3. Model families

There are seven families. Each is cut down to its **crux map**, the map it actually learns. The backbone doesn't matter: small networks on a CPU are enough. In a few dimensions, every crux map can be drawn, probed, and perturbed while it trains. Concepts become objects. Details and imagined "living objects" for each family: [model families](docs/model-families.md).

| Family | Crux map | What becomes alive |
|---|---|---|
| Autoregressive | $x_{<t}\mapsto p_\theta(x_t\mid x_{<t})$ | A conditional curve riding along as you drag $x_1$; the joint rebuilt from slices |
| VAE | $x\mapsto q_\phi(z\mid x)$ and $z\mapsto p_\theta(x\mid z)$ | Posterior ellipses; reconstruction vs. KL as a tug of war; posterior collapse as an event |
| Normalizing flow | Diffeomorphism $T_\theta$ with $p_X=(T_\theta)_\#\,p_Z$ | A rubber sheet with ink; the Jacobian as stretching and squeezing |
| GAN | $G_\theta:z\mapsto x$ against $D_\phi$ | A foldable sheet; the referee's field; the fight and mode collapse in real time |
| Energy-based | $p\propto e^{-E}$; Langevin kernel | Landscape and lake; push-down and push-up pins; a marble with jitter |
| Diffusion | $(x,t)\mapsto\nabla\log p_t(x)$ | Ink dispersing; a stack of landscapes; "where did this come from?" |
| Flow matching | $(x,t)\mapsto v_\theta(x,t)$ | Straight training lines averaged into non-crossing streams; one step lands on the mean |

There are three kinds of map:

- **sample → distribution**
- **distribution → distribution** (pushforward)
- **distribution ↔ field**

In textbooks the families look like unrelated equations. In Tree they are different arrangements of these maps, aimed at the same colored blobs on one space.

## 4. The five questions

Stop every model at the same points and ask:

1. What does this model currently believe a Tree is?
2. Where is that belief stored: density, energy, discriminator, latent decoder, score, or velocity?
3. How does it produce one new Tree from that belief?
4. What does a failure look like geometrically?
5. What information does its training objective actually provide?

## 5. Observation and metrics

Generation means matching distributions, not reproducing individual samples. Per-sample similarity, cosine included, is not the yardstick: it has no pairing to compare against, and it rewards collapse.

Beyond that, metrics are an **open design space**. Candidates to try include divergences against the (well-sampled) true distribution, transport distances, the mass placed in regions we curated as forbidden, and mode coverage. Since the goal is to trace behavior rather than rank models, metrics are instruments, not scores.

## 6. Views

The guiding idea from the brainstorm: **every family on the same canvas**. For now, views exist to **debug** the system, and rough is fine. What should be shown and how to render it are questions to answer later, as a separate track from the core.

## 7. Implementation

- **Small implementations of the principles.** The fresh implementation follows `main`'s model setup and uses a TensorFlow.js WASM worker with Canvas views. The Python server remains supported; jax-js remains a comparison option. Browser use does not require Python. The pure-Python autodiff engine in `tree/` stays as a reference.
- **Implementing before training.** The first job is implementing each principle and checking that its mechanics work on the simple ground truth. Training the models *well* is a separate, later problem.
- **License:** MIT ([LICENSE](LICENSE)).

## 8. Scope

**In scope:** the abstract objects of generative principles, made explicit and traceable on curated low-dimensional data.

**Out of scope:**

- engineering, architectures, and training tricks
- benchmarks and leaderboards
- circular or manifold sample spaces
- designing for a particular audience
- color-vision accessibility

## 9. Roadmap

This is iterative, not a sequence of fixed milestones.

The [development roadmap](docs/roadmap.md) turns the research direction below into a fresh implementation sequence. It includes runtime selection, numerical verification, interactive model slices, and distribution of the application.

1. **Simple ground truths** (simple, then multimodal), plus the crudest views needed to debug.
2. **Minimal implementations** of the seven principles, each checked to *work* on the simple case.
3. **Curate challenging ground truths.** Iterate $p_{\text{data}}$ until the families' behavior is non-trivial and worth tracing, and keep every version.
4. **Explore observation tools and metrics,** continuously, as needs show up.
5. **Later:** a textual description of what this toy is, 1D and 3D cases, and UX/UI experiments.

## 10. Open questions

| # | Question | Notes |
|---|---|---|
| ~~Q1~~ | ~~$\mathbb R^d$ or $[0,1]^d$? Which distribution plays the role of the Gaussian?~~ | Phase 1 follows main: $\mathbb R^2$, Gaussian reference distributions, and the existing squashing display. Other choices may be iterated later. |
| Q2 | The display map: hue-arc span, and the squashing function if $\mathbb R^d$ | One option: use the CDF of the reference distribution, so pure noise renders as uniformly random hue |
| Q3 | AR conditional density family (continuous, multimodal) | Follows from Q1: a Gaussian mixture on $\mathbb R$, or a Beta / logit-normal mixture on $[0,1]$ |
| ~~Q4~~ | ~~Which flow model first?~~ | Resolved in Review 3: normalizing flow and flow matching are separate families, and which variant to use is not an immediate concern |
| ~~Q5~~ | ~~Gradients: PyTorch autograd, or NumPy with hand-written backprop?~~ | Resolved in Review 4: use libraries where they are better (for example PyTorch on CPU, NumPy, Plotly). The pure-Python engine (`tree/engine.py`) is kept as a reference. |
| Q6 | What counts as "it works" for a minimal implementation on the simple ground truth? | Needed before step 2 of the roadmap is done |

## 11. Documents

| Document | Purpose |
|---|---|
| [docs/decisions.md](docs/decisions.md) | Decision log, review log, and withdrawn draft proposals |
| [docs/roadmap.md](docs/roadmap.md) | Current development sequence for the fresh implementation on main |
| [references/README.md](references/README.md) | Credit for Professor Minhyuk Sung's VAE lab and historical experiment references |
| [docs/ground-truth.md](docs/ground-truth.md) | What the ground truth is for and how it is kept: curated, versioned, a ladder |
| [docs/sample-space-and-priors.md](docs/sample-space-and-priors.md) | Analysis: $\mathbb R^d$ vs. $[0,1]^d$, and candidate priors |
| [docs/model-families.md](docs/model-families.md) | The seven crux maps, and how each family's concepts become live objects |
| [docs/background-circular-variables.md](docs/background-circular-variables.md) | Static record: circular representations, von Mises, harmonics, why the torus is out of scope |
| [docs/brainstorm-log.md](docs/brainstorm-log.md) | Static record of the originating brainstorm |
