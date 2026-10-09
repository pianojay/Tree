# Decision Log

Each entry records **what** was decided, **why**, and **where it came from**.

**Provenance key**

| Tag | Meaning |
|---|---|
| **[User]** | Stated by the project owner |
| **[Brainstorm]** | Recommended in the 2026-10-07 ChatGPT session and not contested |
| **[Draft]** | Proposed by Claude while drafting; not yet reviewed |

**Status values:** *Adopted*, *Open*, *Out of scope*, *Superseded*, *Withdrawn*.

---

## Adopted

### D1. The premise

- **Source:** [User] idea, [Brainstorm] framing, sharpened by [User] in Reviews 1 and 2.
- **Decision:** Tree is a miniature that **makes the abstract mathematics of generative models explicit**: pushforward measures, maps from samples to distributions and between distributions, latent spaces, energy fields, score and velocity fields, Langevin dynamics. It does this by making the input space visually traceable.
- **Corollary:** models are not required to succeed or to fail. Whatever happens must be sensibly and intuitively traceable to the principle that produced it.
- **Why:** at scale (e.g. DiT-XL/2 on 512×512×3 inputs), the abstract objects are still there but impossible to see. A miniature is what allows everything to be explicit.

### D2. The sample space is not circular

- **Source:** [User] raised the concern; [Brainstorm] recommended; [User] ruled it out of scope in Review 1.
- **Decision:** no torus and no circular hue. Pixels are Euclidean, either $\mathbb R$ or $[0,1]$ (see D13).
- **Why:** on a manifold, gradient-based dynamics need Riemannian counterparts. That would mix geometry effects into differences between families.
- **Record:** [background-circular-variables.md](background-circular-variables.md)

### D3. Two pixels first

- **Source:** [Brainstorm]
- **Decision:** two pixels is the primary case. How to render 1 and 3 pixels is deliberately left open: [User], Review 1, "do not think ahead too much".

### D4. Abstract nonsense over substrate

- **Source:** [User], [Brainstorm]
- **Decision:** every model gets only a tiny MLP unless its math requires something else. No engineering tricks, no powerful backbones. The project is about principles, not substrates.

### D5. One shared canvas, as the guiding idea

- **Source:** [Brainstorm]
- **Decision:** all families are viewed on the same space.
- **Scope (Review 1):** views are for debugging for now. What to show and how to render it are later questions, and UX/UI experiments are a separate track.

### D6. Distribution-level judgement; the metrics themselves are exploratory

- **Source:** [Brainstorm] caution against cosine similarity; [User] in Review 1: "we can try different metrics".
- **Decision:** judge distributions, not individual samples. Which metrics to use is a design space to explore. None are fixed.

### D7. The ground truth is curated, versioned, and iterated

- **Source:** [User], Review 1
- **Decision:**
  - $p_{\text{data}}$ is a sampling program, curated to be hard. It is *not* an analytic construction and it is not fixed early.
  - There is a ladder of versions: simple (for debugging), multimodal, challenging (iterated).
  - The training data is a finite subset of $p_{\text{data}}$.
- **Detail:** [ground-truth.md](ground-truth.md)

### D8. Autoregressive means pixel by pixel, on continuous values

- **Source:** [User], Review 1
- **Decision:** no discretization. Each conditional is a continuous density, and the output maps to hue.

### D9. Seven families, each reduced to its crux map

- **Source:** [User], Review 3. This supersedes Review 1's single "flow model" family.
- **Decision:**
  - The families are autoregressive, VAE, normalizing flow, GAN, energy-based, diffusion, and flow matching.
  - Each is represented by the map it learns, for example $x_{<t}\mapsto p_\theta(x_t\mid x_{<t})$ for autoregressive models.
  - Variants (MADE / RNADE / transformer; NICE / RealNVP / MAF) and backbone sizes are not immediate concerns. Small CPU-sized networks are enough.
- **Why:** the crux map is what gets visualized. Few dimensions are what make each concept a live, interactive object.
- **Detail:** [model-families.md](model-families.md)

### D10. A minimal Python implementation

- **Source:** [User], Review 1
- **Decision:** a very minimal Python implementation, in the spirit of Karpathy's llm.c. Implementing the principles and checking their mechanics comes before training them well.
- **Substrate (Review 4):** use libraries where they serve better; dependency minimalism is not the first concern. The earlier pure-Python foundation stays as a reference: a scalar autodiff engine (`tree/engine.py`), networks plus Adam (`tree/nn.py`), and tests under `tests/`.

### D11. MIT license

- **Source:** [User], Review 1

### D12. The display maps a scalar to a hue arc

- **Source:** [User], Review 1
- **Decision:** color is a map from the pixel value to a hue arc that does not close on itself (from $[0,1]$, or from $\mathbb R$ through a squashing map). It can be adjusted later. Color-vision accessibility is not a design input. Neither is audience.

## Open

### D13. $\mathbb R^d$ vs. $[0,1]^d$, and what replaces "the Gaussian"

- **Source:** [User], Review 1: "to be scrutinized"
- **Analysis:** [sample-space-and-priors.md](sample-space-and-priors.md)
- **Leaning [Draft]:** $\mathbb R^d$ with a squashing display map, so all model math stays textbook.

## Out of scope

- **Circular / toroidal sample space** (D2).
- **Audience-specific design** and **color-vision accessibility** (Review 1).
- **Engineering:** architectures, training tricks, performance.

## Superseded

- **D-S1. Internal $(\cos h,\sin h)$ encoding of hue** [Brainstorm]. Superseded by D2.

## Withdrawn: draft proposals rejected or retired in Review 1

These were spontaneous choices made while drafting v0.1 of the whitepaper. They are kept here so they are not reintroduced silently.

| Draft proposal | Why withdrawn |
|---|---|
| Ground truth as an analytic Gaussian mixture, with closed-form score, $p_t$, and flow-matching velocity | It made $p_{\text{data}}$ "pretty". The ground truth should be curated raw-data-like difficulty (D7). |
| A fixed Tree-2 layout: named pixel stops R/O/Y/G/B at 0.1…0.9, four named modes with set centers, widths, and weights, forbidden discs | The exact construction is not a present concern (D7) |
| Tree-1 and Tree-3 specifications, including a parity block for Tree-3 | Thinking too far ahead (D3) |
| "Crown/trunk" pixel meanings and season nicknames | Semantics are examples, not specifications |
| Named color stops, OKLab interpolation, end colors, out-of-gamut hatching | Replaced by a plain hue-arc map (D12) |
| AR with 64-bin discretization | Rejected (D8) |
| "Population" training mode with fresh samples every batch | Training data is a finite subset (D7) |
| Numeric defaults for the tiny MLP (3×64 SiLU, Adam 1e-3, batch 512, 5 seeds, parameter budgets within 2×); a list of banned techniques; Fourier time features | Premature detail |
| Per-family specifics: VAE latent sizes and β; GAN loss variants; EBM training variants (CD, oracle-Z quadrature); diffusion VP/ε-prediction; flow-matching couplings and reflow | Premature detail. Families are kept at the level of their principle. |
| Long lists of predicted phenomena per family, and a table of predicted answers to the five questions | Could hide what would otherwise be discovered. Only the brainstorm's anticipations are kept. |
| A verdict rule based on the 99% highest-density region (HDR) | Metrics are exploratory (D6) |
| A fixed metric suite: sample counts, grid size, thresholds, calibration samplers | Metrics are exploratory (D6) |
| A detailed visualization spec: layers, sprite gallery, comparison mode, interactions, Tree editor, light/dark themes | Views are for debugging; what to render is a later question (D5) |
| Milestones with exit criteria; a proposed repository layout | Replaced by the iterative roadmap in the whitepaper |
| Python reference first, browser playground later | Python adopted (D10). The playground question is dropped. |
| GMM, KDE, and oracle-sampler baselines | Not requested; can come back if debugging needs them |

---

## Review log

### Review 1 (2026-10-07), by the owner, on whitepaper v0.1

1. $p_{\text{data}}$ should be the opposite of pretty: challenging enough that each model shows failures derived from its own principle. It should not be learnable perfectly.
2. The exact construction of $p_{\text{data}}$ isn't a present concern. The color examples show possible dynamics, not semantics to program. Think of raw data. The ground truth will be highly curated and iterated.
3. Circular color space is out of scope. Use a $[0,1]\to$ hue map, or possibly $\mathbb R\to$ hue. The Gaussian is awkward on $[0,1]$, and the alternatives (triangular, Beta, logit-normal, truncated normal, or just pretending) need scrutiny.
4. Minimal Python implementation, in the spirit of llm.c.
5. Answers to the open questions:
   - Q1: $\mathbb R^d$ or $[0,1]^d$.
   - Q2: Python.
   - Q3: AR is continuous, pixel by pixel, with no discretization.
   - Q4: the training data is a subset of $p_{\text{data}}$.
   - Q6: audience is not a design input.
   - Q7: just a "flow model".
   - Q8: accessibility is not considered.
   - Q9: MIT.
6. Roadmap:
   - Have several versions of the ground truth (simple, multimodal, challenging).
   - Metrics are to be explored.
   - Implementing the principles and checking that they work matters most; training them is another problem.
   - Views are for debugging.
   - Don't plan 1D/3D rendering ahead of time.
   - The write-up comes later.

### Review 2 (2026-10-07), by the owner, on the premise

1. Tree provides intuitive, live visualization of model principles: flows, latent space, scalar energy fields, and so on.
2. At scale we forget the "abstract". Pushforward measures, maps from a sample to a distribution and from one distribution to another, and Langevin dynamics are what generative models are. Tree makes the abstract (mathematical) explicit, which is possible because the input space is visually traceable.
3. **Correction to Review 1's framing:** this is not "the model must fail". If a failure occurs, it must be sensibly and intuitively traceable. Everything is explicit, and creating a miniature is what allows that. That is the point of Tree.

### Review 3 (2026-10-07), by the owner, on the model families

1. "Abstract nonsense" is meant as a parody of, or homage to, the mathematicians' phrase. It does not refer to category theory.
2. There are seven families: autoregressive, VAE, normalizing flow, GAN, energy-based, diffusion, flow matching. Backbones are not a concern; small CPU-sized networks are enough.
3. Each family is reduced to the map it models. The owner's notes on autoregressive, VAE, normalizing flow, GAN, and energy-based models are quoted in [model-families.md](model-families.md).
4. The goal is to make the concepts alive: concrete, visualizable, interactable objects. That becomes possible when there are only a few dimensions.
5. Claude was asked to use imagination here. The "living objects" in model-families.md are those imagined candidates.

### Review 4 (2026-10-08), by the owner, on the next plan

1. **Next goal:** a working demo for the autoregressive and VAE families, including visualization. Experimenting is exploring: see how it looks before going bigger.
2. **CPU only,** but libraries are welcome where they are better. Dependency minimalism is not the first concern.
3. **Git:** track Tree as a **private** GitHub repository. Pushing the documents is fine.
4. **Location:** the project moved to `D:\Claude\Tree`, because Windows Defender's Controlled Folder Access blocked shell tools from writing under `Documents`. The old copy in `Documents` is stale.
