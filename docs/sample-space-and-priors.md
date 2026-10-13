# Sample Space and Priors

**Status:** Under review. Nothing here is decided yet (whitepaper Q1–Q3).
**Parent:** [whitepaper.md](whitepaper.md)

**Settled so far:** the sample space is **not circular**. That is out of scope; see [background-circular-variables.md](background-circular-variables.md).

**Still open:** $\mathbb R^d$ or $[0,1]^d$, and what plays the role of "the Gaussian" in each.

---

## 1. Where a "prior" actually enters

"Prior" covers several different objects. Only some of them live in *data space*, and only those care whether data space is $\mathbb R^d$ or $[0,1]^d$.

| Family | Distribution in **latent space** (free to choose) | Distribution in **data space** (affected by the choice) |
|---|---|---|
| AR | — | Each conditional $p(h_i\mid h_{<i})$ needs a continuous density family over the pixel's range |
| VAE | Prior $p(z)$, posterior $q(z\mid x)$ | Decoder likelihood $p(x\mid z)$ |
| GAN | $p(z)$ | None. The generator only has to *output* values in the range. |
| EBM | — | The energy's domain; Langevin dynamics must respect it |
| Diffusion | — | Noising process and its terminal (reference) distribution |
| Flow | — | Base or source distribution: the NF base after its bijection, or the flow-matching source and its interpolation path |

**Observation:** the latent priors of VAEs and GANs are never a problem. A Gaussian on $\mathbb R^k$ is fine there, because a decoder or generator maps it into data space. The question bites in three places:

1. **Output densities:** AR conditionals and the VAE decoder.
2. **Data-space dynamics:** diffusion and Langevin.
3. **The reference distribution of transport:** flows and diffusion.

## 2. Three options

### Option A: $\mathbb R^d$, with a squashing display map

- **Models:** they live entirely in $\mathbb R^d$. Every textbook formula holds as written: Gaussian noise, Gaussian sources and bases, Gaussian or mixture likelihoods.
- **Display:** color is given by a bijection $c:\mathbb R\to(0,1)$ followed by a hue arc, $\text{hue}(x)=H\cdot c(x)$ with $H<360^\circ$ so the ends never meet.
- **Mathematical status:** the display map is a *rendering*. No model ever sees hue, so its nonlinearity cannot affect any model. It only changes how a point *looks*. Distances in hue are distorted toward the extremes, but no model computes in hue.
- **What $[0,1]$ distributions become:** pushing a Gaussian on $\mathbb R$ through $c=\text{logistic}$ gives the **logit-normal** on $(0,1)$. "Pretending with Gaussian" in $[0,1]$, done consistently, *is* Option A.
- **Precedent:** RealNVP preprocessed bounded pixel values with a logit transform before modeling them as unbounded.
- **A neat lever.** Choose $c$ to be the CDF of the reference distribution, and that distribution renders as **uniformly random hue**. A random variable pushed through its own CDF is uniform. With $c=\Phi$ (the standard normal CDF), $\mathcal N(0,1)$ noise looks like uniformly random colors. With $c=\text{logistic}$, the logistic distribution does. "Pure noise" then *looks* like pure noise.
- **Cost:** data has to be authored in $\mathbb R^d$, or authored in $(0,1)^d$ and pulled back through $c^{-1}$. Pulling back stretches mass near the ends into long tails.

### Option B: $[0,1]^d$ natively, with bounded families

The models must respect the boundary. What each family needs:

| Family | What it needs on $[0,1]^d$ |
|---|---|
| AR, VAE decoder | A density family on $[0,1]$, usually a mixture of one (see §3) |
| GAN | A squashing output layer. Trivial. |
| Normalizing flow | Either Gaussian base → bijection → final sigmoid (a logit-normal base, which is Option A in disguise), or a uniform base with interval-preserving bijections (spline flows are defined on an interval) |
| Flow matching | **Uniform source plus linear interpolation.** The box is convex, so the interpolation path never leaves it. The exact process stays inside; a learned or discretized one may stray slightly. |
| Diffusion | Gaussian noising exits the box. Proper alternatives are **reflected diffusion**, whose stationary distribution is uniform (Lou & Ermon, 2023), or a **Jacobi / Wright–Fisher** process, whose stationary distribution is Beta (as in Dirichlet diffusion score models, Avdeyev et al., 2023). Both add boundary conditions to the score. |
| EBM / Langevin | Reflection or projection at the walls, or reparameterization (which leads back to Option A) |

**Cost:** the boundary is a *geometric feature*, the same kind of issue as the torus, only milder. It touches diffusion and Langevin much more than AR or GAN. That risks the same confound that ruled out circular hue: differences caused by boundary handling would look like differences between principles.

### Option C: "Just pretend with Gaussian"

- **How it works:** author data inside $[0,1]^d$. Models treat it as $\mathbb R^d$, with Gaussian noise and MSE / Gaussian likelihoods. Values outside the range are clipped for display.
- **Precedent:** this is what nearly all image generative models do in practice. Pixels are bounded, yet DDPM adds Gaussian noise and clips, and VAEs use MSE decoders.
- **Pros:** display coordinates equal model coordinates, linearly, so the hue axis reads directly as model space.
- **Cons:** the display is not injective outside $[0,1]$. Leaked mass piles up at the end colors and hides *how far* it leaked. That leak is exactly a failure we want to see, so it needs a separate marker.

### Leaning

**Option A.** It keeps every family's mathematics *exactly* textbook, with no per-family boundary engineering. That fits the abstract-nonsense premise best. It also removes the original complaint: nothing Gaussian lives on $[0,1]$. What lives on $[0,1]$ is just the rendering.

Option C is the practical fallback and matches common practice. Option B would be its own topic, "what boundaries do", in the same spirit as the circular case, and therefore out of scope for now.

## 3. Candidate distributions on $[0,1]$ (relevant to Options B and C)

| Distribution | Support | Working principle (why it exists) | Density | Sampling and reparam. | Score $\partial_x\log p$ | Notes |
|---|---|---|---|---|---|---|
| **Gaussian ("pretend")** | $\mathbb R$ | Max entropy for fixed mean and variance; CLT; stable under addition, which is why diffusion uses it | closed form | trivial | linear | Leaks outside $[0,1]$ |
| **Truncated normal** | $[a,b]$ | Max entropy on an interval for fixed mean and variance | closed form (erf normalizer) | inverse CDF; numerically delicate in the tails | **Same as the untruncated Gaussian in the interior.** Truncation is invisible to the score, so score-based samplers don't "know" about the wall. | Instructive for exactly that reason |
| **Logit-normal** | $(0,1)$ | Gaussian pushed through the logistic function | closed form with the Jacobian $1/(x(1-x))$ | trivial, reparameterizable | smooth in the interior | Equivalent to Option A. Turns bimodal (U-shaped) for large σ. |
| **Beta** | $[0,1]$ | Exponential family with sufficient statistics $\log x$, $\log(1-x)$; max entropy for fixed $\mathbb E\log x$ and $\mathbb E\log(1-x)$; conjugate to the Bernoulli; stationary law of the Jacobi diffusion; Beta(1,1) is uniform | closed form | via Gamma variates. Reparameterization needs implicit gradients (Figurnov et al., 2018). | $(\alpha-1)/x-(\beta-1)/(1-x)$, singular at the walls | The most principled bounded family |
| **Kumaraswamy** | $[0,1]$ | A Beta look-alike with a closed-form CDF $1-(1-x^a)^b$ | closed form | inverse CDF, reparameterizable | closed form | A pragmatic Beta substitute, used in stick-breaking VAEs. Weaker theory. |
| **Uniform** | $[0,1]$ | Max entropy on the box: the natural "pure noise" of a bounded world | constant | trivial | 0 | Natural flow-matching source and reflected-diffusion terminal; also a GAN latent |
| **Triangular** | $[a,b]$ | The mean of two independent U(0,1) is triangular on $[0,1]$; otherwise mostly a "three-point estimate" heuristic | piecewise linear | inverse CDF | piecewise, **undefined at the mode** | Poor for gradient-based models. At most a data-making ingredient. |
| **Continuous Bernoulli** | $[0,1]$ | One-parameter family introduced to fix the misuse of a Bernoulli likelihood for continuous pixels in VAEs (Loaiza-Ganem & Cunningham, 2019) | closed form | inverse CDF | closed form | Monotone shape. Only useful as a VAE decoder likelihood. |

**Multimodal conditionals.** Every row above is unimodal or U-shaped. An AR conditional such as "given pixel 1, pixel 2 is either orange or green" needs a **mixture**: a mixture of Gaussians in Option A, or of Betas or logit-normals in Options B and C. The mixture is a choice of conditional family, not part of the AR principle itself (whitepaper Q3).

*References are from general knowledge and should be verified before being cited.*
