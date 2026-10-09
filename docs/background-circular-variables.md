# Background: Circular Variables, Priors, and Topology

**Status:** Static record. It summarizes the opening of the 2026-10-07 brainstorm, which led to Tree, and the topology concern that settled [decision D2](decisions.md#d2-euclidean-sample-space-circular-hue-deferred).
After Review 1, circular and toroidal sample spaces are **out of scope**. This document is kept only as a record.
**Parent:** [whitepaper.md](../whitepaper.md)

---

## 1. Representing a circular variable

Hue, planar orientation, phase, and longitude all live on the circle $S^1$, not on $\mathbb R$.

| Representation | Floats | Pros | Cons |
|---|---|---|---|
| Angle $\theta\in[0,2\pi)$ with $\theta\equiv\theta+2\pi$ | 1 | Minimal | Artificial seam: $0.001$ and $2\pi-0.001$ are neighbors on the circle but nearly $2\pi$ apart as numbers; bad for regression and ML |
| Unit vector $(\cos\theta,\sin\theta)\in S^1\subset\mathbb R^2$ | 2 | Continuous, no seam | Redundant and constrained ($\lVert x\rVert=1$) |
| Harmonic embedding $(\cos k\theta,\sin k\theta)_{k=1..K}$ | $2K$ | The natural basis of functions on $S^1$ | Redundant |

**Topological fact:** no continuous, one-to-one map takes $S^1$ to a single unconstrained real coordinate. A single float always needs either periodic identification or a discontinuity.

3D machine learning makes the same trade-off, picking an over-complete but well-behaved representation over a minimal one: rotation matrices, quaternions, and the 6D representation for $SO(3)$, rather than three angles.

## 2. Gaussian-like priors on the circle and beyond

**Von Mises distribution**, the "Gaussian of the circle":

$$
p(\theta\mid\mu,\kappa)=\frac{\exp[\kappa\cos(\theta-\mu)]}{2\pi I_0(\kappa)}.
$$

The concentration $\kappa$ plays the role of an inverse variance:

- $\kappa=0$ gives the uniform distribution on the circle.
- $\kappa\gg1$ gives $\theta\approx\mathcal N(\mu,1/\kappa)$ locally.

With $x=(\cos\theta,\sin\theta)$ and $m=(\cos\mu,\sin\mu)$, we have $m^\top x=\cos(\theta-\mu)$, so

$$
p(x)\propto e^{\kappa\,m^\top x}.
$$

Von Mises is an exponential family over the dot product of unit vectors, and it automatically treats 359° and 1° as close.

**Wrapped Gaussian:** take $X\sim\mathcal N(\mu,\sigma^2)$ and set $\Theta=X\bmod 2\pi$. Its density is

$$
p(\theta)=\sum_{k\in\mathbb Z}\mathcal N(\theta+2\pi k;\mu,\sigma^2).
$$

It is literally "a Gaussian with the real line wrapped around". Analytically it is usually less convenient than von Mises.

**The hierarchy:**

| Space | Gaussian-like distribution |
|---|---|
| $\mathbb R$ | Gaussian |
| $S^1$ | von Mises |
| $S^{d-1}$ | von Mises–Fisher, $p(x)\propto\exp(\kappa\mu^\top x)$; for 3D unit directions ($S^2$) this is the Gaussian-like prior |
| $SO(3)$ | matrix Fisher; Bingham on unit quaternions $S^3/\{\pm1\}$; isotropic Gaussian-like distributions via the Lie algebra |

## 3. Harmonic decomposition

A function on the circle is periodic, so its natural basis is Fourier:

$$
f(\theta)=\sum_{k\in\mathbb Z}c_k e^{ik\theta}.
$$

The embedding $(\cos\theta,\sin\theta)$ is exactly the **first harmonic** of this basis. Any density on $S^1$ also decomposes harmonically:

$$
p(\theta)=\frac{1}{2\pi}\Big[1+\sum_{k\ge1}a_k\cos k\theta+b_k\sin k\theta\Big].
$$

This gives two complementary answers:

- For a **simple parametric prior**, use von Mises.
- For **arbitrary densities and functions** on $S^1$, use Fourier / harmonic decomposition.

The same pattern continues to higher dimensions:

| Space | Harmonic basis |
|---|---|
| $S^1$ | Fourier harmonics |
| $S^2$ | spherical harmonics |
| $SO(3)$ | Wigner-D matrices (irreducible representations) |

## 4. The topology concern (why Tree is not circular)

If pixels were hues, a $d$-pixel Tree would live on the torus $T^d=(S^1)^d$. Most generative-model formulations assume $\mathbb R^d$. How much that matters depends on the family:

| Family | Adapting to $T^d$ |
|---|---|
| AR, VAE, GAN, EBM | Fairly natural: use circular likelihoods or priors (for example von Mises) or circular embeddings |
| Score models, diffusion, flow matching, Langevin | **Changes at a basic level.** Gaussian noise, linear interpolation, $\nabla_x$, and ODE/SDE dynamics all assume Euclidean geometry, and the proper counterparts are periodic or Riemannian |

Embedding each hue as $(\cos\theta,\sin\theta)$ does **not** fully fix the problem. The model then works in $\mathbb R^{2d}$ while the data sit on a constrained manifold, and naive Gaussian noise pushes samples off it.

**Conclusion (D2):** circular hue would add a manifold-learning problem the demo isn't trying to teach. It would also risk making effects of $S^1$/$T^d$ geometry look like differences between model families. Tree therefore uses a bounded Euclidean scalar per pixel and keeps the colorful display through a non-circular ramp.

## 5. Notes in case this is ever revisited (out of scope)

*"What happens when the data space itself has topology."*

- **Data:** a circular Tree-2 on $T^2$, built as a mixture of products of von Mises distributions. It is drawn on the unwrapped square $[0,2\pi)^2$ with opposite edges identified, and optionally on a 3D torus.
- **Naive vs. correct, side by side:**
  - Naive: Euclidean diffusion and flow matching on raw angles, which shows seam artifacts.
  - Naive: the $(\cos,\sin)$ embedding with ambient Gaussian noise, which shows samples leaving the manifold.
  - Correct: diffusion with wrapped-Gaussian noise on the torus, and Riemannian flow matching along geodesic interpolation $\theta_t=\theta_0+t\cdot\mathrm{wrap}(\theta_1-\theta_0)$.
- **Literature to revisit before building** (from general knowledge; verify the details):
  - Riemannian score-based generative modeling (De Bortoli et al., 2022)
  - Riemannian flow matching on general geometries (Chen & Lipman)
  - Torsional diffusion, which applies diffusion on a hypertorus to molecular conformers (Jing et al., 2022)
