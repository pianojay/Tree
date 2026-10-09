# Brainstorm Log: 2026-10-07

**Status:** Static record. It summarizes the conversation where the Tree project began; it does not reproduce it word for word.
**Source:** ChatGPT session "Circle Representation Priors", 13:16–14:11 local time. Raw export: [../ChatGPT-Circle Representation Priors-20261007-2051.md](../ChatGPT-Circle%20Representation%20Priors-20261007-2051.md)
**Parent:** [whitepaper.md](../whitepaper.md)

---

## Timeline

| Time | Turn | Gist | Outcome |
|---|---|---|---|
| 13:16 | **Q: How do we represent a circle (like hue) with floats, and what is the Gaussian-like prior on it?** | An angle has a seam. The $(\cos,\sin)$ embedding is continuous but redundant, like 3D rotation representations. Von Mises is the circle's "Gaussian" ($p(x)\propto e^{\kappa m^\top x}$). Also covered: wrapped Gaussian; vMF on $S^{d-1}$; matrix Fisher and Bingham for $SO(3)$. | Background → [background-circular-variables.md](background-circular-variables.md) §1–2 |
| 13:19 | **Q: Something to do with harmonic decomposition?** | $(\cos\theta,\sin\theta)$ is the first Fourier harmonic. Harmonic embeddings and Fourier densities on $S^1$. The pattern continues as spherical harmonics on $S^2$ and Wigner-D on $SO(3)$. | Background → §3 |
| 13:34 | **Idea: the Tree toy experiment.** Datasets of 1, 2, or 3 hue "pixels". Example Trees: (G,O,O), (R,O,O), (Y,G,O); (G,Y,O) is not one. Generate Trees with AR, VAE, ODE/flow, GAN, energy, and diffusion models. Expected to run on a CPU, be easy to evaluate (cosine similarity?), and be easy to visualize, with circular topology parked for later. Asked: is the problem too small? | **No.** Low dimensionality is not the same as a simple distribution. The data space is $T^3$. Each family has a natural visualization: AR factorization, VAE latent bridges, GAN discriminator field and mode collapse, EBM landscape with Langevin, flow particles and velocity, diffusion score arrows. **Proposed: 2 pixels as the primary case**, on one common canvas. Suggested a $(\cos,\sin)$ internal encoding. **Cautioned against cosine similarity**; use distribution metrics instead, since $p_{\text{tree}}$ is known. | D1, D3, D5, D6, D-S1 |
| 13:47 | **Point: the demo shows generative models "as is"**, with no engineering tricks or powerful backbones, focusing on abstract nonsense and high-level observations of miniature model families. | Agreed; this is the strongest justification. **The tiny-MLP rule.** A taxonomy of the families (explicit density, latent variable, energy, transport, vector field, score, adversarial game). **The five interrogation questions.** What each objective's signal does and doesn't provide. Ground truth that researchers normally can't see. Coined **"generative-model microscope"**. | D4; whitepaper §1, §3, §4 |
| 14:11 | **Concern: topology.** On a manifold, the usual Euclidean gradients and math may not hold (an implementation-level concern, not a UX one). A brief answer was requested. | Legitimate. AR, VAE, GAN, and EBM adapt easily. Score, diffusion, flow matching, and Langevin need periodic or Riemannian counterparts. The $(\cos,\sin)$ embedding still leaves an off-manifold noise problem. **Recommendation: don't make hue circular.** Use a bounded scalar with a non-circular color ramp, and save the circular version as an optional topology demo. | D2 (supersedes D-S1); background §4–5 |

## Key quotes (paraphrased, except the premise)

- Premise: *"Make $p_{\text{data}}$ simple enough to see, but complicated enough that learning it is nontrivial."*
- The common coordinate system is the strongest part of the idea: six answers to one question, on one distribution.
- Neural networks that are deliberately boring make the differences between principles sharp.
- Less a toy benchmark than a microscope. The simplification is the feature.

## Threads left open at the end of the session

The assistant closed by offering three follow-ups, none of them taken up yet:

1. Implementing Tree with a Euclidean representation, which is now the plan of record (D2).
2. Ways to adapt gradient-based generative models to circular or harmonic topology. **Out of scope** (D2, Review 1).
3. Extending Tree later to visualize the effects of topology explicitly. **Out of scope** (D2, Review 1).

Later developments are recorded in [decisions.md](decisions.md). That includes the owner's Review 1 of whitepaper v0.1, and the draft proposals withdrawn there.
