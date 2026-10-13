# Circle Representation Priors

**User:** Anonymous  
**Created:** 10/7/2026 13:16:00  
**Updated:** 10/7/2026 20:41:08  
**Exported:** 10/7/2026 20:51:53  
**Link:** [https://chatgpt.com/c/6ac5c77d-d4d0-83ee-90a4-e65dafe53b12](https://chatgpt.com/c/6ac5c77d-d4d0-83ee-90a4-e65dafe53b12)  

## Prompt:
2026. 10. 7. 오후 1:16:11

Something circular: consider a unit circle. The structure of this object corresponds to this (like hue)

1. How do we represent a circle using real number (or float)? In 3DML, we try using directional vector, Rotational matrix or quintics.
2. What is a good parametrized distribution for prior, analogous to Gaussian?

## Response:
2026. 10. 7. 오후 1:16:14 · gpt-5-6

Yes. The mathematical object you are describing is a variable whose domain is the circle $S^1$, rather than $\mathbb R$. Hue is a canonical example, as are planar orientation, phase, and longitude.

genui{"learning_viz":{"type_id":"UNIT_CIRCLE","initial_values":{"angleDeg":45},"locale_override":"en-US"}}

The two questions then become: **how do we embed/parameterize $S^1$?** and **what is the analogue of a Gaussian distribution on $S^1$?**

### 1. Representing a circular variable

The mathematically minimal representation is simply an angle

$$
\theta\in[0,2\pi),\qquad \theta\equiv\theta+2\pi.
$$

It uses one float, but creates an artificial discontinuity:

$$
0.001 \approx 2\pi-0.001
$$

geometrically, while their numerical distance is nearly $2\pi$. This is often bad for regression/ML.

The standard continuous embedding is therefore

$$
\theta\mapsto
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}\in\mathbb R^2,
\qquad \|x\|=1.
$$

So a circle is represented as $S^1\subset\mathbb R^2$. Two floats are redundant but remove the discontinuity. This is exactly analogous to representing a 3D rotation by something larger than its intrinsic 3 DOF-rotation matrix, quaternion, 6D representation, etc.-because a globally well-behaved Euclidean parameterization is preferable to a minimal one.

There is actually a topology reason you cannot have everything simultaneously: **there is no globally continuous, one-to-one representation of $S^1$ by a single unconstrained real coordinate.** A single float requires either periodic identification or a discontinuity.

### 2. What is the "Gaussian of a circle"?

The most direct answer is the **von Mises distribution**:

$$
p(\theta\mid\mu,\kappa)
=
\frac{\exp[\kappa\cos(\theta-\mu)]}
{2\pi I_0(\kappa)}.
$$

Here $\mu$ is the mean direction and $\kappa\ge0$ is concentration, playing roughly the inverse-variance role:

$$
\kappa=0 \Rightarrow \text{uniform circle},
\qquad
\kappa\gg1 \Rightarrow
\theta\approx\mathcal N(\mu,1/\kappa)
$$

locally.

There is a particularly satisfying connection to your representation question. If

$$
x(\theta)=(\cos\theta,\sin\theta),\qquad
m=(\cos\mu,\sin\mu),
$$

then

$$
m^\top x=\cos(\theta-\mu),
$$

and hence

$$
p(x)\propto e^{\kappa m^\top x}.
$$

So von Mises is essentially an exponential distribution over the **dot product of unit directional vectors**. It respects the geometry automatically: $359^\circ$ and $1^\circ$ are close.

Another option is the **wrapped Gaussian**:

$$
X\sim\mathcal N(\mu,\sigma^2),
\qquad
\Theta=X\bmod 2\pi.
$$

Its density is

$$
p(\theta)
=
\sum_{k\in\mathbb Z}
\mathcal N(\theta+2\pi k;\mu,\sigma^2).
$$

This literally answers "take a Gaussian and make the real line circular." But analytically the von Mises distribution is often cleaner.

There is a useful hierarchy here:

$$
\boxed{
\mathbb R
\rightarrow \text{Gaussian}
}
$$

$$
\boxed{
S^1
\rightarrow \text{von Mises}
}
$$

and, more generally,

$$
\boxed{
S^{d-1}
\rightarrow \text{von Mises--Fisher (vMF)}
}
$$

with

$$
p(x\mid\mu,\kappa)\propto
\exp(\kappa\mu^\top x),
\qquad
x,\mu\in S^{d-1}.
$$

That last one is especially relevant to 3DML: for a **3D unit direction**, $x\in S^2$, vMF is essentially the Gaussian-like prior you are looking for.

For rotations themselves, however, the domain becomes $SO(3)$, not a sphere, and the corresponding distributional story gets more interesting: matrix Fisher on $SO(3)$, Bingham distributions on unit quaternions $S^3/\{\pm1\}$, isotropic Gaussian-like distributions via the Lie algebra, etc. This is the natural continuation of the circle example.

## Prompt:
2026. 10. 7. 오후 1:19:50

Something to do with harmonic decomposition?

## Response:
2026. 10. 7. 오후 1:19:52 · gpt-5-6

Yes. If your intuition was "there should be a representation intrinsic to circular structure involving harmonics," then the relevant object is **Fourier/harmonic representation on $S^1$**.

A scalar function on the circle is periodic,

$$
f(\theta)=f(\theta+2\pi),
$$

so its natural basis is

$$
1,\quad \cos\theta,\sin\theta,\quad
\cos2\theta,\sin2\theta,\quad\ldots
$$

or equivalently

$$
f(\theta)=\sum_{k\in\mathbb Z}c_k e^{ik\theta}.
$$

This gives a useful reinterpretation of the representation I mentioned:

$$
\theta\mapsto(\cos\theta,\sin\theta).
$$

That's precisely the **first harmonic $k=1$** of the Fourier representation. You could represent a circular variable more generally by

$$
\phi_K(\theta)
=
(\cos\theta,\sin\theta,
\cos2\theta,\sin2\theta,\ldots,
\cos K\theta,\sin K\theta).
$$

This is often called a **Fourier feature / harmonic embedding**.

And the distributional side connects beautifully. Any density on the circle can itself be harmonically decomposed:

$$
p(\theta)
=
\frac{1}{2\pi}
\left[
1+
\sum_{k=1}^{\infty}
a_k\cos(k\theta)+b_k\sin(k\theta)
\right].
$$

So there are actually two related answers to your original question:

- If you want a **simple parametric "Gaussian-like" distribution** on $S^1$: von Mises.
- If you want a **general representation of arbitrary distributions/functions on $S^1$**: Fourier/harmonic decomposition.

This generalizes in exactly the direction you may be remembering from 3DML:

$$
S^1 \rightarrow \text{Fourier harmonics},
$$

$$
S^2 \rightarrow \text{spherical harmonics},
$$

$$
SO(3) \rightarrow \text{Wigner-}D\text{ matrices / irreducible representations}.
$$

So spherical harmonics and the representation theory of rotations are essentially the higher-dimensional version of the very simple fact that a circle naturally gives you $e^{ik\theta}$.

For something like **hue**, Fourier/harmonic encoding is therefore especially natural: hue isn't merely "a number that wraps around"; mathematically it is a coordinate on $S^1$, and its harmonics are the natural basis functions on that space.

## Prompt:
2026. 10. 7. 오후 1:34:11

I am thinking of a toy experiment to visualize confusing concepts in generative models.

1. We have a dataset called “Tree”. Each are images with … one/two/three pixels, stacked as a vector. (one/two/three since we can easily visualize up to three dimensions.)
2. Actually, we only consider hues. So there is only one channel, and we really have one or two or three dimensions for “x”.
3. Tree dataset is prepared with some interesting distribution. (green, orange, orange) is a tree. (Red, orange, orange) or (Yellow, green, orange) is also a tree. But (Green, Yellow, orange) is not a tree. And so on. (The exemplar color descriptions does not mean there is no variance)
4. Our job is to generate “tree”. There could be many methods with different principles. (autoregressive, vae, ode / flow, gan, energy, diffusion) With possibly differing results.

Some expectations with this toy demo:
- This will not require extensive computing; CPU may work, so it is accessible. It is just three variables to generate… The evaluation is also easy. Cosine similarity or other distance metrics will be very easy to compute.
- Visualization could be interesting and help intuition. Which maps to where, how sampling works, how training works in real time, and so on. However, because hue is circular, R^3 representation of the locus could be confusing. (dim=1 could use a literal circle, and dim=2 could use sphere.) But we can think of creative methods to solve this part so we do not think about this for now.

What do you think of this? Is this problem too small to demonstrate generative models?

## Response:
2026. 10. 7. 오후 1:34:13 · gpt-5-6

No. I think it is almost the opposite: it is small enough that you can expose things that image-scale experiments hide.

The important qualification is that this would not really be an "image-generation toy problem." It is a problem of learning a deliberately complicated distribution on a very low-dimensional structured space. That makes it potentially quite good for explaining generative models.

Your three-pixel case is mathematically something like

$$
x=(h_1,h_2,h_3)\in(S^1)^3=T^3,
$$

where each $h_i$ is hue. So the actual data manifold is a 3-torus. "Tree" is some multimodal probability distribution $p_{\text{tree}}(x)$ on that space.

That formulation already gives the experiment considerably more substance than "three variables."

The key is to make the Tree distribution sufficiently interesting. Suppose it contains several modes such as

$$
(G,O,O),\quad(R,O,O),\quad(Y,G,O),
$$

with continuous variation around each, but deliberately excludes nearby combinations such as

$$
(G,Y,O).
$$

Then the learner has to capture several distinct things simultaneously: marginal hue distributions, dependencies among pixels, multimodality, forbidden regions, and relative mode probabilities. You can additionally make some modes broad, some narrow, some common, and some rare.

That is enough structure for generative models to behave differently.

For example, an autoregressive model exposes factorization directly:

$$
p(h_1,h_2,h_3)
=
p(h_1)p(h_2\mid h_1)p(h_3\mid h_1,h_2).
$$

You could literally animate "choose pixel 1 → this changes the distribution of pixel 2 → those two determine the possible distributions of pixel 3." The semantics of autoregression become visible rather than abstract.

A VAE gives a quite different picture. You could show the Tree distribution being mapped into a 1D or 2D latent space, sample

$$
z\sim\mathcal N(0,I),
$$

and display $D(z)$ moving through Tree-space. With an intentionally insufficient latent representation, interpolation through forbidden Tree configurations could become visually obvious. Posterior/prior mismatch also becomes something you can inspect directly.

GAN is perhaps even more interesting. Put real and generated Trees into the same visualization and show the discriminator field over the space. Mode collapse becomes almost comically literal: perhaps the generator discovers that producing variations around $(G,O,O)$ reliably fools $D$, while two other Tree modes disappear.

Energy-based modeling would give you an actual scalar landscape:

$$
E(x)
\quad\Longleftrightarrow\quad
p(x)=\frac{e^{-E(x)}}{Z}.
$$

Then "what is an energy model?" becomes: valleys are Trees, hills are non-Trees; sampling means wandering through this landscape. Langevin dynamics could be animated directly. This connects particularly well with the statistical-mechanics questions you were asking earlier today.

Flow matching / ODE gives perhaps the strongest visualization:

$$
x_0\sim p_0
\quad\longrightarrow\quad
x_t
\quad\longrightarrow\quad
x_1\sim p_{\text{tree}}.
$$

You can display hundreds of particles and the velocity field simultaneously. Questions such as crossing trajectories, transport coupling, curvature, NFE, straightness, and what a learned velocity field actually means become geometrical.

Diffusion gives the reverse movie: progressively destroy Tree structure until you reach the prior, then watch the learned reverse process reconstruct the modes. Score vectors can actually be drawn:

$$
s_\theta(x,t)\approx\nabla_x\log p_t(x).
$$

In ordinary image space, saying "the network estimates the score of a 786,432-dimensional distribution" is nearly impossible to visualize. Here you can point at an arrow.

So I would not worry that three variables are "too few." There is an important distinction:

$$
\boxed{\text{low dimensionality}\neq\text{simple probability distribution}.}
$$

A distribution in two dimensions can already have arbitrarily complicated topology and density. Three dimensions are more than enough.

There is one issue I would change in your design, though. I would probably make **2 pixels the primary pedagogical case and 3 pixels an extension**, rather than starting with three.

With two hues,

$$
(h_1,h_2)\in S^1\times S^1=T^2.
$$

That is a torus, but you can initially unwrap it into a square:

$$
(\theta_1,\theta_2)\in[0,2\pi)^2
$$

with opposite edges identified. Then you can directly plot the true density, generated density, energy, discriminator, score, velocity field, trajectories, training samples, etc. Every generative model gets the *same underlying 2D canvas*.

For example, imagine defining Tree approximately as several continuous blobs/ridges:

```text
h₂
↑
│     ●●●                    mode C
│    ●●●●
│
│                         ●●
│                       ●●●●       mode B
│
│       forbidden region
│            × × ×
│
│  ●●●
│ ●●●●                             mode A
└──────────────────────────────→ h₁
  0                              2π
```

Remember that left/right and top/bottom wrap around.

Now almost every model can be rendered as an operation on exactly this picture:

$$
\begin{array}{ll}
\text{MLE} & \text{fit density contours}\\
\text{AR} & p(h_1)\rightarrow p(h_2|h_1)\\
\text{VAE} & x\rightarrow z\rightarrow x\\
\text{GAN} & z\rightarrow x,\quad D(x)\text{ as background field}\\
\text{EBM} & E(x)\text{ as topographic surface}\\
\text{Langevin} & x_{k+1}=x_k-\eta\nabla E(x_k)+\sqrt{2\eta}\epsilon\\
\text{Diffusion} & p_{\rm data}\rightarrow p_t\rightarrow p_{\rm noise}\\
\text{Score} & \nabla_x\log p_t(x)\text{ as arrows}\\
\text{Flow} & v_\theta(x,t)\text{ as arrows + particle trajectories}.
\end{array}
$$

That common coordinate system is, in my view, the strongest part of the idea. You aren't merely making six tiny generative models. You are making six different answers to the same question visible on the same probability distribution.

Then the 3-pixel version becomes useful specifically when you want to demonstrate phenomena that require another dependency:

$$
p(h_3\mid h_1,h_2),
$$

conditional ambiguity, higher-order correlations, etc.

And there is another useful trick: you don't actually need hue to be your *computational* representation. Internally you can encode each hue as

$$
h_i\mapsto(\cos h_i,\sin h_i),
$$

so a two-pixel Tree is represented in $\mathbb R^4$, while the UI still displays it as two hue variables. This prevents the artificial $0/2\pi$ discontinuity from contaminating your experiment. The generative-model issue and the visualization issue can therefore be separated.

One caution: I would **not use cosine similarity as the primary evaluation metric**. Generation is about matching distributions, not reconstructing individual samples. With a known synthetic $p_{\text{tree}}$, you have an unusual luxury: you can calculate quantities that are normally inaccessible. Depending on how you define the ground-truth density, you could measure KL/JS divergence, Wasserstein distance, mode coverage, probability assigned to forbidden regions, marginal/conditional errors, or simply compare the learned density against the exact target density.

That could make this substantially more educational than MNIST-style demonstrations. In real generative modeling, we usually don't know $p_{\rm data}(x)$. In **Tree**, you created it, so you do.

The resulting project has a rather clean conceptual premise:

> **Make $p_{\text{data}}$ simple enough to see, but complicated enough that learning it is nontrivial.**

That is exactly the regime in which differences among GANs, VAEs, EBMs, diffusion, autoregression, and flows are easiest to understand.

## Prompt:
2026. 10. 7. 오후 1:47:40

A good thing about this educational demo is that you can talk about generative models as is. No additional engineering tricks or powerful backbone architectures. We just focus on abstract nonsense and higher level observations of each miniaturized model family.

## Response:
2026. 10. 7. 오후 1:47:41 · gpt-5-6

Exactly. I think that is the strongest justification for the experiment.

In normal image generation, the conceptual object

$$
p_\theta(x)
$$

is almost immediately buried underneath engineering choices: U-Net vs DiT, attention, normalization, embeddings, augmentation, pretrained encoders, CFG, resolution, optimizer details, and GPU constraints. A student can successfully train a diffusion model without acquiring much intuition for what the diffusion model is actually doing as a distribution-learning procedure.

Your Tree world removes most of that.

You could deliberately impose a rule:

> Every model gets only a tiny MLP unless its mathematical definition intrinsically requires something else.

Then architecture stops being an explanatory variable. If GAN collapses, you inspect the adversarial game. If a VAE fills an undesirable region between modes, you inspect the latent-variable objective. If a flow bends strangely, you inspect the transport objective. If Langevin sampling mixes poorly, you inspect the energy landscape and sampling dynamics.

This also lets you discuss fairly abstract things without apologizing for their impracticality. For example:

$$
\text{Data distribution}
\rightarrow
\begin{cases}
\text{explicit density} & p_\theta(x)\\
\text{latent variable} & p(z)p_\theta(x|z)\\
\text{energy} & e^{-E_\theta(x)}/Z_\theta\\
\text{transport map} & x=T_\theta(z)\\
\text{vector field} & \dot x=v_\theta(x,t)\\
\text{score field} & \nabla_x\log p_t(x)\\
\text{adversarial game} & G_\theta\leftrightarrow D_\phi
\end{cases}
$$

These normally look like unrelated equations. In Tree, they are different mechanisms trying to reproduce **the same colored blobs on the same little space**.

That creates some unusually good comparisons. For example, you could stop every model halfway through training and ask not merely "which has lower loss?" but:

- What does this model currently believe a Tree is?
- Where is that belief actually stored—density, energy, discriminator, latent decoder, score, velocity?
- How does it produce one new Tree from that belief?
- What does a failure look like geometrically?
- What information does its training objective actually provide?

The last question is particularly valuable. GAN never receives "move this generated point toward that data point." Flow matching essentially does receive local transport supervision under a chosen coupling. An EBM learns relative preference but then requires a sampling procedure. A VAE introduces an auxiliary latent inference problem. Autoregression turns the joint problem into conditional ones. Diffusion deliberately constructs intermediate distributions to make the learning problem tractable.

Those distinctions become much sharper when all the neural networks are boring.

I would even embrace the "abstract nonsense" character rather than make Tree resemble a serious benchmark. The pixels and hues provide just enough semantics that `(green, orange, orange)` feels like an object rather than $(1.7,4.2,3.9)$, while the underlying world remains mathematically inspectable.

There is also an interesting endpoint: because you define the true Tree distribution, you can show **ground truth objects that real generative-model researchers normally cannot see**—the actual $p_{\text{data}}$, actual score $\nabla\log p$, perhaps actual energy $-\log p+C$, exact marginals and conditionals, and samples in arbitrary quantity. Then beside each one you show what the mini-model learned.

So it could be less a "toy generative-model benchmark" and more a **generative-model microscope**. The simplification is the feature: remove computer vision, large-scale optimization, and architecture design until almost nothing remains except the generative principle itself.

## Prompt:
2026. 10. 7. 오후 2:11:33

My concern is topology. Not that it is hard to wrap (?) one’s head around of it, but because the mathematics could work different. Our normal gradients think of R^d without curvature, but we now have this weird manifold thing. This is implementation-level low level part and not related to the user experience. (it is not intended)

Do not derive the foundations in full detail and answer briefly based on your general knowledge of mathematics.

## Response:
2026. 10. 7. 오후 2:11:35 · gpt-5-6

Yes. This is a legitimate concern. By choosing hue, you accidentally changed the sample space from ordinary Euclidean space to a torus:

$$
x\in (S^1)^d=T^d,
$$

whereas most textbook generative-model formulations implicitly assume $x\in\mathbb R^d$.

That matters differently across model families. Autoregressive models, VAEs, GANs, and EBMs can be adapted fairly naturally by using circular representations/distributions. But gradient-based dynamics—score models, diffusion, flow matching, Langevin dynamics—are where the distinction becomes conceptually significant. Ordinary Gaussian noise, Euclidean interpolation, $\nabla_x$, and ODE/SDE dynamics are all defined assuming Euclidean geometry. On $T^d$, their mathematically proper counterparts involve periodic/Riemannian geometry.

You can hide this implementation detail by embedding each hue as

$$
\theta\mapsto(\cos\theta,\sin\theta),
$$

but then the model operates in $\mathbb R^{2d}$ while the data occupy a constrained manifold. Naively adding Gaussian noise pushes samples off that manifold. So that doesn't completely eliminate the issue either.

A simpler alternative for the educational project may actually be: **don't make hue intrinsically circular.** Choose a bounded scalar such as grayscale intensity, or use a restricted hue interval that never crosses the $0/2\pi$ seam. Then Tree lives approximately in an ordinary box

$$
x\in[0,1]^d\subset\mathbb R^d,
$$

and all the standard generative-model mathematics works essentially as presented in introductory treatments.

You can still color the scalar values for visualization. For instance, define a non-circular color ramp from yellow → green → blue and call those values "pixel types." The UI remains colorful without making the mathematical sample space circular.

I would favor that. **Circular hue introduces a genuine manifold-learning problem that your demo isn't trying to teach.** It risks making differences caused by $S^1/T^d$ geometry look like differences between generative-model families. Save the circular version as an optional later demonstration of "what happens when the data space itself has topology."



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)