# Model Families: Making the Concepts Alive

**Status:** Working notes, Review 3.
**Parent:** [whitepaper.md](../whitepaper.md)

**Provenance:** the quoted blocks are the owner's notes. Everything under "Living objects" is imagined by Claude, as candidate ideas rather than specifications.

---

## 0. Idea

Each family is reduced to its **crux map**: the map it actually learns. The backbone doesn't matter. Small networks on a CPU are enough.

When the input space has only a few dimensions, every crux map can be:

- **drawn** as a curve, a warped grid, a field, or a moving particle
- **probed** by hovering, dragging, or clicking
- **perturbed** with a knob, while it trains

Concepts that normally stay on paper (reparameterization, Jacobians, minimax games, Boltzmann–Gibbs, Langevin kernels, probability paths) become objects you can see and touch.

Notation: a Tree is $x=(x_1,x_2)$. A latent is $z$, in 1D or 2D, for families that have one.

## Seven crux maps

| Family | Crux map | Kind of map |
|---|---|---|
| Autoregressive | $x_{<t}\mapsto p_\theta(x_t\mid x_{<t})$ | sample → distribution |
| VAE | $x\mapsto q_\phi(z\mid x)$ and $z\mapsto p_\theta(x\mid z)$ | sample → distribution, twice, in opposite directions |
| Normalizing flow | diffeomorphism $T_\theta:z\mapsto x$, with $p_X=(T_\theta)_\#\,p_Z$ | distribution → distribution, with exact density bookkeeping |
| GAN | $G_\theta:z\mapsto x$ and critic $D_\phi:x\mapsto[0,1]$ | distribution → distribution, without density; a critic field |
| Energy-based | $E_\theta:x\mapsto\mathbb R$ with $p_\theta\propto e^{-E_\theta}$, plus a kernel $K:x\mapsto x'$ | distribution ↔ scalar field; Markov kernel |
| Diffusion | $(x,t)\mapsto\nabla_x\log p_t(x)$, equivalently the denoiser $x_t\mapsto\mathbb E[x_0\mid x_t]$ | a time-indexed family of fields |
| Flow matching | $(x,t)\mapsto v_\theta(x,t)$, whose flow $\varphi_t$ pushes $p_0$ to $p_t$ | distribution → distribution, through time |

Three kinds of map keep recurring:

- **sample → distribution**
- **distribution → distribution** (pushforward)
- **distribution ↔ field** (scalar or vector)

The families differ in which kinds they use and in what signal trains them.

---

## 1. Autoregressive

> MADE comes to mind; RNADE may be more suitable to demonstrate sequential generation; we could even use transformers. Not the main concern. The point is that it models the map $x_{<t}\mapsto p_\theta(x_t\mid x_{<t})$, and that is enough to visualize. That is the crux.

Living objects:

- **The conditional on a string.** Drag $x_1$ along its axis. A 1D density curve $p(x_2\mid x_1)$ rides along the perpendicular line and reshapes as it moves. The map from a point to a curve becomes a literal picture.
- **The joint as a stack of slices.** Sweep $x_1$ across and leave the curves behind, each scaled by $p(x_1)$. The joint density reassembles itself from slices. The chain rule becomes a picture.
- **Two-marble sampling.** A marble drops into $p(x_1)$ and lands. Its landing point lights up a line, and a second marble drops along that line into $p(x_2\mid x_1)$. One pixel at a time.
- **Likelihood you can hover.** Any point shows $\log p(x)$ as a stacked bar: $\log p(x_1)+\log p(x_2\mid x_1)$.
- **The other order.** A second model generates $x_2$ first. Both claim to be the same joint. Wherever they disagree, you are seeing approximation error, not the principle.
- **Inside a conditional** (RNADE-style): each conditional is a small mixture. Its components appear as sub-bumps that slide and trade weight as $x_1$ moves.
- **Autoregression as wiring** (MADE-style): the mask is a tiny connectivity diagram. The output for $x_2$ can see $x_1$; the output for $x_1$ sees nothing.

## 2. VAE

> VAE contains two maps. First the encoder, $x\mapsto q_\phi(z\mid x)$. You sample $z$ from $p(z)$, then decode, $z\mapsto p_\theta(x\mid z)$. Other fun and intuitive things: how the reconstruction term and the prior-KL term clash, the reparameterization trick, β-VAE, data KL vs. average posterior KL, posterior collapse. These concepts will not remain concepts; they become actual, concrete, visualizable, interactable objects.

Two linked canvases: the data plane and the latent plane.

- **The encoder as a lens.** Hover a data point, and its posterior ellipse $q(z\mid x)$ appears in the latent plane. Brush across the data, and you leave a trail of ellipses.
- **The decoder as a projector.** Hover a latent point, and a blob $p(x\mid z)$ appears in the data plane. Drag through latent space, and the blob travels through data space, including through forbidden regions when the latent is too small.
- **Aggregate posterior vs. prior.** All the ellipses together form $q(z)=\mathbb E_x\,q(z\mid x)$. Lay the prior $p(z)$ over it. Anywhere the prior has mass and $q(z)$ has none is a *hole*. Click into a hole, decode, and see what a prior sample there turns into.
- **The reparameterization trick, drawn.** A particle $\varepsilon$ in the unit disk is shifted and stretched into the ellipse: $z=\mu+\sigma\odot\varepsilon$. Gradient arrows flow back to $\mu$ and $\sigma$. The $\varepsilon$ particle is a fixed source that the gradient never has to pass through.
- **Tug of war.** Each ellipse feels two forces:
  - *reconstruction* pulls it to shrink and move apart from the others, so that $z$ identifies $x$
  - *prior KL* pulls it to grow and drift toward the unit circle

  A β slider changes the balance live. As β → 0, the ellipses shrink to dots: a plain autoencoder, with prior holes everywhere. As β grows, they merge into the prior.
- **Posterior collapse as an event.** Push β up, or give the decoder enough freedom, and watch the ellipses converge onto the prior circle. The decoder blob stops moving when you drag $z$, and every $z$ decodes to the same average blob. A per-point KL gauge falls to zero.
- **The KL bookkeeping as gauges.** There are two standard decompositions. Both can be drawn here, and either may be what "data KL vs. average posterior KL" points at:
  - **Model error vs. encoder error.** $\mathbb E_{p_{\text{data}}}[-\text{ELBO}]=H(p_{\text{data}})+\mathrm{KL}(p_{\text{data}}\,\|\,p_\theta)+\mathbb E_x\,\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)$. With a 2D latent, the *true* posterior $p_\theta(z\mid x)$ can be computed on a grid. Draw it behind the Gaussian ellipse; the mismatch between their shapes *is* the last term.
  - **Information vs. mismatch** ("ELBO surgery"). $\mathbb E_x\,\mathrm{KL}(q(z\mid x)\,\|\,p(z))=I_q(x;z)+\mathrm{KL}(q(z)\,\|\,p(z))$. The first term is how separated the ellipses are. The second is how badly the cloud of ellipses misses the prior.

## 3. Normalizing flow

> Affine flows, element-wise flows, NICE, RealNVP with coupling layers, affine autoregressive flows, MAF, permutations, and so on. None of these details are our immediate concern. The important fact is that we introduce a diffeomorphism and its Jacobian to stretch and shrink the probability distribution.

- **A rubber sheet with ink.** A grid on the base plane carries Gaussian ink. $T$ warps the sheet: the ink thins where the sheet is stretched and thickens where it is squeezed. The density of $p_X$ *is* the ink.
- **The Jacobian as a heatmap.** $\log\lvert\det J_T\rvert$ colored by expansion and contraction. The change-of-variables formula $p_X(x)=p_Z(z)/\lvert\det J_T(z)\rvert$ becomes "ink per unit area".
- **A square under the microscope.** Hover a point, and a tiny square at $z$ appears next to its image, a parallelogram at $x$. Their area ratio is $\lvert\det J_T\rvert$.
- **Layer by layer.** Scrub through the layers. A coupling layer keeps one coordinate fixed and slides or scales the other as a function of it: the sheet shears along one axis at a time. A permutation swaps the axes.
- **Two-way traffic.** Drag a data point, and its pre-image traces back through the layers to the base. Invertibility is a path you can walk in both directions.
- **The sheet can't tear.** A diffeomorphism keeps a connected base connected. To separate two modes, the sheet has to stretch thin between them, which leaves a faint ink bridge. The limit of the principle shows up as geometry.

## 4. GAN

> Now we hide the loss function behind the discriminator. We don't think about diffeomorphisms; it is just a direct mapping. We can watch the fight between discriminator and generator in real time, converging as a minimax game. Meanwhile, we observe mode collapse in real time.

- **The sheet can fold now.** $G$ warps the latent grid with no invertibility and no Jacobian bookkeeping. Creases appear where many $z$ land on the same $x$. The contrast with the normalizing flow is the point.
- **The referee's field.** $D(x)$ is drawn as a background heatmap. Generated points slide uphill along it, toward "real". The generator never sees data, only this field at its own points, so the arrows are drawn at the fake points only.
- **The fight.** $D$'s field reshapes, the generated cloud moves, and $D$ reshapes again. Mode hopping shows up as the cloud jumping between modes with $D$'s field chasing behind.
- **Mode collapse as a color.** Color the latent plane by which mode each $z$ lands in. A healthy generator gives a patchwork. Collapse turns the latent plane one color.
- **Freeze a player.**
  - Freeze $D$ and train $G$ alone: the whole cloud rushes toward $D$'s maximum, ideally a single point.
  - Freeze $G$ and train $D$ alone: $D$ becomes confidently perfect. Under the original minimax loss, the generator's arrows fade away (vanishing gradients). Under the non-saturating loss, they survive. A choice of loss you can see.
- **The ideal referee.** In 2D, the optimal discriminator $D^*=p_{\text{data}}/(p_{\text{data}}+p_G)$ can be estimated directly. Show it next to the actual $D$.

## 5. Energy-based

> Now we introduce an interesting map between a probability distribution $p$ and a scalar energy field $E$ (the Boltzmann–Gibbs equation). Training is equivalent to decreasing the energy of $x\sim p_{\text{data}}$ while increasing the energy of $x\sim p_\theta$. This nudges the scalar field closer to $p_{\text{data}}$, starting from an initial shape that comes from a normal distribution. Sampling uses Langevin dynamics as the transition kernel. Removing the noise term gives no varying result, and setting η to zero preserves the transition.

- **Landscape and lake.** $E$ is terrain, and $p=e^{-E}/Z$ is water pooled in the valleys. $Z$ is the total volume of water, the quantity that forces training to need negative samples. In 2D, $Z$ can be computed on a grid and shown changing as the terrain moves.
- **A temperature dial.** $p\propto e^{-E/T}$. As $T\to0$, the water drains into the deepest points; as $T$ grows, it floods everything flat. Boltzmann–Gibbs as a knob.
- **Push-down and push-up pins.** The training gradient is $\mathbb E_{\text{data}}[\nabla_\theta E]-\mathbb E_{p_\theta}[\nabla_\theta E]$. Pins press the terrain *down* under data points and *up* under the model's own samples. When the two clouds coincide, the forces cancel. Equilibrium is the fixed point of learning.
- **A normal starting bowl.** Write $E_\theta(x)=\tfrac12\lVert x\rVert^2+f_\theta(x)$ with $f_\theta\approx0$ at initialization. The initial model is then exactly the standard normal, which makes "the initial shape comes from a normal distribution" literal.
- **Langevin as a marble with jitter.** The update is $x\leftarrow x-\eta\nabla E(x)+\sqrt{2\eta}\,\epsilon$.
  - **Noise off:** this is gradient descent. Marbles roll to the nearest minimum and stop. Every sample is a minimum, with no variety.
  - **η = 0:** nothing moves. That is the identity kernel, which trivially preserves every distribution.
  - **η too large:** marbles overshoot, and the stationary distribution is visibly biased. A Metropolis-correction toggle removes the bias.
  - **Mixing:** watch how long a marble waits before it crosses from one valley to another.

## 6. Diffusion

- **Ink dispersing.** The forward process melts the data cloud into a Gaussian, with a time slider to play it forward and backward. Each point's noising path is a jittery walk drifting toward the center.
- **A stack of landscapes.** Each time $t$ has its own $p_t$ and score field $\nabla\log p_t$.
  - At large $t$, the arrows all point to the center. That is the easy problem.
  - At small $t$, they point to individual modes. That is the hard one.

  Diffusion is a curriculum indexed by $t$. It is also a stack of energies $E_t=-\log p_t$, which links straight back to the EBM.
- **"Where did this come from?"** Hover a point $x_t$ at time $t$. The posterior $p(x_0\mid x_t)$ over the training points is highlighted; it can be computed exactly in 2D. The denoiser $\mathbb E[x_0\mid x_t]$ is its mean (Tweedie's formula), drawn as an arrow into the middle of the highlighted cloud. At large $t$ the arrow points to the data mean; at small $t$, to the nearest training point.
- **Two ways back.** The reverse SDE (jittery paths) and the probability-flow ODE (smooth paths) have the same marginals at every $t$ but tell different particle stories. A noise toggle switches between them.
- **The training set is the truth.** Training data is a finite subset of $p_{\text{data}}$. The exact optimum of the denoising objective is the score of the *noised training set*, and its reverse process returns training points. Generalization becomes visible: samples that land *between* training points.

## 7. Flow matching

- **A bundle of straight lines.** Pair noise points with data points and draw the straight segments between them. Many segments pass through any given $(x,t)$, pointing different ways. The learned velocity there is their average: one arrow as the mean of a bundle.
- **Crossing lines, non-crossing streams.** The training segments cross freely. The averaged field integrates into curved streamlines that never cross. Conditional vs. marginal velocity, as a picture.
- **The coupling knob.**
  - Random pairing: tangled segments and curved streamlines.
  - Optimal-transport pairing: untangled segments, straighter streamlines, and fewer steps needed.
- **One step lands on the mean.** With independent pairing, the velocity at $t=0$ is $\mathbb E[x_1]-x$. A single Euler step therefore sends every particle to the data mean. Add steps, and the modes reappear. A slider for the number of steps makes this visible.
- **A continuous normalizing flow.** The flow map is a diffeomorphism too: the NF rubber sheet, now deforming continuously. The Jacobian becomes the divergence, $\frac{d}{dt}\log p=-\nabla\!\cdot v$, and a divergence heatmap shows where the ink spreads and where it gathers. Flow matching trains without ever computing it.

---

## Bridges between families

| Pair | What connects them |
|---|---|
| NF ↔ flow matching | A diffeomorphism in discrete steps vs. in continuous time; the Jacobian vs. the divergence |
| NF ↔ GAN | An invertible pushforward that keeps its density vs. a foldable pushforward that doesn't |
| EBM ↔ diffusion | One landscape with Langevin vs. a stack of landscapes with annealed dynamics |
| Diffusion ↔ flow matching | The probability-flow ODE *is* a flow; the denoiser and the velocity are two readings of the same posterior |
| VAE ↔ GAN / NF | The decoder mean is a non-invertible pushforward; the VAE adds a stochastic decoder and a learned approximate inverse (the encoder) |
| AR ↔ NF | Affine autoregressive flows (MAF) are both: an autoregressive model read as a diffeomorphism |

## The five questions (from the brainstorm)

Ask every family the same questions at the same points in training:

1. What does this model currently believe a Tree is?
2. Where is that belief stored: density, energy, discriminator, latent decoder, score, or velocity?
3. How does it produce one new Tree from that belief?
4. What does a failure look like geometrically?
5. What information does its training objective actually provide?
