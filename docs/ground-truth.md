# Ground Truth

**Status:** Working notes. The ground truth is **deliberately not fixed**. It is curated and re-curated as the project goes.
**Parent:** [whitepaper.md](whitepaper.md)

---

## 1. What the ground truth is for

$p_{\text{data}}$ is the thing each principle acts on. The ground truth has to be non-trivial enough that this action is worth watching.

It is **not** built to make models fail, and **not** built to be solved. Either outcome is fine. What matters is that whatever happens can be traced, sensibly and intuitively, to the principle that produced it. An easy problem gives the principles nothing to show.

Consequences:

- **It's the opposite of pretty.** Think of actual raw data rather than a tidy parametric construction.
- **It's iterated.** We keep changing it until it provokes something worth seeing. Versions are kept, because any observed behavior is only meaningful relative to the version that produced it.
- **It isn't solved right away,** and it may never be. That's fine.
- **Semantics are examples, not specifications.** "(green, orange, orange) is a Tree; (green, yellow, orange) is not" describes the *kind of dynamics* we want: modes, dependencies between pixels, forbidden combinations next to valid ones. It is not a rulebook to program. Some semantics will hold and some won't, just as with raw data.

## 2. How it is represented

- **A sampling program.** $p_{\text{data}}$ is whatever a generator script emits. How it is constructed is free: hand-placed clusters, curves, rejection rules, noise, arbitrary transformations. Nothing requires a closed-form density.
- **The training set is a finite subset** of $p_{\text{data}}$, drawn once and kept fixed. Fresh draws from the same program are available as held-out data, and as many as we like for estimating the true distribution, which is easy in 2D.
- **Versioned:** every ground truth has a name and a version, and is reproducible from its program and seed.

Analytic objects (an exact density, an exact score) are a possible bonus for particular versions. They are not a design requirement.

## 3. The ladder

Several ground truths exist side by side, from a debugging case to a hard one:

| Rung | Purpose | Character |
|---|---|---|
| **Simple** | Check that each miniature model *learns at all*; debug the implementations | One blob, or something equally boring |
| **Multimodal** | The first non-trivial case | A few separated modes |
| **Challenging** | The actual object of study. Iterated, and expected to grow into several variants. | Highly curated so that each principle's behavior, success or failure, is non-trivial and traceable |

The simple rung is a tool, not a result. It is how we tell "the code is broken" apart from "the principle fails here".

## 4. Ingredients (a palette, not a plan)

These are kinds of difficulty that raw data has and that the challenging rung can draw from. They are listed only to have words for things. What each one provokes is for the experiments to show, not for this list to predict.

- Several modes, with unequal weights and unequal widths
- Rare modes
- Thin or curved support, as opposed to round blobs
- Forbidden regions right next to valid ones, or between them
- Structure in the joint that is absent from the marginals
- Mass near the edges of the displayed range
- Sharp edges, holes, and discontinuities in density
- Clutter: outliers and background noise

## 5. Dimensionality

- **Primary: 2 pixels.** Everything can be drawn on one square (from the brainstorm).
- **1 and 3 pixels** come later. How to render them is deliberately left open until the 2-pixel work suggests an answer.
