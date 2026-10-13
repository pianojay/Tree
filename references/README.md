# Tree reference materials

## VAE interactive lab

[vae_2d_lab.html](vae_2d_lab.html) is credited to **Professor Minhyuk Sung, KAIST**, [mhsung@kaist.ac.kr](mailto:mhsung@kaist.ac.kr), who created it with Claude.

The Tree owner supplied Professor Sung's announcement on 2026-10-13. The supplied message is dated **September 15 at 22:46 KST** and describes a single HTML file that accepts a formula for a two-dimensional density, trains a VAE in the browser, and lets the reader inspect likelihood, the model posterior and encoder approximation, ELBO, and log evidence. No year or message permalink was supplied for that announcement.

The file was preserved from `origin/tfjs-ui`, commit `de8b632b96c7b45f3daee3d3922f1a0f845999ba`. This copy adds an attribution comment and a visible credit; the lab's implementation is unchanged.

Tree was independently devised by its owner, with its early discussion recorded in [Circle Representation Priors](chatgpt-conversation://6ac5c77d-d4d0-83ee-90a4-e65dafe53b12) and the [local brainstorm](../docs/brainstorm-log.md). Professor Sung's lab is a separate reference for interactive mathematical inspection. It is not the owner's Tree implementation or a validated replacement for Tree's VAE.

The lab illustrates useful interactions, especially the comparison of the model posterior with the encoder distribution. Its numerical displays use finite-grid integration and Monte Carlo estimates; its wording about exact evidence and amortization error should be reviewed before being reused in Tree.

## Historical runtime experiments

The following branches are reference material for the [development roadmap](../docs/roadmap.md):

- `bench/tfjs-vs-jaxjs`, commit `370c06f3da1892be75b72335f9c0c11669ad1a4b`: fixed PyTorch fixtures and browser-library comparisons.
- `tfjs-ui`, commit `de8b632b96c7b45f3daee3d3922f1a0f845999ba`: a four-model TensorFlow.js GUI engine, worker support, and saved GUI measurements.

Consult their approaches and failure cases when designing fresh tests. They do not select the new runtime, supply the new implementation, or establish completion of the rebuild.
