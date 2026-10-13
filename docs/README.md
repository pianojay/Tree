# Tree documentation

Start with the [whitepaper](whitepaper.md) for Tree's purpose, mathematical framing, and model families. Use the [repository README](../README.md) to install and run the application.

The remaining documents preserve development reasoning and evidence. Their presence does not mean the owner authored every passage, endorsed every proposal, or intended every record as public reading material. Status statements inside each document distinguish current implementation from historical discussions and open questions.

## Development

| Document | Purpose |
|---|---|
| [Roadmap](roadmap.md) | Implemented Phases 1–5 and pending work |
| [Implementation contract](implementation-contract.md) | Current setup and runtime behavior |
| [Runtime comparison](runtime-comparison.md) | Matched backbone comparisons and limitations |
| [Validation results](validation/runtime-results.json) | Recorded machine-readable checks |
| [jax-js bug memo](jax-js-bugs.md) | Historical upstream bug and current verification |
| [Decisions](decisions.md) | Owner decisions, review history, and withdrawn proposals |

## Research and historical records

| Document | Purpose |
|---|---|
| [Ground truth](ground-truth.md) | Dataset design reasoning |
| [Model families](model-families.md) | Family-specific mathematical objects and proposed interactions |
| [Sample space and priors](sample-space-and-priors.md) | Earlier analysis and unresolved alternatives |
| [Circular variables](background-circular-variables.md) | Background preceding Tree's Euclidean setup |
| [Brainstorm log](brainstorm-log.md) | Summary of the originating discussion |
| [Substrate audit](substrate-audit.md) | Historical runtime investigation |
| [References](references/README.md) | Raw conversation export, separate VAE lab, and provenance |

## Public release boundary

Public release contents are not yet approved. The intended main reading material is the whitepaper; setup instructions, code, licenses, and necessary attribution also belong in a usable release. Review the whitepaper's supporting links when selecting the final file list so the release has no dangling references.

Before publishing a curated release, review authorship, personal conversation content, and permission to redistribute third-party references. Attribution for Professor Minhyuk Sung's lab records its source; it does not establish redistribution permission.

Prefer a reviewed release file list and a tagged version over a separately maintained public branch. If public contents must have separate Git history, create a curated repository or an orphan release branch after that review. An ordinary branch sharing main's history retains earlier documents even when they are removed from its current tree. No public-release branch has been created during this cleanup.
