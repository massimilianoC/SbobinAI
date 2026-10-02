# Documentation index

The current project contracts and evidence are organized by purpose. Start with
the specification for behavior, compatibility for runtime setup, and verification
for actual outcomes. All public documents use generic examples.

| Document | Role | Read it for |
| --- | --- | --- |
| [Project README](../README.md) | Operational entry point | Setup, CLI commands and sample selection. |
| [MVP specification](specification.md) | Acceptance contract | Workflow, required behavior, scope and document maintenance. |
| [Architecture](architecture.md) | Design decisions | Domain interfaces, orchestration and adapter boundaries. |
| [Implementation plan](implementation-plan.md) | Delivery plan | Work order, review and validation responsibilities. |
| [Qwen2-Audio framework](framework/nexaAI_Qwen2-Audio.md) | Structured technical reference | Supplied capabilities, historical commands, adapter contract and extensions. |
| [Backend compatibility](backend-compatibility.md) | Runtime evidence | Verified interfaces, constraints and unresolved compatibility. |
| [Shared runtime setup](runtime-setup.md) | Operational contract | Global binaries, local API, model conversion, startup and shutdown. |
| [Inference controls](inference-controls.md) | Model task contract | Prompt, structured output, seed, temperature and experiment identity. |
| [Verification](verification.md) | Dated execution evidence | Automated checks, real audio preparation and inference status. |
| [Documentation policy](documentation-policy.md) | Editorial contract | Source preservation, semantic structure and completion checks. |
| [TODO](TODO.md) | Follow-up scope | Deferred extensions and validation. |
| [Contributor instructions](../AGENTS.md) | Shared agent policy | Development, privacy, documentation and collaboration rules. |

## Document roles and access

Supplied sources are archived beside their structured working documents under
`sources/`; the structured reference links to its complete original. Preserve
archives as provenance and revise the working reference for readability.

Private vision and review materials remain in ignored local directories. Their
local documents provide navigation to their sources and public specifications;
the public index does not depend on private files being published.

Semantic IDs identify intent and obligations. Status tables distinguish requested
capabilities, implemented behavior, verified results and future extensions.
At session completion, follow the documentation policy to keep these aligned.
