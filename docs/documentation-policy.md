# Documentation maintenance policy

**Established:** 2026-10-02  
**Scope:** vision, framework references, specifications, architecture and reviews

## Purpose and reading paths

At the end of each development or review session, revise the affected documents
so they remain organized, digested, readable and useful for the next reader or
agent. Preserve original content and provenance while making its meaning easier
to find and apply. This is part of completing the work, not optional cleanup.

Use the [documentation index](README.md) to find current contracts and evidence.
Source archives record what was originally supplied; structured documents explain
intent and context; specifications state the agreed behavior; verification records
what was actually checked. Keep these roles explicit.

## Preservation rules

- Before the first editorial rewrite of supplied material, retain an exact source
  copy in its local `sources/` directory and verify byte equality or SHA-256.
- Link the structured document to its source archive. Preserve original commands,
  numbers, claims, qualifications, unresolved points and references.
- Keep archives immutable. Archive later supplied revisions separately rather
  than replacing the earlier original or adding editor commentary inside it.
- Reorganize the working document; an archive alone does not satisfy readability.
  Include a clear source-to-requirement mapping or topic coverage table.
- Correct transcription mistakes in the structured version only. State how an
  ambiguous model/term was normalized and leave the raw source recoverable.
- Identify editorial additions, user follow-ups, implementation decisions,
  proposals and verified results. Retain deferred intentions with their status.
- Retain existing privacy exclusions. Private source archives and analyzed-media
  details must not leak into public indexes, specifications or examples.

## Semantic structure

Use a descriptive title, document role, source, date and review status. Add a
contents list or reading paths for substantial documents. Use a consistent
heading hierarchy and short sections organized by meaning, not dictation order.

For each relevant concept, make its **intent**, **inputs/outputs**, **constraints**,
**relationships**, **implementation implications**, **status** and **evidence**
easy to identify. Useful headings include goals, workflow, capabilities,
requirements, decisions, validation, open questions and extension hints. Adapt
the structure to the document; do not add empty template sections.

Use stable IDs such as `VIS-01`, `FW-01` and `DOC-01` for semantic traceability.
Preserve IDs when moving sections; do not renumber existing requirements just
to insert a new one. Add focused hints explaining overloaded terms, prerequisites
and development consequences. Label proposed schemas and features as proposals.

Cross-link requirements to the authoritative specification, architecture,
implementation entry points and relevant verification evidence. A table can
compare original intent, current support, limitations and next evidence needed.
Source marketing claims must not become acceptance results.

## Session completion procedure

1. Identify the documents affected by the delivered work and user follow-ups.
2. Preserve new source material before rewriting; check topic/content coverage.
3. Restructure the working documents and update semantic IDs, status and hints.
4. Reconcile related specifications, architecture, plans and dated evidence;
   record unresolved differences explicitly.
5. Update the documentation index and cross-links. Keep private analysis inside
   its ignored local area and public documentation in English. Vision/review
   analysis may remain in Italian; an immutable supplied archive keeps its language.
6. Check relative file links, Markdown anchors, code fences, heading hierarchy,
   UTF-8 text and privacy boundaries. Use generic examples for public commands.
7. Add a brief dated revision record with preservation and material extensions.
   Report documentation completion and any remaining substantive blockers honestly.

This procedure does not require approval for routine authorized editorial changes,
does not require application tests for documentation-only edits, and does not
authorize publishing or committing private source material.

## Acceptance requirements

| ID | Requirement | Completion evidence |
| --- | --- | --- |
| DOC-01 | Preserve supplied content | Linked immutable source copy with initial equality/hash check. |
| DOC-02 | Provide an accessible structured version | Clear title, role, headings, navigation and readable topic coverage. |
| DOC-03 | Make semantics actionable | Stable IDs, concept hints, dependencies and development links. |
| DOC-04 | Distinguish provenance and status | Original claims, user additions, decisions, proposals and verification labeled. |
| DOC-05 | Maintain consistency at completion | Related specs/index updated; links and privacy boundaries checked. |

**Revision 2026-10-02:** formalized the user's requirement for source-preserving,
semantically structured document revision at session completion.
