# Contributing

Thank you for your interest. Bug reports, reproducible test cases and focused
pull requests are welcome.

## Before you start

- Read the [specification](docs/specification.md) (acceptance contract) and
  [AGENTS.md](AGENTS.md) (project rules, also followed by AI coding agents).
- Keep the core on the Python standard library; new dependencies must be
  optional extras loaded lazily behind the domain protocols.
- Never commit media, transcripts, model files, logs, credentials or absolute
  machine paths. Use generic examples in documentation.

## Workflow (fork and pull request)

All changes, including the maintainer's, go through forks or branches and pull
requests; `main` is never pushed to directly.

1. Open an issue describing the problem or proposal.
2. Fork the repository, create a topic branch, add tests that describe the
   behaviour, then the change.
3. Run the checks:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   .\.venv\Scripts\python.exe -m ruff check src tests scripts
   .\.venv\Scripts\python.exe -m ruff format --check src tests scripts
   ```

4. Update the affected documents (see the
   [documentation policy](docs/documentation-policy.md)).
5. Open a pull request explaining what changed and how it was verified.

## Contributor License Agreement

The project is dual-licensed (AGPL-3.0-only and a commercial license, see
[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). To keep that possible,
contributions can be merged only after the contributor agrees to the
[Individual Contributor License Agreement](CLA.md) (based on the Apache ICLA),
which lets the maintainer license contributions under both the AGPL and
commercial terms. A CLA check on pull requests records the agreement.

## Reporting security issues

Do not open public issues for vulnerabilities; see [SECURITY.md](SECURITY.md).
