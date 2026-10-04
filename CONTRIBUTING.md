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

## Translating the guided wizard

The wizard's questions and messages live in one JSON file per language in
[`src/audio_transcript/locales/wizard/`](src/audio_transcript/locales/wizard/);
`en.json` is the reference.

- **Fix a translation:** edit the value in that language's file; keep every
  `{placeholder}` exactly as in `en.json`.
- **Add a language:** copy `en.json` to `<code>.json` (ISO 639-1, for example
  `nl.json`), translate the values and `_language` (the language's own name),
  then add the code to `UI_LANGUAGES` in
  `src/audio_transcript/application/wizard_text.py` and to the `--ui-language`
  choices in `clidoc.py`.
- Keep the column alignment of the summary and result lines (`label : {value}`).
- `tests/test_wizard_text.py` checks that every file has the same keys and
  placeholders as English. Native-speaker reviews of the existing files are
  especially welcome.

## Contributor License Agreement

The project is dual-licensed (AGPL-3.0-only and a commercial license, see
[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). To keep that possible,
contributions can be merged only after the contributor agrees to the
[Individual Contributor License Agreement](CLA.md) (based on the Apache ICLA),
which lets the maintainer license contributions under both the AGPL and
commercial terms. A CLA check on pull requests records the agreement.

## Reporting security issues

Do not open public issues for vulnerabilities; see [SECURITY.md](SECURITY.md).
