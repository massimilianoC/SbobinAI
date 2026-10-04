"""The CLI documentation is generated from one source; these tests keep it honest."""

import ast
import contextlib
import dataclasses
import io
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from audio_transcript import clidoc, climan
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.cli import _parser, _subparser, main
from audio_transcript.config import _SERVER_KEYS, AppConfig, load_config

ROOT = Path(__file__).resolve().parent.parent
CLI_SOURCE = ROOT / "src" / "audio_transcript" / "cli.py"
COMMANDS = [command.name for command in clidoc.COMMANDS]


def run_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
    return code, out.getvalue(), err.getvalue()


def subparsers():
    parser = _parser()
    return {name: _subparser(parser, name) for name in COMMANDS}


def visible_actions(sub):
    return [
        action
        for action in sub._actions
        if action.help is not None and action.help != "==SUPPRESS==" and action.dest != "help"
    ]


class HelpTests(unittest.TestCase):
    def test_every_command_has_help_and_every_option_has_text(self):
        parser = _parser()
        root_actions = {a.dest: a for a in parser._actions}
        self.assertIn("command", root_actions)
        listed = [a for a in parser._subparsers._group_actions[0]._choices_actions]
        self.assertEqual([a.dest for a in listed], COMMANDS)
        for action in listed:
            self.assertTrue(action.help and action.help.strip(), action.dest)
        for name, sub in subparsers().items():
            for action in sub._actions:
                if action.dest == "help":
                    continue
                self.assertTrue(
                    action.help and action.help.strip(),
                    f"{name} {action.option_strings or action.dest} has no help",
                )

    def test_documented_options_match_argparse(self):
        for name, sub in subparsers().items():
            parsed = set()
            for action in sub._actions:
                if action.dest == "help" or action.help == "==SUPPRESS==":
                    continue
                parsed.update(action.option_strings)
            documented = set()
            for opt in clidoc.visible_options(clidoc.COMMAND_BY_NAME[name]):
                if opt.positional:
                    continue
                documented.update(opt.flags)
                if opt.kind == "bool":
                    documented.add(f"--no-{opt.long_flag[2:]}")
            self.assertEqual(parsed, documented, name)

    def test_every_command_help_has_examples_and_exit_status_last(self):
        for name, sub in subparsers().items():
            text = sub.format_help()
            self.assertIn("Examples:", text, name)
            self.assertIn("Exit status:", text, name)
            self.assertGreater(text.index("Exit status:"), text.index("Examples:"))
            self.assertNotIn("\n\n\n", text)
            tail = text[text.index("Exit status:") :]
            self.assertNotRegex(tail, r"\n[A-Z][a-z ]+:\n")

    def test_help_command_equals_dash_help(self):
        for name in COMMANDS:
            code, via_help, _ = run_main("help", name)
            self.assertEqual(code, 0)
            code, via_flag, _ = run_main(name, "--help")
            self.assertEqual(code, 0)
            self.assertEqual(via_help, via_flag, name)
        _, overview, _ = run_main("help")
        _, flagged, _ = run_main("--help")
        self.assertEqual(overview, flagged)

    def test_unknown_help_topic_is_usage_error(self):
        code, _, err = run_main("help", "nope")
        self.assertEqual(code, 2)
        self.assertIn("invalid choice", err)

    def test_help_is_ascii_and_fits_80_columns(self):
        texts = {"overview": climan.overview()}
        texts.update({name: sub.format_help() for name, sub in subparsers().items()})
        for name, text in texts.items():
            self.assertTrue(text.isascii(), name)
            in_examples = False
            for line in text.splitlines():
                if line.startswith("Examples:"):
                    in_examples = True
                elif line.endswith(":") and not line.startswith(" "):
                    in_examples = False
                if not in_examples:
                    self.assertLessEqual(len(line), 80, f"{name}: {line!r}")

    def test_overview_content(self):
        text = climan.overview()
        for heading in ("Transcribe:", "Inspect:", "Setup:", "Maintenance:", "Typical workflow:"):
            self.assertIn(heading, text)
        for name in COMMANDS:
            self.assertIn(name, text)
        self.assertIn("sbobinai help <command>", text)
        self.assertIn("sbobinai man", text)
        self.assertIn("--config", text)

    def test_default_values_are_documented(self):
        text = subparsers()["run"].format_help()
        self.assertIn("[default: 15.0]", text)
        self.assertIn("{silero,energy,none}", text)
        self.assertIn("{json,plain,qwen3-asr}", text)


class ManualTests(unittest.TestCase):
    def test_man_json_is_valid_and_complete(self):
        code, out, _ = run_main("man", "--format", "json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["schema_version"], clidoc.SCHEMA_VERSION)
        self.assertEqual([c["name"] for c in data["commands"]], COMMANDS)
        for key in ("exit_codes", "error_codes", "config", "event_types", "environment", "files"):
            self.assertTrue(data[key], key)
        parsers = subparsers()
        for command in data["commands"]:
            sub = parsers[command["name"]]
            argparse_flags = {
                flag
                for action in sub._actions
                if action.dest != "help" and action.help != "==SUPPRESS=="
                for flag in action.option_strings
            }
            json_flags = {
                flag
                for option in command["options"]
                if not option["positional"]
                for flag in option["flags"]
            }
            json_flags |= {
                f"--no-{option['flags'][0][2:]}"
                for option in command["options"]
                if option["negatable"]
            }
            self.assertEqual(argparse_flags, json_flags, command["name"])
            for option in command["options"]:
                for field in ("name", "flags", "type", "default", "choices", "help", "group"):
                    self.assertIn(field, option)
                self.assertTrue(option["help"])
        config_keys = {c["key"] for c in data["config"] if c["table"] == "audio_transcript"}
        fields = set(AppConfig.__dataclass_fields__) - {"resources", "server"}
        self.assertEqual(config_keys, fields)

    def test_man_json_single_command_and_json_flag(self):
        _, out, _ = run_main("man", "run", "--format", "json")
        self.assertEqual([c["name"] for c in json.loads(out)["commands"]], ["run"])
        _, shortcut, _ = run_main("man", "--json")
        self.assertEqual(json.loads(shortcut)["program"], "sbobinai")

    def test_man_text_sections(self):
        _, out, _ = run_main("man")
        for section in (
            "NAME",
            "SYNOPSIS",
            "DESCRIPTION",
            "COMMANDS",
            "OPTIONS",
            "CONFIGURATION",
            "FILES",
            "ENVIRONMENT",
            "EXIT STATUS",
            "ERRORS",
            "EXAMPLES",
            "AGENT USAGE",
            "SEE ALSO",
        ):
            self.assertRegex(out, rf"(?m)^{section}$")
        self.assertTrue(out.isascii())

    def test_man_for_one_command(self):
        code, out, _ = run_main("man", "doctor")
        self.assertEqual(code, 0)
        self.assertIn("sbobinai doctor", out)
        self.assertNotIn("CONFIGURATION", out)

    def test_committed_markdown_matches_generated(self):
        code, out, _ = run_main("man", "--format", "markdown")
        self.assertEqual(code, 0)
        committed = (ROOT / "docs" / "cli-reference.md").read_bytes().decode("utf-8")
        self.assertEqual(out, committed, "run: python scripts/gen-cli-docs.py")

    def test_committed_skill_copies_match_generated(self):
        code, out, _ = run_main("man", "--format", "skill")
        self.assertEqual(code, 0)
        for path in (
            ROOT / "skills" / "sbobinai" / "SKILL.md",
            ROOT / "src" / "audio_transcript" / "skills" / "SKILL.md",
        ):
            self.assertEqual(path.read_bytes().decode("utf-8"), out, str(path))

    def test_skill_manifest_shape(self):
        text = climan.skill_text()
        self.assertTrue(text.startswith("---\nname: sbobinai\ndescription: "))
        head, _, body = text[4:].partition("\n---\n")
        self.assertIn("name: sbobinai", head)
        for name in COMMANDS:
            self.assertIn(f"`{name}`", body)
        for heading in ("## Agent guidance", "## Exit status", "## JSON errors", "## Events"):
            self.assertIn(heading, body)


class ConfigTableTests(unittest.TestCase):
    def test_table_covers_every_app_config_field(self):
        documented = [k.key for k in clidoc.CONFIG_KEYS if k.section == "audio_transcript"]
        fields = set(AppConfig.__dataclass_fields__) - {"resources", "server"}
        self.assertEqual(set(documented), fields)
        self.assertEqual(len(documented), len(set(documented)))
        server = [k.key for k in clidoc.CONFIG_KEYS if k.section == "server"]
        self.assertEqual(set(server), set(_SERVER_KEYS))
        self.assertEqual([k.key for k in clidoc.CONFIG_KEYS if k.section == "resources"], ["store"])

    def test_flags_exist_and_defaults_match(self):
        run_flags = {f for a in subparsers()["run"]._actions for f in a.option_strings}
        defaults = AppConfig()
        for key in clidoc.CONFIG_KEYS:
            if key.section != "audio_transcript":
                continue
            flag = clidoc.config_flag(key.key)
            if flag is not None:
                self.assertIn(flag.split("/")[0], run_flags, key.key)
            if clidoc.config_default(key.key) not in {"unset"}:
                shown = clidoc.config_default(key.key)
                self.assertTrue(shown, key.key)
        self.assertEqual(clidoc.config_default("chunk_seconds"), str(defaults.chunk_seconds))
        self.assertEqual(clidoc.config_default("fallback_temperatures"), "0.2,0.4")

    def test_job_identity_column_matches_the_fingerprint(self):
        base = load_config(None, {"backend": "llamacpp", "vad": "silero"})
        alternatives = {
            "language": "it",
            "prompt": "context",
            "max_duration": 30.0,
            "model_path": Path("model.gguf"),
            "projector_path": Path("proj.gguf"),
            "fallback_temperatures": (0.9,),
            "max_file_size": 1000,
        }

        def fingerprint(config):
            return TranscriptionPipeline._fingerprint(SimpleNamespace(config=config))

        reference = fingerprint(base)
        for key in clidoc.CONFIG_KEYS:
            if key.section != "audio_transcript":
                continue
            current = getattr(base, key.key)
            if key.key in alternatives:
                changed = alternatives[key.key]
            elif isinstance(current, bool):
                changed = not current
            elif isinstance(current, (int, float)):
                changed = current + 3
            elif isinstance(current, Path):
                changed = Path(str(current) + "-other")
            elif isinstance(current, str):
                changed = current + "-other"
            else:
                self.fail(f"no alternative value for {key.key}")
            if key.key == "parallel_requests":
                changed = 2
            altered = dataclasses.replace(base, **{key.key: changed})
            differs = fingerprint(altered) != reference
            self.assertEqual(
                differs,
                not key.identity.startswith("no"),
                f"{key.key}: documented identity {key.identity!r}",
            )


class ExitAndErrorCodeTests(unittest.TestCase):
    def test_exit_table_matches_constants(self):
        table = {code for code, _name, _text in clidoc.EXIT_CODES}
        self.assertEqual(
            table,
            {clidoc.EXIT_OK, clidoc.EXIT_FAILED, clidoc.EXIT_USAGE, clidoc.EXIT_INTERRUPTED},
        )
        self.assertEqual(table, {0, 1, 2, 130})
        for command in clidoc.COMMANDS:
            for code, _text in command.exit_status:
                self.assertIn(code, table, command.name)

    def test_cli_returns_only_table_codes(self):
        tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
            ):
                self.fail(f"literal return {node.value.value!r} at line {node.lineno}")
        allowed = {"EXIT_OK", "EXIT_FAILED", "EXIT_USAGE", "EXIT_INTERRUPTED"}
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertTrue(allowed <= names)

    def test_error_codes_are_used_and_have_valid_status(self):
        source = CLI_SOURCE.read_text(encoding="utf-8")
        valid = {c for c, _n, _t in clidoc.EXIT_CODES}
        for code, status, _text in clidoc.ERROR_CODES:
            self.assertIn(status, valid)
            self.assertIn(f'"{code}"', source, f"error code {code} is not used by cli.py")
        for used in re.findall(r'_fail\(\s*(?:\w+|True|False),\s*"(\w+)"', source):
            self.assertIn(used, {c for c, _s, _t in clidoc.ERROR_CODES})

    def test_dynamic_exit_statuses(self):
        self.assertEqual(run_main("run", "--config", "does-not-exist.toml")[0], 2)
        self.assertEqual(run_main("run", "--bogus")[0], 2)
        self.assertEqual(run_main("man", "--format", "nope")[0], 2)
        self.assertEqual(run_main("server-profile", "--config", "does-not-exist.toml")[0], 2)

    def test_environment_variables_are_really_read(self):
        sources = "".join(
            p.read_text(encoding="utf-8") for p in (ROOT / "src" / "audio_transcript").rglob("*.py")
        )
        tests = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("*.py"))
        for name, _text in clidoc.ENVIRONMENT:
            if name == "PATH":
                self.assertIn("shutil.which", sources)
            elif name == "SILERO_VAD_MODEL":
                self.assertIn(name, tests)
            else:
                self.assertIn(f'"{name}"', sources, name)

    def test_event_types_are_documented_in_events_md(self):
        text = (ROOT / "docs" / "events.md").read_text(encoding="utf-8")
        for kind, _payload in clidoc.EVENT_TYPES:
            self.assertIn(f"`{kind}`", text)


class DocumentationLinksTests(unittest.TestCase):
    def test_readme_and_index_link_the_reference(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/cli-reference.md", readme)
        self.assertIn("sbobinai help", readme)
        self.assertIn("sbobinai man", readme)
        self.assertIn("cli-reference.md", index)

    def test_documented_files_exist(self):
        for relative in (
            "docs/cli-reference.md",
            "skills/sbobinai/SKILL.md",
            "src/audio_transcript/skills/SKILL.md",
            "scripts/gen-cli-docs.py",
            "TEST.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_no_private_paths_in_generated_docs(self):
        for text in (climan.markdown(), climan.skill_text(), climan.man_text()):
            self.assertNotRegex(text, r"[A-Z]:\\Users|/home/|@gmail")


if __name__ == "__main__":
    unittest.main()
