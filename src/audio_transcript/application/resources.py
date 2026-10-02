"""Pinned resource manifest, setup plan, provenance records and local config creation."""

from __future__ import annotations

import json
import platform
import re
import shutil
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

MANIFEST_NAME = "resources.json"
KINDS = ("model", "projector", "vad", "runtime")
PROVENANCE_NAME = "PROVENANCE.txt"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = (
    "id",
    "kind",
    "platforms",
    "url",
    "sha256",
    "size_bytes",
    "license",
    "source",
    "destination",
)


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    platforms: tuple[str, ...]
    url: str
    revision: str
    sha256: str
    size_bytes: int
    license: str
    source: str
    destination: str


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    alias: str
    response_mode: str
    resources: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    resources: dict[str, Resource]
    profiles: dict[str, Profile]
    default_profile: str


@dataclass(frozen=True)
class PlanItem:
    resource: Resource
    path: Path
    status: str  # verified | download | resume | conflict | installer | installed | other-platform
    present_bytes: int = 0

    @property
    def download_bytes(self) -> int:
        if self.status in {"download", "resume"}:
            return self.resource.size_bytes - self.present_bytes
        return 0


def default_manifest_path() -> Path:
    for candidate in (Path(__file__).resolve().parents[3] / MANIFEST_NAME, Path(MANIFEST_NAME)):
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[3] / MANIFEST_NAME


def current_platform() -> str:
    machine = platform.machine().lower()
    arch = "x64" if machine in {"amd64", "x86_64"} else machine
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    return f"{system}-{arch}"


def parse_manifest(data: object) -> Manifest:
    """Validate a decoded manifest; raise ValueError with an actionable message."""
    if not isinstance(data, dict):
        raise ValueError("The resource manifest must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported resource manifest schema_version (expected 1)")
    raw_resources = data.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise ValueError("The resource manifest needs a non-empty 'resources' list")
    resources: dict[str, Resource] = {}
    for entry in raw_resources:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest resource must be an object")
        missing = [key for key in _REQUIRED if key not in entry]
        if missing:
            raise ValueError(f"Resource {entry.get('id', '?')} is missing: {', '.join(missing)}")
        rid = entry["id"]
        if not isinstance(rid, str) or not rid or rid in resources:
            raise ValueError(f"Resource ids must be unique non-empty strings: {rid!r}")
        if entry["kind"] not in KINDS:
            raise ValueError(f"Resource {rid}: kind must be one of {', '.join(KINDS)}")
        platforms = entry["platforms"]
        if (
            not isinstance(platforms, list)
            or not platforms
            or not all(isinstance(item, str) for item in platforms)
        ):
            raise ValueError(f"Resource {rid}: platforms must be a non-empty list of strings")
        if not isinstance(entry["url"], str) or not entry["url"].startswith("https://"):
            raise ValueError(f"Resource {rid}: url must be an https URL")
        if not isinstance(entry["sha256"], str) or not _SHA256.match(entry["sha256"]):
            raise ValueError(f"Resource {rid}: sha256 must be 64 lowercase hex characters")
        size = entry["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Resource {rid}: size_bytes must be a positive integer")
        destination = entry["destination"]
        pure = PurePosixPath(destination) if isinstance(destination, str) else None
        if (
            pure is None
            or not destination
            or "\\" in destination
            or pure.is_absolute()
            or ".." in pure.parts
            or re.match(r"^[A-Za-z]:", destination)
        ):
            raise ValueError(
                f"Resource {rid}: destination must be a relative path with forward slashes"
            )
        for key in ("license", "source"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError(f"Resource {rid}: {key} must be a non-empty string")
        resources[rid] = Resource(
            id=rid,
            kind=entry["kind"],
            platforms=tuple(platforms),
            url=entry["url"],
            revision=str(entry.get("revision", "")),
            sha256=entry["sha256"],
            size_bytes=size,
            license=entry["license"],
            source=entry["source"],
            destination=destination,
        )
    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("The resource manifest needs a non-empty 'profiles' object")
    profiles: dict[str, Profile] = {}
    for name, entry in raw_profiles.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("resources"), list):
            raise ValueError(f"Profile {name}: needs a 'resources' list")
        unknown = [rid for rid in entry["resources"] if rid not in resources]
        if unknown:
            raise ValueError(f"Profile {name} references unknown resource(s): {', '.join(unknown)}")
        profiles[name] = Profile(
            name=name,
            description=str(entry.get("description", "")),
            alias=str(entry.get("alias", name)),
            response_mode=str(entry.get("response_mode", "qwen3-asr")),
            resources=tuple(entry["resources"]),
        )
    default = data.get("default_profile", next(iter(profiles)))
    if default not in profiles:
        raise ValueError(f"default_profile {default!r} is not a defined profile")
    return Manifest(resources=resources, profiles=profiles, default_profile=default)


def load_manifest(path: Path | None = None) -> Manifest:
    target = path or default_manifest_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Resource manifest not found: {target}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read resource manifest {target}: {exc}") from exc
    return parse_manifest(data)


def get_profile(manifest: Manifest, name: str | None) -> Profile:
    chosen = name or manifest.default_profile
    if chosen not in manifest.profiles:
        raise ValueError(f"Unknown profile {chosen!r}; available: {', '.join(manifest.profiles)}")
    return manifest.profiles[chosen]


def resource_path(store: Path, resource: Resource) -> Path:
    return store.joinpath(*PurePosixPath(resource.destination).parts)


def build_plan(
    manifest: Manifest,
    profile: Profile,
    store: Path,
    *,
    platform_id: str | None = None,
    hasher: Callable[[Path], str],
) -> list[PlanItem]:
    """Classify each profile resource without changing anything on disk."""
    here = platform_id or current_platform()
    items: list[PlanItem] = []
    for rid in profile.resources:
        res = manifest.resources[rid]
        path = resource_path(store, res)
        if "any" not in res.platforms and here not in res.platforms:
            items.append(PlanItem(res, path, "other-platform"))
        elif res.kind == "runtime":
            installed = (path / "runtime-manifest.json").is_file()
            items.append(PlanItem(res, path, "installed" if installed else "installer"))
        elif path.is_file():
            if path.stat().st_size == res.size_bytes and hasher(path) == res.sha256:
                items.append(PlanItem(res, path, "verified", res.size_bytes))
            else:
                items.append(PlanItem(res, path, "conflict", path.stat().st_size))
        else:
            part = path.with_name(path.name + ".part")
            have = part.stat().st_size if part.is_file() else 0
            if 0 < have < res.size_bytes:
                items.append(PlanItem(res, path, "resume", have))
            else:
                items.append(PlanItem(res, path, "download"))
    return items


def free_space(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _mb(value: int) -> str:
    return f"{value / 1_000_000:,.1f} MB"


_STATUS_TEXT = {
    "verified": "present, verified",
    "download": "to download",
    "resume": "partial, will resume",
    "conflict": "DIFFERENT FILE EXISTS",
    "installer": "installer (see command below)",
    "installed": "installed (not re-verified)",
    "other-platform": "not for this platform",
}


def format_plan(items: list[PlanItem], store: Path, free_bytes: int) -> list[str]:
    width = max(len(item.resource.id) for item in items)
    lines = [f"Resource store: {store}", ""]
    lines.append(f"{'Resource'.ljust(width)}  {'Size':>12}  Status")
    for item in items:
        lines.append(
            f"{item.resource.id.ljust(width)}  {_mb(item.resource.size_bytes):>12}  "
            f"{_STATUS_TEXT[item.status]}"
        )
    download = sum(item.download_bytes for item in items)
    runtime = sum(item.resource.size_bytes for item in items if item.status == "installer")
    lines.append("")
    lines.append(f"To download now: {_mb(download)}")
    if runtime:
        lines.append(f"Runtime download by the installer: {_mb(runtime)}")
    lines.append(f"Free disk space at the store: {_mb(free_bytes)}")
    if download > free_bytes:
        lines.append("WARNING: not enough free disk space for this download.")
    return lines


def runtime_command(store: Path, runtime_dir: Path | None = None) -> str:
    target = runtime_dir or store / "runtimes" / "llama.cpp-cuda"
    return (
        "powershell -NoProfile -ExecutionPolicy Bypass -File "
        f'scripts\\install-llamacpp-cuda.ps1 -RuntimeDirectory "{target}"'
    )


def write_provenance(path: Path, resource: Resource, *, now: datetime | None = None) -> None:
    """Create or update the per-file block in ``PROVENANCE.txt`` next to ``path``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = [
        f"File:     {path.name}",
        f"Resource: {resource.id}",
        f"Source:   {resource.url}",
        f"Revision: {resource.revision}",
        f"SHA-256:  {resource.sha256}",
        f"Size:     {resource.size_bytes}",
        f"License:  {resource.license}",
        f"Page:     {resource.source}",
        f"Fetched:  {stamp}",
    ]
    target = path.parent / PROVENANCE_NAME
    blocks: list[str] = []
    if target.is_file():
        for existing in target.read_text(encoding="utf-8").split("\n\n"):
            text = existing.strip()
            if text and not text.startswith(f"File:     {path.name}\n"):
                blocks.append(text)
    blocks.append("\n".join(block))
    target.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _set_key(text: str, section: str, key: str, value: str) -> str:
    """Replace ``key = ...`` inside ``[section]``; the template must contain it."""
    pattern = re.compile(
        rf"(^\[{re.escape(section)}\]\s*\n(?:(?!^\[).*\n)*?)(^{re.escape(key)}\s*=.*$)",
        re.MULTILINE,
    )
    if not pattern.search(text):
        raise ValueError(f"The config template has no '{key}' in [{section}]")
    return pattern.sub(lambda m: m.group(1) + f"{key} = {_toml_string(value)}", text, count=1)


def render_config(template: str, manifest: Manifest, profile: Profile, store: Path) -> str:
    """Fill the local-config template for ``profile``; paths are relative to the store."""
    by_kind = {
        manifest.resources[rid].kind: manifest.resources[rid]
        for rid in profile.resources
        if manifest.resources[rid].kind != "runtime"
    }
    for kind in ("model", "projector", "vad"):
        if kind not in by_kind:
            raise ValueError(f"Profile {profile.name} has no '{kind}' resource")
    runtime = next(
        (
            manifest.resources[rid]
            for rid in profile.resources
            if manifest.resources[rid].kind == "runtime"
        ),
        None,
    )
    text = template
    for section, key, value in (
        ("audio_transcript", "model", profile.alias),
        ("audio_transcript", "response_mode", profile.response_mode),
        ("audio_transcript", "vad_model_path", by_kind["vad"].destination),
        ("resources", "store", store.resolve().as_posix()),
        ("server", "runtime_dir", runtime.destination if runtime else "runtimes/llama.cpp-cuda"),
        ("server", "model_path", by_kind["model"].destination),
        ("server", "projector_path", by_kind["projector"].destination),
        ("server", "alias", profile.alias),
        ("server", "state_dir", f"services/{profile.alias}"),
    ):
        text = _set_key(text, section, key, value)
    tomllib.loads(text)
    return text
