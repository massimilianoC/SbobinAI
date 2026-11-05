"""Selectable public artifacts, private canonical exports and verified intermediate ZIPs."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from ..config import DEFAULT_OUTPUT_FORMATS, AppConfig
from ..domain.models import TranscriptionError
from ..fsutil import replace_with_retry
from .state import atomic_json, read_json

OUTPUT_FILES = {
    "txt": "transcript.txt",
    "md": "transcript.md",
    "srt": "transcript.srt",
    "vtt": "transcript.vtt",
    "json": "transcript.json",
    "report-md": "run-report.md",
    "report-json": "report.json",
    "mp3": "audio.mp3",
    "flac": "audio.flac",
    "mp4": "video.mp4",
}
CANONICAL_FILES = tuple(OUTPUT_FILES[key] for key in DEFAULT_OUTPUT_FORMATS)


def copy_artifact(source: Path, destination: Path) -> None:
    """Publish a complete generated artifact atomically; the source remains recoverable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, target, 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != source.stat().st_size:
            raise OSError("Artifact copy was incomplete")
        replace_with_retry(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ArtifactDelivery:
    def __init__(self, config: AppConfig, job_id: str):
        self.config = config
        self.output = config.output_dir / job_id
        self.work = config.process_dir / job_id
        self.canonical = self.work / "artifacts"

    @property
    def enabled(self) -> bool:
        return (
            self.config.archive_intermediate
            or self.config.output_formats != DEFAULT_OUTPUT_FORMATS
            or self.canonical.is_dir()
        )

    def existing(self, name: str) -> Path:
        """Resolve a canonical export, including versions made before delivery selection."""
        private = self.canonical / name
        return private if private.is_file() else self.output / name

    def capture(self) -> None:
        for name in CANONICAL_FILES:
            source = self.output / name
            if name == "run-report.md" and (self.canonical / name).is_file():
                # The pipeline renders this directly into the canonical location.
                # A public copy from a previous selection must not overwrite it.
                continue
            if source.is_file():
                copy_artifact(source, self.canonical / name)

    def publish(self) -> None:
        """Only selected known artifacts remain loose in the user-facing version folder."""
        for key, name in OUTPUT_FILES.items():
            private, public = self.canonical / name, self.output / name
            if key in self.config.output_formats:
                if not private.is_file():
                    raise TranscriptionError(f"Requested output is missing: {name}")
                copy_artifact(private, public)
            elif public.is_file():
                if not private.is_file():
                    copy_artifact(public, private)
                public.unlink()

    def validate_request(self, info, processor) -> None:
        """Refuse an MP4 request that cannot succeed, before any inference time is spent."""
        if "mp4" not in self.config.output_formats:
            return
        if (self.canonical / OUTPUT_FILES["mp4"]).is_file() and not self.config.force:
            return  # already rendered; recovery needs no source video
        if not info.has_video:
            raise TranscriptionError(
                "MP4 output needs a recording with a video stream; this source is audio only. "
                "Deselect MP4 and run again."
            )
        check = getattr(processor, "check_video_encoder", None)
        if check is not None:
            check()

    def prepare(
        self, source: Path, processor, duration: float, stage=None, progress=None, warn=None
    ) -> None:
        """Build and verify every requested delivery before the source may be deleted.

        ``stage(name)`` returns a context manager timing a section (``audio_export``,
        ``video_export``) and yields a dict that receives extra stage data; ``progress(info)``
        receives throttled render progress; ``warn(message)`` hears encoder fall-throughs. All
        are optional so the same code serves the normal and the reuse path.
        """
        if not self.enabled:
            return
        stage = stage or (lambda name: contextlib.nullcontext({}))
        render = None
        self.capture()
        if any(key in self.config.output_formats for key in ("mp3", "flac")):
            with stage("audio_export"):
                for key in ("mp3", "flac"):
                    if key in self.config.output_formats:
                        self._deliver(key, source, processor, duration, progress)
        if "mp4" in self.config.output_formats:
            with stage("video_export") as extra:
                info = self._deliver("mp4", source, processor, duration, progress, warn)
                render = getattr(info, "render", None)
                if render and extra is not None:
                    extra.update(render)
        self.publish()
        previous = read_json(self.work / "delivery.json", {}) or {}
        atomic_json(
            self.work / "delivery.json",
            {
                "output_formats": list(self.config.output_formats),
                "archive_intermediate": self.config.archive_intermediate,
                "canonical_exports": "artifacts",
                # Which encoder produced the preview; kept when a re-run reuses the file.
                **(
                    {"video_render": render or previous.get("video_render")}
                    if "mp4" in self.config.output_formats
                    else {}
                ),
            },
        )

    def _deliver(self, key: str, source: Path, processor, duration: float, progress, warn=None):
        destination = self.canonical / OUTPUT_FILES[key]
        if destination.is_file() and not self.config.force:
            info = processor.probe(destination)
        elif key == "mp4":
            srt = self.canonical / OUTPUT_FILES["srt"]
            info = processor.export_preview_video(
                source,
                destination,
                start_seconds=self.config.start_seconds,
                max_duration=self.config.max_duration,
                subtitles=srt if srt.is_file() else None,
                progress=progress,
                warn=warn,
            )
        else:
            info = processor.export_selection_audio(
                source,
                destination,
                sample_rate=self.config.sample_rate,
                audio_format=key,
                start_seconds=self.config.start_seconds,
                max_duration=self.config.max_duration,
            )
        if key == "mp4" and (
            not info.has_video or info.codec != "aac" or (info.video_height or 10**6) > 720
        ):
            destination.unlink(missing_ok=True)
            raise TranscriptionError("The mp4 delivery is not a 720p-or-smaller video")
        if key != "mp4" and info.codec not in ({"mp3"} if key == "mp3" else {"flac"}):
            raise TranscriptionError(f"The {key} delivery has an unexpected codec")
        if abs(info.duration - duration) > max(1.0, duration * 0.001):
            destination.unlink(missing_ok=True)
            raise TranscriptionError(f"The {key} delivery does not cover the transcribed interval")
        return info

    def finish(self) -> None:
        if not self.enabled:
            return
        # Reports have just been finalized; refresh those snapshots before packaging.
        self.capture()
        self.publish()
        if self.config.archive_intermediate:
            archive_intermediate(self.output)


def archive_intermediate(output: Path) -> None:
    """Read back the full ZIP before removing its generated loose directory."""
    output = output.resolve()
    folder = output / "intermediate"
    if not folder.exists():
        return
    if folder.is_symlink() or folder.resolve().parent != output or folder.name != "intermediate":
        raise OSError("Refusing to package an intermediate directory outside its output version")
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    signatures = {
        path.relative_to(folder).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    }
    if any(
        path.is_symlink() or not path.resolve().is_relative_to(folder.resolve())
        for path in folder.rglob("*")
    ):
        raise OSError("Refusing symbolic links inside intermediate exports")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".intermediate-", suffix=".zip", delete=False
        ) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(folder).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != {
                path.relative_to(folder).as_posix() for path in files
            }:
                raise OSError("Intermediate ZIP verification failed; loose files were kept")
        current = {
            path.relative_to(folder).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in folder.rglob("*")
            if path.is_file()
        }
        if current != signatures:
            raise OSError("Intermediate files changed while packaging; loose exports were kept")
        replace_with_retry(temporary, output / "intermediate.zip")
        # The checked target is the fixed child of this exact output version; never a parent.
        shutil.rmtree(folder)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
