"""Convert Nexa's Qwen2-Audio projector GGUF to llama.cpp's MMPROJ layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

SOURCE_ARCHITECTURE = "nano-omni-audio-encoder"
EXPECTED_SOURCE_METADATA = {
    "d_model": 1280,
    "encoder_layers": 32,
    "encoder_attention_heads": 20,
    "n_mel": 128,
    "max_source_positions": 1500,
}
EXPECTED_SOURCE_TENSORS = {
    "audio_tower.conv1.weight": (3, 128, 1280),
    "audio_tower.conv1.bias": (1280,),
    "audio_tower.conv2.weight": (3, 1280, 1280),
    "audio_tower.conv2.bias": (1280,),
    "audio_tower.embed_positions.weight": (1280, 1500),
    "audio_tower.layers.0.fc1.weight": (1280, 5120),
    "multi_modal_projector.linear.weight": (1280, 4096),
    "multi_modal_projector.linear.bias": (4096,),
}


def map_tensor_name(name: str, name_map: Any) -> str:
    """Apply llama.cpp's canonical MMPROJ mapping to a source tensor name."""
    normalized = name
    if normalized.startswith("multi_modal_projector."):
        normalized = "audio." + normalized
    mapped = name_map.get_name(normalized, try_suffixes=(".weight", ".bias"))
    if mapped is None:
        raise ValueError(f"No llama.cpp MMPROJ tensor mapping exists for {name!r}.")
    return mapped


def _field(reader: Any, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None:
        raise ValueError(f"Input projector is missing required GGUF metadata {key!r}.")
    return field.contents()


def _validate_source(reader: Any) -> None:
    architecture = _field(reader, "general.architecture")
    if architecture != SOURCE_ARCHITECTURE:
        raise ValueError(
            f"Unsupported source architecture {architecture!r}; expected {SOURCE_ARCHITECTURE!r}."
        )
    for key, expected in EXPECTED_SOURCE_METADATA.items():
        actual = _field(reader, key)
        if actual != expected:
            raise ValueError(f"Unexpected source metadata {key}={actual!r}; expected {expected}.")

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    for name, expected_shape in EXPECTED_SOURCE_TENSORS.items():
        tensor = tensors.get(name)
        if tensor is None:
            raise ValueError(f"Input projector is missing required tensor {name!r}.")
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {name}: {actual_shape}; expected {expected_shape}."
            )

    skipped = [tensor.name for tensor in reader.tensors if tensor.name == "mel_filters_data"]
    if len(skipped) != 1:
        raise ValueError("Expected exactly one mel_filters_data tensor to omit from the projector.")
    if len(reader.tensors) != 490:
        raise ValueError(f"Expected 490 source tensors, found {len(reader.tensors)}.")


def _write_metadata(writer: Any) -> None:
    writer.add_clip_has_audio_encoder(True)
    writer.add_clip_has_vision_encoder(False)
    writer.add_clip_projector_type("qwen2a")
    writer.add_audio_embedding_length(1280)
    writer.add_audio_feed_forward_length(5120)
    writer.add_audio_block_count(32)
    writer.add_audio_projection_dim(4096)
    writer.add_audio_head_count(20)
    writer.add_audio_attention_layernorm_eps(1e-5)
    writer.add_audio_num_mel_bins(128)
    writer.add_audio_max_pos_emb(1500)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_output(
    path: Path,
    expected_names: set[str],
    expected_shapes: dict[str, tuple[int, ...]],
    expected_types: dict[str, Any],
    gguf: Any,
) -> None:
    reader = gguf.GGUFReader(path)
    try:
        if _field(reader, "general.architecture") != "clip":
            raise ValueError("Converted output does not declare the clip architecture.")
        expected_metadata = {
            "clip.has_audio_encoder": True,
            "clip.has_vision_encoder": False,
            "clip.projector_type": "qwen2a",
            "clip.audio.embedding_length": 1280,
            "clip.audio.feed_forward_length": 5120,
            "clip.audio.block_count": 32,
            "clip.audio.projection_dim": 4096,
            "clip.audio.attention.head_count": 20,
            "clip.audio.attention.layer_norm_epsilon": 1e-5,
            "clip.audio.num_mel_bins": 128,
            "clip.audio.max_pos_emb": 1500,
        }
        for key, expected in expected_metadata.items():
            actual = _field(reader, key)
            if isinstance(expected, float):
                matches = math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"Converted metadata {key}={actual!r}; expected {expected!r}.")
        if len(reader.tensors) != len(expected_names):
            raise ValueError(
                f"Converted output has {len(reader.tensors)} tensors; expected {len(expected_names)}."
            )
        actual_names = {tensor.name for tensor in reader.tensors}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise ValueError(
                f"Converted tensor names differ (missing={missing}, unexpected={unexpected})."
            )
        for tensor in reader.tensors:
            actual_shape = tuple(int(dim) for dim in tensor.shape)
            if actual_shape != expected_shapes[tensor.name]:
                raise ValueError(
                    f"Converted tensor {tensor.name} has shape {actual_shape}; "
                    f"expected {expected_shapes[tensor.name]}."
                )
            if tensor.tensor_type != expected_types[tensor.name]:
                raise ValueError(
                    f"Converted tensor {tensor.name} changed dtype from "
                    f"{expected_types[tensor.name].name} to {tensor.tensor_type.name}."
                )
    finally:
        _release_reader(reader)


def _release_reader(reader: Any) -> None:
    """Release the file mapping explicitly so temporary GGUFs can be renamed on Windows."""
    data = getattr(reader, "data", None)
    mapping = getattr(data, "_mmap", None)
    if mapping is not None:
        mapping.close()


def convert(input_path: Path, output_path: Path) -> Path:
    """Convert one local projector and write an adjacent integrity report."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    report_path = output_path.with_name(output_path.name + ".conversion-report.json")
    if not input_path.is_file():
        raise ValueError(f"Input projector does not exist or is not a file: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different; the source is never modified.")
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output_path}")
    if report_path.exists():
        raise ValueError(f"Refusing to overwrite existing conversion report: {report_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "The optional llama.cpp gguf Python package is required. Run this script with "
            "the project's .venv after installing the llama.cpp conversion dependencies."
        ) from exc

    source_hash = _sha256(input_path)
    reader = gguf.GGUFReader(input_path)
    _validate_source(reader)
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, 32)

    mapped_names: dict[str, str] = {}
    expected_shapes: dict[str, tuple[int, ...]] = {}
    expected_types: dict[str, Any] = {}
    skipped_names: list[str] = []
    for tensor in reader.tensors:
        if tensor.name == "mel_filters_data":
            skipped_names.append(tensor.name)
            continue
        mapped = map_tensor_name(tensor.name, name_map)
        if mapped in mapped_names:
            raise ValueError(
                f"Tensor name mapping collision: {mapped_names[mapped]!r} and "
                f"{tensor.name!r} both map to {mapped!r}."
            )
        mapped_names[mapped] = tensor.name
        shape = tuple(int(dim) for dim in tensor.shape)
        if tensor.name.endswith(("conv1.bias", "conv2.bias")):
            shape = (1, *shape)
        expected_shapes[mapped] = shape
        expected_types[mapped] = tensor.tensor_type
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            raise ValueError(
                f"Unexpected dtype {tensor.tensor_type.name} for {tensor.name}; "
                "only the source F16 and F32 weights are supported."
            )
    if len(mapped_names) != 489 or skipped_names != ["mel_filters_data"]:
        raise ValueError(
            f"Expected to map 489 weights and skip mel_filters_data; mapped {len(mapped_names)}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp.gguf", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    report_fd, report_temp_name = tempfile.mkstemp(
        prefix=f".{report_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(report_fd)
    report_temp_path = Path(report_temp_name)

    writer = None
    try:
        writer = gguf.GGUFWriter(temporary_path, "clip")
        _write_metadata(writer)
        for tensor in reader.tensors:
            if tensor.name == "mel_filters_data":
                continue
            data = tensor.data
            if tensor.name.endswith(("conv1.bias", "conv2.bias")):
                data = data.reshape((data.shape[0], 1))
            target_name = next(
                name for name, source_name in mapped_names.items() if source_name == tensor.name
            )
            writer.add_tensor(target_name, data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None

        expected_names = set(mapped_names)
        _release_reader(reader)
        reader = None
        _verify_output(temporary_path, expected_names, expected_shapes, expected_types, gguf)
        target_hash = _sha256(temporary_path)
        report = {
            "source_sha256": source_hash,
            "target_sha256": target_hash,
            "source_architecture": SOURCE_ARCHITECTURE,
            "target_architecture": "clip",
            "mapped_tensor_count": len(mapped_names),
            "skipped_tensors": skipped_names,
            "conversion_note": (
                "Converted to llama.cpp MMPROJ Qwen2-Audio layout. Preserved F16/F32 tensor "
                "values and reshaped Whisper conv1/conv2 biases to [1, channels]."
            ),
        }
        report_temp_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary_path, output_path)
        os.replace(report_temp_path, report_path)
        return report_path
    except BaseException:
        if writer is not None:
            writer.close()
        if reader is not None:
            _release_reader(reader)
        temporary_path.unlink(missing_ok=True)
        report_temp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Original Nexa projector GGUF")
    parser.add_argument(
        "--output", required=True, type=Path, help="Converted llama.cpp MMPROJ GGUF"
    )
    args = parser.parse_args(argv)
    try:
        report = convert(args.input, args.output)
    except Exception as exc:
        print(f"Projector conversion failed: {exc}", file=sys.stderr)
        return 1
    print(f"Converted projector: {args.output.resolve()}")
    print(f"Integrity report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
