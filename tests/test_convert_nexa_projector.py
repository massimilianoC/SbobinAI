import importlib.util
import unittest
from pathlib import Path

try:
    import gguf
except ImportError:  # pragma: no cover - the optional converter dependency
    gguf = None


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert-nexa-projector.py"
SPEC = importlib.util.spec_from_file_location("convert_nexa_projector", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


@unittest.skipIf(gguf is None, "optional llama.cpp gguf package is not installed")
class ProjectorNameMappingTests(unittest.TestCase):
    def test_maps_audio_encoder_and_projector_tensors(self):
        name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, 32)
        self.assertEqual(
            converter.map_tensor_name("audio_tower.conv1.weight", name_map),
            "a.conv1d.1.weight",
        )
        self.assertEqual(
            converter.map_tensor_name("multi_modal_projector.linear.weight", name_map),
            "mm.a.fc.weight",
        )

    def test_unknown_tensor_name_fails_closed(self):
        class EmptyMap:
            def get_name(self, name, try_suffixes=()):
                return None

        with self.assertRaisesRegex(ValueError, "No llama.cpp MMPROJ tensor mapping"):
            converter.map_tensor_name("unmapped.tensor", EmptyMap())


if __name__ == "__main__":
    unittest.main()
