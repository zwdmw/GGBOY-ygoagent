import shutil
import tempfile
import unittest
from pathlib import Path

from ygo_sky.paths import contained, project_root, read_json, write_json
from ygo_sky.resources import verify
from ygo_sky.runner import flags, training


class IdentityTests(unittest.TestCase):
    def test_models_have_distinct_identity(self):
        root = project_root()
        base, _, _ = verify(root, "base-463m")
        specialist, _, _ = verify(root, "specialist-464m")
        self.assertEqual(base["global_step"], 463001600)
        self.assertNotEqual(base["artifacts"]["checkpoint"]["sha256"], specialist["artifacts"]["checkpoint"]["sha256"])

    def test_corruption_is_rejected(self):
        root = project_root()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            manifest = read_json(root / "configs/model/base-463m.json")
            manifest["artifacts"] = {"checkpoint": {"path": "bad.bin", "sha256": "0" * 64}}
            write_json(target / "configs/model/base-463m.json", manifest)
            (target / "bad.bin").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify(target)

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            contained(project_root(), "../outside")

    def test_training_paths_are_independent_and_additional_steps(self):
        root = project_root()
        result = training(root, root / "configs/train/smoke.json", root / "runs/test-preview", "cpu", True)
        args = result["command"]
        self.assertEqual(args[args.index("--tb-offset") + 1], "463001600")
        self.assertEqual(args[args.index("--total-timesteps") + 1], "32")
        self.assertEqual(Path(args[args.index("--deck") + 1]), root / "resources/game/assets/deck/expert-sky-20260919/train")
        self.assertFalse((root / "runs/test-preview").exists())

    def test_nested_boolean_flags(self):
        self.assertEqual(flags({"m1.card_mask": False, "concurrency": False}), ["--m1.no-card-mask", "--no-concurrency"])


if __name__ == "__main__":
    unittest.main()
