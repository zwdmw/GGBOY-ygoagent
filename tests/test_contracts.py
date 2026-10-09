import ast
import hashlib
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

    def test_specialist_recipe_reaches_published_step(self):
        root = project_root()
        config = read_json(root / "configs/train/mirror-specialist.json")
        manifest = read_json(root / "configs/model/specialist-464m.json")
        batch = (config["args"]["local_num_envs"] * config["args"]["num_actor_threads"]
                 * config["args"]["num_steps"] * len(config["args"]["actor_device_ids"]))
        self.assertEqual(config["args"]["total_timesteps"] % batch, 0)
        self.assertEqual(
            read_json(root / "configs/model/base-463m.json")["global_step"]
            + config["args"]["total_timesteps"],
            manifest["global_step"],
        )

    def test_295m_recipe_reaches_batch_aligned_target(self):
        root = project_root()
        config = read_json(root / "configs/train/to-295m.json")
        initial = read_json(root / "configs/model/initial-259m.json")
        batch = (config["args"]["local_num_envs"] * config["args"]["num_actor_threads"]
                 * config["args"]["num_steps"] * len(config["args"]["actor_device_ids"]))
        self.assertEqual(config["args"]["total_timesteps"] % batch, 0)
        self.assertEqual(initial["global_step"] + config["args"]["total_timesteps"], 295004160)

    def test_legacy_lineage_archive_matches_its_manifest(self):
        root = project_root()
        archive = root / "third_party/legacy-lineage"
        manifest = read_json(archive / "docs/hash-manifest.json")
        rows = manifest["archived_files"]
        self.assertTrue(rows)
        self.assertEqual(manifest["schema"], "ygo-sky-legacy-lineage/v1")
        for row in rows:
            with self.subTest(path=row["path"]):
                path = contained(archive.resolve(), row["path"])
                self.assertTrue(path.is_file(), f"missing archived file: {row['path']}")
                data = path.read_bytes()
                self.assertEqual(len(data), row["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
        # Every file under the archive except the manifest itself must be listed.
        listed = {row["path"] for row in rows}
        present = {path.relative_to(archive).as_posix() for path in archive.rglob("*")
                   if path.is_file() and path.name != "hash-manifest.json"}
        self.assertEqual(present - listed, set())
        # The two expert snapshots are byte-identical copies, recorded as such.
        expert = {row["file"]: row["sha256"] for row in manifest["trainer_shas"]}
        self.assertEqual(expert["cleanba.expert0919.py"], expert["cleanba.corrected0920.py"])

    def test_training_recipes_use_supported_arguments(self):
        root = project_root()
        for path in (root / "configs/train").glob("*.json"):
            config = read_json(path)
            module = root / "src/ygo_sky/training" / (config["trainer"] + ".py")
            tree = ast.parse(module.read_text(encoding="utf-8"))
            args = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
            supported = {node.target.id for node in args.body
                         if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
            with self.subTest(recipe=path.name):
                self.assertEqual(set(config["args"]) - supported, set())


if __name__ == "__main__":
    unittest.main()
