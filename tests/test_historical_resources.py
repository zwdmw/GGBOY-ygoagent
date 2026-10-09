"""Historical native archives install verified files and preserve different existing modules."""
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_historical_resources import install


class HistoricalInstallTests(unittest.TestCase):
    def make_archive(self, root, members, expected=None):
        archive = root / "historical.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, data in members:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
        expected = expected or members
        pin = {"bytes": archive.stat().st_size,
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "artifacts": [{"path": name, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()} for name, data in expected]}
        return archive, pin

    def test_verified_install_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "native/history/v11/module.so"
            archive, pin = self.make_archive(root, [(name, b"frozen native fixture")])
            install(root, archive, pin)
            install(root, archive, pin)
            self.assertEqual((root / name).read_bytes(), b"frozen native fixture")

    def test_unlisted_path_is_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [("native/history/v11/module.so", b"expected")]
            archive, pin = self.make_archive(root, [("../outside", b"unexpected")], expected)
            with self.assertRaisesRegex(ValueError, "Unexpected historical archive member"):
                install(root, archive, pin)
            self.assertFalse((root / "native").exists())

    def test_missing_module_does_not_publish_partial_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = [("native/history/v11/module.so", b"one")]
            expected = members + [("native/history/tribute-fix/module.so", b"two")]
            archive, pin = self.make_archive(root, members, expected)
            with self.assertRaisesRegex(ValueError, "missing a pinned module"):
                install(root, archive, pin)
            self.assertFalse((root / "native").exists())

    def test_different_existing_module_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "native/history/v11/module.so"
            destination = root / name
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"another experiment")
            archive, pin = self.make_archive(root, [(name, b"published snapshot")])
            with self.assertRaisesRegex(ValueError, "different identity"):
                install(root, archive, pin)
            self.assertEqual(destination.read_bytes(), b"another experiment")


if __name__ == "__main__":
    unittest.main()
