import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from download_resources import download, prepare


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.body = b"small archive fixture"
        self.pin = {
            "version": "fixture", "filename": "sky-striker-resources.tar.gz",
            "url": "https://example.org/resources.tar.gz", "bytes": len(self.body),
            "sha256": hashlib.sha256(self.body).hexdigest(),
        }

    def test_valid_download_is_cached_and_reused_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            with patch("urllib.request.urlopen", return_value=io.BytesIO(self.body)) as request:
                archive = download(self.pin, cache)
                self.assertEqual(archive.read_bytes(), self.body)
                self.assertEqual(download(self.pin, cache), archive)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(list(cache.iterdir()), [archive])

    def test_wrong_hash_is_never_published(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            pin = {**self.pin, "sha256": "0" * 64}
            with patch("urllib.request.urlopen", return_value=io.BytesIO(self.body)):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    download(pin, cache)
            self.assertEqual(list(cache.iterdir()), [])

    def test_interrupted_download_cleans_partial_file(self):
        class InterruptedResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise urllib.error.URLError("fixture disconnect")
                return super().read(4)

        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            with patch("urllib.request.urlopen", return_value=InterruptedResponse(self.body)):
                with self.assertRaises(urllib.error.URLError):
                    download(self.pin, cache)
            self.assertEqual(list(cache.iterdir()), [])

    def test_bad_cache_is_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            archive = cache / self.pin["filename"]
            archive.write_bytes(b"bad cache")
            with patch("urllib.request.urlopen") as request:
                with self.assertRaises(ValueError):
                    download(self.pin, cache)
                request.assert_not_called()
            self.assertEqual(archive.read_bytes(), b"bad cache")

    def test_existing_invalid_resources_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            resources = root / "resources"
            resources.mkdir()
            marker = resources / "keep.txt"
            marker.write_bytes(b"keep this file")
            with patch("urllib.request.urlopen") as request:
                with self.assertRaisesRegex(ValueError, "preserved"):
                    prepare(root, root / "release", install=True)
                request.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"keep this file")

    def test_offline_install_verifies_resources_and_repeated_run_skips_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = b"fixture model bytes"
            checkpoint_hash = hashlib.sha256(checkpoint).hexdigest()
            sidecar = json.dumps({"schema": "ygo-training-checkpoint/v2", "architecture": "decision-v1",
                                  "global_step": 463001600, "sha256": checkpoint_hash}).encode()
            (root / "configs/model").mkdir(parents=True)
            (root / "configs/model/base-463m.json").write_text(json.dumps({
                "schema": "ygo-sky-model/v1", "global_step": 463001600,
                "artifacts": {
                    "checkpoint": {"path": "resources/model.bin", "sha256": checkpoint_hash},
                    "sidecar": {"path": "resources/model.bin.json", "sha256": hashlib.sha256(sidecar).hexdigest()},
                },
            }), encoding="utf-8")
            archive = root / self.pin["filename"]
            with tarfile.open(archive, "w:gz") as stream:
                for name, body in (("resources/model.bin", checkpoint), ("resources/model.bin.json", sidecar)):
                    member = tarfile.TarInfo(name)
                    member.size = len(body)
                    stream.addfile(member, io.BytesIO(body))
            pin = {**self.pin, "schema": "ygo-sky-resource-download/v1", "bytes": archive.stat().st_size,
                   "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            (root / "configs/resources.json").write_text(json.dumps(pin), encoding="utf-8")
            with patch("urllib.request.urlopen") as request:
                prepare(root, root / "release", install=True, archive=archive)
                prepare(root, root / "release", install=True)
                prepare(root, root / "release", verify_only=True)
                request.assert_not_called()
            self.assertEqual((root / "resources/model.bin").read_bytes(), checkpoint)
            self.assertFalse((root / "resources.installing").exists())


if __name__ == "__main__":
    unittest.main()
