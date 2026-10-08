import hashlib
import json
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .paths import contained, read_json, write_json


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, model="base-463m"):
    manifest = read_json(root / "configs/model" / f"{model}.json")
    if manifest.get("schema") != "ygo-sky-model/v1":
        raise ValueError("Unsupported model manifest")
    rows = []
    for name, item in manifest["artifacts"].items():
        path = contained(root, item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"{name}: SHA256 mismatch for {path}")
        rows.append({"name": name, "path": item["path"], "sha256": actual})
    sidecar = read_json(contained(root, manifest["artifacts"]["sidecar"]["path"]))
    if sidecar.get("schema") != "ygo-training-checkpoint/v2" or sidecar.get("architecture") != "decision-v1":
        raise ValueError("Unsupported checkpoint schema/architecture")
    if sidecar["global_step"] != manifest["global_step"]:
        raise ValueError("Checkpoint step differs from model manifest")
    if sidecar["sha256"] != manifest["artifacts"]["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint sidecar hash identity mismatch")
    return manifest, sidecar, rows


def install_native(root):
    if sys.version_info[:2] != (3, 11) or platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("Reference native binary requires Linux x86_64 and CPython 3.11; build native/ on other targets")
    manifest = read_json(root / "configs/model/base-463m.json")
    item = manifest["artifacts"]["native_reference"]
    source = contained(root, item["path"])
    if sha256(source) != item["sha256"]:
        raise ValueError("Reference native binary checksum mismatch")
    target = root / "native/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-311-x86_64-linux-gnu.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    write_json(root / "native/runtime.json", {"schema": "ygo-sky-runtime/v1", "kind": "reference",
               "module": str(target.relative_to(root)), "sha256": sha256(target),
               "observation_revision": manifest["observation_revision"]})
    return target


def register(root, checkpoint, name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise ValueError("Model name requires lowercase letters, digits and hyphens")
    destination = root / "configs/model" / (name + ".json")
    if destination.exists():
        raise FileExistsError(destination)
    sidecar_path = Path(str(checkpoint) + ".json")
    metadata = read_json(sidecar_path)
    if metadata.get("schema") != "ygo-training-checkpoint/v2" or metadata.get("architecture") != "decision-v1":
        raise ValueError("Unsupported checkpoint metadata")
    if metadata.get("sha256") != sha256(checkpoint):
        raise ValueError("Checkpoint checksum mismatch")
    manifest = read_json(root / "configs/model/base-463m.json")
    source_metadata = read_json(contained(root, manifest["artifacts"]["sidecar"]["path"]))
    for key in ("model_args", "semantic_shape", "bfloat16", "switch"):
        if metadata[key] != source_metadata[key]:
            raise ValueError(f"New model differs from the pinned contract: {key}")
    target = root / "resources/models" / (name + ".flax_model")
    if target.exists() or Path(str(target) + ".json").exists():
        raise FileExistsError(target)
    shutil.copy2(checkpoint, target)
    shutil.copy2(sidecar_path, Path(str(target) + ".json"))
    manifest.update(name=name, global_step=metadata["global_step"])
    for key, path in (("checkpoint", target), ("sidecar", Path(str(target) + ".json"))):
        manifest["artifacts"][key] = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
    write_json(destination, manifest)
    return {"model": name, "step": metadata["global_step"], "sha256": metadata["sha256"]}


def install_archive(root, archive, expected_sha256):
    """Install a hash-pinned resource pack atomically into an empty resource tree."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("Resource archive requires an explicit SHA256")
    with tempfile.TemporaryDirectory(prefix="ygo-resources-") as temporary:
        folder = Path(temporary)
        if archive.startswith(("https://", "http://")):
            source = folder / "download.tar.gz"
            with urllib.request.urlopen(archive, timeout=120) as response, source.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        else:
            source = Path(archive).expanduser().resolve()
        if sha256(source) != expected_sha256.lower():
            raise ValueError("Resource archive SHA256 mismatch")
        extracted = folder / "extracted"
        extracted.mkdir()
        with tarfile.open(source) as stream:
            members = stream.getmembers()
            for member in members:
                parts = Path(member.name).parts
                if member.issym() or member.islnk() or not parts or parts[0] != "resources":
                    raise ValueError("Resource pack must contain only regular files/directories under resources/")
            stream.extractall(extracted, filter="data")
        destination = root / "resources"
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Resource destination must be empty; existing resources are preserved")
        staged = root / "resources.installing"
        if staged.exists():
            raise ValueError("A previous resource installation is still staged")
        shutil.copytree(extracted / "resources", staged)
        if destination.exists():
            destination.rmdir()
        staged.replace(destination)
    return {"installed": str(destination), "archive_sha256": expected_sha256.lower()}
