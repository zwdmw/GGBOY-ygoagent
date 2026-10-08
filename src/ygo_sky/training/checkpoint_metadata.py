"""Publish model bytes last, so consumers never see a partial checkpoint."""
import hashlib
import json
import os
from pathlib import Path


def read_metadata(path, verify=False):
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if not sidecar.exists():
        return None
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if verify:
        digest = metadata.get("sha256")
        if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"checkpoint metadata checksum mismatch: {path}")
    return metadata


def checkpoint_step(path):
    metadata = read_metadata(path)
    if metadata and "global_step" in metadata:
        return int(metadata["global_step"])
    name = Path(path).stem.rsplit("_", 1)[-1]
    if name.endswith("M") and name[:-1].isdigit():
        return int(name[:-1]) * 1048576
    raise ValueError(f"checkpoint has no exact step: {path}")


def publish_checkpoint(path, data, metadata):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    sidecar = Path(str(path) + ".json")
    sidecar_tmp = Path(str(sidecar) + ".tmp")
    from ygo_sky.paths import project_root, read_json
    runtime = read_json(project_root() / "native/runtime.json")
    metadata = {**metadata, "sha256": hashlib.sha256(data).hexdigest(),
                "observation_revision": runtime["observation_revision"],
                "native_module_sha256": runtime["sha256"], "optimizer_restored": False}
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    with sidecar_tmp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(sidecar_tmp, sidecar)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
