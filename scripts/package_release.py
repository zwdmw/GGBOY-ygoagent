"""Create deterministic local release candidates from an explicit source allowlist."""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

SOURCE_DIRS = ("src", "configs", "docs", "examples", "deploy", "tests", "third_party")
SOURCE_FILES = ("README.md", "LICENSE", "NOTICE.md", "pyproject.toml", ".env.example", ".gitignore")
SCRIPT_EXCLUDES = {"audit_snapshot.py", "prepare_configs.py"}
EXCLUDED_PARTS = {"__pycache__", "nnx", ".git", ".venv", "build", "dist"}
PRIVATE_PATH = re.compile(r"/ro[o]t/|[A-Za-z]:[\\/](?:Users|Cards)|auto[d]l-tmp|connect\.west[b-d]\.seetacloud")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_resource(path, target):
    # Frozen files are only read while staging; the tar contains independent bytes.
    try:
        os.link(path, target)
    except OSError:
        shutil.copyfile(path, target)


def allowed(path):
    return not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts) and path.suffix not in {".pyc", ".pyo", ".so"}


def sanitize(value, changes, field=""):
    if isinstance(value, dict):
        return {k: sanitize(v, changes, f"{field}.{k}".strip(".")) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, changes, field) for v in value]
    if isinstance(value, str) and PRIVATE_PATH.search(value):
        changes.append(field)
        return value.replace("\\", "/").rsplit("/", 1)[-1]
    return value


def inventory(folder):
    return [{"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(folder.rglob("*")) if path.is_file()]


def archive(folder, destination):
    # Sorting and normalized tar/gzip metadata make identical inputs reproducible.
    with destination.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as stream:
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                data = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(folder).as_posix())
                info.size = len(data)
                info.mode = 0o755 if path.suffix == ".sh" else 0o644
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                stream.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-resources", action="store_true", help="Verify and reuse an existing staged resource pack while refreshing source")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root / "release").resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Release output must not contain the project root")
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "staging"
    resource_tar = output / "sky-striker-resources.tar.gz"
    cached_rows = None
    if args.reuse_resources:
        cached_rows = json.loads((stage / "source/release-info/resource-files.json").read_text(encoding="utf-8"))
        expected_names = {row["path"] for row in cached_rows}
        actual_names = {p.relative_to(root).as_posix() for p in (root / "resources").rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}}
        if actual_names != expected_names:
            raise ValueError("Resource file set changed; rebuild without --reuse-resources")
        for row in cached_rows:
            if digest(root / row["path"]) != row["source_sha256"] or digest(stage / "resources" / row["path"]) != row["sha256"]:
                raise ValueError("Staged resources changed; rebuild without --reuse-resources")
        if not resource_tar.is_file():
            raise ValueError("No existing resource archive to reuse")
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if digest(resource_tar) != previous["resources_sha256"]:
            raise ValueError("Existing resource archive checksum mismatch")
    if stage.exists():
        # This path is fixed beneath the explicitly resolved release output.
        if stage.resolve().parent != output or stage.is_symlink():
            raise ValueError("Unsafe staging path")
        if args.reuse_resources:
            shutil.rmtree(stage / "source")
        else:
            shutil.rmtree(stage)
    source = stage / "source"
    resources = stage / "resources"
    source.mkdir(parents=True)
    resources.mkdir(exist_ok=args.reuse_resources)
    chosen = [root / name for name in SOURCE_FILES]
    for name in SOURCE_DIRS:
        chosen.extend(p for p in (root / name).rglob("*") if p.is_file() and allowed(p.relative_to(root)))
    chosen.extend(p for p in (root / "scripts").rglob("*")
                  if p.is_file() and allowed(p.relative_to(root)) and p.name not in SCRIPT_EXCLUDES)
    chosen.extend(root / "native" / name for name in ("build.sh", "build_cached.py", "record_runtime.py", "xmake.lua"))
    for name in ("ygoenv", "bridge/ygoenv", "repo"):
        chosen.extend(p for p in (root / "native" / name).rglob("*") if p.is_file() and allowed(p.relative_to(root)))
    for path in sorted(set(chosen)):
        target = source / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    resource_rows = cached_rows or []
    for path in ([] if args.reuse_resources else sorted((root / "resources").rglob("*"))):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        target = resources / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        changes = []
        if path.suffix in {".json", ".jsonl"}:
            text = path.read_text(encoding="utf-8")
            if PRIVATE_PATH.search(text):
                if path.suffix == ".json":
                    save(target, sanitize(json.loads(text), changes))
                else:
                    rows = [sanitize(json.loads(line), changes) for line in text.splitlines() if line.strip()]
                    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            else:
                stage_resource(path, target)
        else:
            stage_resource(path, target)
        resource_rows.append({"path": path.relative_to(root).as_posix(), "bytes": target.stat().st_size,
                              "source_sha256": digest(path), "sha256": digest(target), "sanitized_fields": sorted(set(changes))})
    for path in (source / "configs/model").glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for item in manifest["artifacts"].values():
            item["sha256"] = digest(resources / item["path"])
        save(path, manifest)
    info = source / "release-info"
    save(info / "resource-files.json", resource_rows)
    save(info / "source-files.json", inventory(source))
    # Review text only; frozen binary resources keep their original bytes.
    violations = []
    for folder in (source, resources):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".jsonl", ".md", ".txt", ".conf", ".yaml", ".lua", ".ydk", ".sh", ".toml", ".h", ".cpp"}:
                if PRIVATE_PATH.search(path.read_text(encoding="utf-8", errors="replace")):
                    violations.append(path.relative_to(stage).as_posix())
    if violations:
        raise ValueError("Internal paths remain: " + ", ".join(violations[:20]))
    source_tar = output / "sky-striker-source.tar.gz"
    archive(source, source_tar)
    if not args.reuse_resources:
        archive(resources, resource_tar)
    result = {"schema": "ygo-sky-release/v1", "version": "0.1.0", "status": "local_candidate_licensing_pending",
              "publicly_published": False, "source_archive": source_tar.name, "source_sha256": digest(source_tar),
              "source_bytes": source_tar.stat().st_size, "resources_archive": resource_tar.name,
              "resources_sha256": digest(resource_tar), "resources_bytes": resource_tar.stat().st_size,
              "source_file_count": len(inventory(source)), "resource_file_count": len(resource_rows),
              "sanitized_resource_files": sum(bool(row["sanitized_fields"]) for row in resource_rows),
              "private_text_path_scan": "passed", "acceptance": "acceptance.json",
              "notes": ["No SSH credentials, raw snapshots, logs, virtual environments or build caches are included.",
                        "Weights and frozen binary resources retain their original bytes.",
                        "Source sidecar hashes and sanitized sidecar hashes are both preserved in release-info/resource-files.json.",
                        "Original integration code is MIT licensed; third-party code retains its own terms. Resource and network licensing still has unresolved items; see NOTICE.md."]}
    save(output / "manifest.json", result)
    (output / "SHA256SUMS").write_text(f"{result['source_sha256']}  {source_tar.name}\n{result['resources_sha256']}  {resource_tar.name}\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
