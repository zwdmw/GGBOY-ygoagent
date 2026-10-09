"""Verify the frozen historical source and deck archive with the standard library."""
import argparse
import hashlib
import json
from pathlib import Path


def verify_archive(root):
    archive = root / "third_party/legacy-lineage"
    manifest = json.loads((archive / "docs/hash-manifest.json").read_text(encoding="utf-8"))
    rows = manifest["archived_files"]
    for row in rows:
        path = (archive / row["path"]).resolve()
        path.relative_to(archive.resolve())
        if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Historical archive identity mismatch: {row['path']}")
    actual = {path.relative_to(archive).as_posix() for path in archive.rglob("*")
              if path.is_file() and path.name != "hash-manifest.json" and "__pycache__" not in path.parts}
    if actual != {row["path"] for row in rows}:
        raise ValueError("Historical archive file set differs from its manifest")
    counts = {split: len(list((archive / "assets/deck/largepool-v3" / split).glob("*.ydk")))
              for split in ("train", "validation", "test", "official_validation")}
    if counts != {"train": 11459, "validation": 1433, "test": 1432, "official_validation": 15}:
        raise ValueError("Historical deck split counts differ from the recorded dataset")
    return {"verified_files": len(rows), "deck_splits": counts, "status": "archive hashes verified"}


def verify_native(root):
    pin = json.loads((root / "configs/historical-resources.json").read_text(encoding="utf-8"))
    for row in pin["artifacts"]:
        path = (root / row["path"]).resolve()
        path.relative_to((root / "native/history").resolve())
        if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Historical native identity mismatch: {row['path']}")
    return {"verified_modules": len(pin["artifacts"]), "platform": pin["platform"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--native", action="store_true", help="Also verify the installed historical native modules")
    args = parser.parse_args()
    result = verify_archive(args.root.resolve())
    if args.native:
        result["native"] = verify_native(args.root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
