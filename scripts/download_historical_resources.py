"""Download and install the pinned historical Linux native modules."""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from download_resources import check_archive, download


def read_pin(root):
    pin = json.loads((root / "configs/historical-resources.json").read_text(encoding="utf-8"))
    if pin.get("schema") != "ygo-sky-historical-resources/v1":
        raise ValueError("Expected a historical native resource manifest")
    if pin.get("filename") != "historical-native-resources.tar.gz" or not pin.get("url", "").startswith("https://"):
        raise ValueError("Expected the pinned HTTPS historical native archive")
    if not re.fullmatch(r"[a-f0-9]{64}", pin.get("sha256", "")) or not isinstance(pin.get("bytes"), int) or pin["bytes"] <= 0:
        raise ValueError("Expected a SHA256 and positive archive size")
    for row in pin["artifacts"]:
        parts = PurePosixPath(row["path"]).parts
        if len(parts) != 4 or parts[:2] != ("native", "history") or parts[2] not in ("v11", "tribute-fix") or not parts[-1].endswith(".so"):
            raise ValueError("Historical modules must install beneath native/history/<variant>/")
        if not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", "")) or row.get("bytes", 0) <= 0:
            raise ValueError("Expected a SHA256 and size for every historical module")
    if not pin["artifacts"] or len({r["path"] for r in pin["artifacts"]}) != len(pin["artifacts"]):
        raise ValueError("Historical module paths must be unique")
    return pin


def install(root, archive, pin):
    """Verify every member before publishing modules; preserve existing different files."""
    check_archive(archive, pin)
    rows = {row["path"]: row for row in pin["artifacts"]}
    root = root.resolve()
    staged = {}
    with tempfile.TemporaryDirectory(prefix="ygo-history-") as directory:
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream:
                if member.name not in rows or not member.isfile() or member.name in staged:
                    raise ValueError(f"Unexpected historical archive member: {member.name}")
                row = rows[member.name]
                if member.size != row["bytes"]:
                    raise ValueError(f"Historical module size mismatch: {member.name}")
                data = stream.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"Historical module SHA256 mismatch: {member.name}")
                path = Path(directory) / f"module-{len(staged)}"
                path.write_bytes(data)
                destination = (root / member.name).resolve()
                destination.relative_to(root)
                destination.relative_to((root / "native/history").resolve())
                if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError(f"Existing module has a different identity and was preserved: {member.name}")
                staged[member.name] = (path, destination)
        if set(staged) != set(rows):
            raise ValueError("Historical archive is missing a pinned module")
        for name, (path, destination) in staged.items():
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            print(f"Verified historical module: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--archive", type=Path, help="Use an existing offline archive")
    parser.add_argument("--install", action="store_true", help="Install into native/history/")
    args = parser.parse_args()
    pin = read_pin(args.root)
    archive = args.archive or download(pin, args.cache_dir or args.root / "release")
    check_archive(archive, pin)
    if args.install:
        install(args.root, archive, pin)
    else:
        print(f"Historical archive verified: {archive}; add --install to install native/history/")


if __name__ == "__main__":
    main()
