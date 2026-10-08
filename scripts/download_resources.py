"""Download and verify the pinned release resources using the Python standard library."""
import argparse
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ygo_sky.resources import install_archive, sha256, verify


def read_pin(root):
    pin = json.loads((root / "configs/resources.json").read_text(encoding="utf-8"))
    if pin.get("schema") != "ygo-sky-resource-download/v1":
        raise ValueError("Unsupported resource download manifest")
    if not re.fullmatch(r"[a-f0-9]{64}", pin.get("sha256", "")):
        raise ValueError("The resource download manifest needs a SHA256 digest")
    if pin.get("filename") != "sky-striker-resources.tar.gz" or not pin.get("url", "").startswith("https://"):
        raise ValueError("Expected a pinned HTTPS resource archive")
    if not isinstance(pin.get("bytes"), int) or pin["bytes"] <= 0:
        raise ValueError("Expected a positive resource archive size")
    return pin


def check_archive(path, pin):
    if path.stat().st_size != pin["bytes"] or sha256(path) != pin["sha256"]:
        raise ValueError(f"Archive size/SHA256 mismatch: {path}")


def download(pin, cache):
    """Publish a cache entry only after checking its bytes; remove interrupted downloads."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / pin["filename"]
    if archive.exists():
        try:
            check_archive(archive, pin)
        except ValueError as error:
            raise ValueError(f"{error}. Cache preserved; move this file aside and retry.") from error
        print(f"Using verified cache: {archive}")
        return archive
    print(f"Downloading {pin['version']} resources ({pin['bytes'] / 1024**2:.1f} MiB)...", flush=True)
    request = urllib.request.Request(pin["url"], headers={"User-Agent": "GGBOY-ygoagent-resource-installer"})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=archive.name + ".", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                received = 0
                reported = 0
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    received += len(chunk)
                    if received > pin["bytes"]:
                        raise ValueError("Download exceeds the pinned archive size")
                    if received - reported >= 32 * 1024 * 1024:
                        print(f"  {received / 1024**2:.1f} / {pin['bytes'] / 1024**2:.1f} MiB", flush=True)
                        reported = received
        check_archive(temporary, pin)
        temporary.replace(archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Archive SHA256 verified: {pin['sha256']}")
    return archive


def prepare(root, cache, install=False, archive=None, verify_only=False):
    if verify_only:
        manifest, _, _ = verify(root)
        print(f"Installed resources verified: base-463m, step {manifest['global_step']}")
        return
    if install:
        destination = root / "resources"
        if destination.exists() and any(destination.iterdir()):
            try:
                manifest, _, _ = verify(root)
            except (OSError, ValueError, KeyError) as error:
                raise ValueError("Existing resources failed verification and were preserved. "
                                 "Move resources/ aside or use a fresh checkout before installing again.") from error
            print(f"Resources already installed and verified: step {manifest['global_step']}; no download needed.")
            return
    pin = read_pin(root)
    if archive is None:
        archive = download(pin, cache)
    else:
        try:
            check_archive(archive, pin)
        except ValueError as error:
            raise ValueError(f"{error}. Your local archive was preserved.") from error
        print(f"Using verified local archive: {archive}")
    if install:
        install_archive(root, str(archive), pin["sha256"])
        manifest, _, _ = verify(root)
        print(f"Installed resources verified: base-463m, step {manifest['global_step']}")
    else:
        print("Download ready. Add --install to install and verify resources/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path, help="Defaults to PROJECT/release")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true", help="Install into an empty resources/ or verify an existing installation")
    mode.add_argument("--verify-only", action="store_true", help="Check installed baseline resources without network access")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive, without network access")
    args = parser.parse_args()
    if args.verify_only and args.archive:
        parser.error("--verify-only cannot be combined with --archive")
    root = args.root.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve() if args.cache_dir else root / "release"
    archive = args.archive.expanduser().resolve() if args.archive else None
    try:
        prepare(root, cache, args.install, archive, args.verify_only)
    except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
        print(f"Resource preparation failed: {error}\nHelp: docs/常见问题.md (resource download/verification)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
