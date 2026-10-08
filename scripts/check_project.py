"""Run contributor checks without downloading weights or importing JAX."""
import argparse
import json
import re
import sys
import tomllib
import unittest
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
for directory in ("src", "scripts", "tests"):
    sys.path.insert(0, str(ROOT / directory))


def without_fences(text):
    return re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text, flags=re.M | re.S)


def anchors(text):
    found = set()
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", without_fences(text), re.M):
        # Match the plain headings used by the project; retain Unicode letters.
        slug = re.sub(r"[^\w\-\s]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        found.add(slug if not count else f"{slug}-{count}")
        counts[slug] = count + 1
    return found


def check_docs():
    docs = [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md"), *(ROOT / ".github").rglob("*.md")]
    errors = []
    for path in docs:
        text = without_fences(path.read_text(encoding="utf-8"))
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip().strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            linked = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not linked.is_relative_to(ROOT) or not linked.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing local target {target}")
            elif url.fragment and linked.suffix == ".md":
                if unquote(url.fragment) not in anchors(linked.read_text(encoding="utf-8")):
                    errors.append(f"{path.relative_to(ROOT)}: missing heading {target}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Local Markdown links checked: {len(docs)} files")


def check_formats():
    import yaml

    configs = [*(ROOT / "configs").rglob("*.json"), *(ROOT / "third_party").rglob("*.json")]
    for path in configs:
        json.loads(path.read_text(encoding="utf-8-sig"))
    yaml_files = list((ROOT / ".github").rglob("*.yml"))
    for path in yaml_files:
        # BaseLoader preserves GitHub's `on` key instead of YAML 1.1 boolean coercion.
        yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    from download_resources import read_pin
    pin = read_pin(ROOT)
    released = json.loads((ROOT / "release-info/manifest.json").read_text(encoding="utf-8")) if (ROOT / "release-info/manifest.json").exists() else None
    if released and pin["version"] == released["version"]:
        if pin["sha256"] != released["resources_sha256"] or pin["bytes"] != released["resources_bytes"]:
            raise ValueError("Download pin differs from the recorded release resource identity")
    print(f"Formats checked: {len(configs)} JSON, {len(yaml_files)} YAML, pyproject.toml and resource download pin")


def check_syntax():
    count = 0
    for folder in ("src", "scripts", "tests", "examples"):
        for path in (ROOT / folder).rglob("*.py"):
            compile(path.read_bytes(), str(path), "exec")
            count += 1
    print(f"Python syntax checked: {count} files")


def check_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in (
        "test_contracts.IdentityTests.test_corruption_is_rejected",
        "test_contracts.IdentityTests.test_path_escape_is_rejected",
        "test_contracts.IdentityTests.test_nested_boolean_flags",
        "test_wire",
        "test_download_resources",
    ))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise ValueError("Resource-free tests failed")


def check_wheel(folder):
    wheels = list(folder.glob("sky_striker_expert-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one project wheel in {folder}, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as stream:
        notices = {Path(name).name: stream.read(name) for name in stream.namelist()
                   if ".dist-info/" in name and "/licenses/" in name and not name.endswith("/")}
        for relative in ("LICENSE", "NOTICE.md", "third_party/ygo-agent/YGO-AGENT-LICENSE.txt", "third_party/ygo-agent/APACHE-2.0.txt"):
            if notices.get(Path(relative).name) != (ROOT / relative).read_bytes():
                raise ValueError(f"Wheel missing or altered license/notice: {relative}")
    print(f"Wheel notices verified: {wheels[0].name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, help="Also verify licenses in a locally built wheel")
    parser.add_argument("--wheel-only", action="store_true", help="Only check the wheel")
    args = parser.parse_args()
    if args.wheel_only and not args.wheel_dir:
        parser.error("--wheel-only requires --wheel-dir")
    try:
        if not args.wheel_only:
            check_docs()
            check_formats()
            check_syntax()
            check_tests()
        if args.wheel_dir:
            check_wheel(args.wheel_dir)
    except ImportError as error:
        print(f"Check dependency missing: {error}. Install PyYAML for project checks.", file=sys.stderr)
        return 1
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Check failed: {error}", file=sys.stderr)
        return 1
    print("All requested lightweight checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
