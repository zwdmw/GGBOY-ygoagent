"""Audit content overlap without inventing absent historical split manifests."""
import hashlib
import argparse
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=root / "runs/deck-audit.json")
args = parser.parse_args()


def fingerprint(path):
    sections = {"main": [], "extra": [], "side": []}
    section = "main"
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line == "#extra":
            section = "extra"
        elif line == "!side":
            section = "side"
        elif line.isdigit():
            sections[section].append(int(line))
    content = json.dumps({key: sorted(value) for key, value in sections.items()}, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


training = root / "resources/game/assets/deck/expert-sky-20260919/train"
rows = [{"file": path.relative_to(root).as_posix(), "content_sha256": fingerprint(path),
         "cluster": path.stem.split("__", 1)[0]} for path in sorted(training.glob("*.ydk"))]
target = fingerprint(root / "resources/decks/sky-striker.ydk")
counts = Counter(row["content_sha256"] for row in rows)
report = {"schema": "ygo-sky-deck-audit/v1", "training_decks": len(rows),
          "unique_contents": len(counts), "duplicate_contents": {key: value for key, value in counts.items() if value > 1},
          "eval_target_sha256": target, "target_overlap": [row for row in rows if row["content_sha256"] == target],
          "interpretation": "Mirror evaluation uses the trained target deck; it is not a held-out deck generalization test.",
          "training_manifest_available": (training.parent / "manifests/train.jsonl").is_file(),
          "historical_validation_test_split_available": False, "training": rows}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"training_decks": len(rows), "unique_contents": len(counts), "target_overlap": len(report["target_overlap"])}))
