from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SECTION_MAIN = "main"
SECTION_EXTRA = "extra"
SECTION_SIDE = "side"
SECTION_MARKERS = {
    "#main": SECTION_MAIN,
    "#extra": SECTION_EXTRA,
    "!side": SECTION_SIDE,
}


@dataclass(frozen=True)
class DeckRecord:
    source: Path
    source_relative: str
    format_name: str
    year: str
    family: str
    main: tuple[int, ...]
    extra: tuple[int, ...]
    side: tuple[int, ...]
    core_sha256: str
    full_sha256: str
    output_name: str
    is_anchor: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build flat, deterministic train/validation YDK pools from the "
            "large compatible deck archive."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--anchor-deck", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cards-cdb", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument(
        "--token-deck",
        type=Path,
        help="Optional _tokens.ydk copied into both output pools.",
    )
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument(
        "--max-variants-per-family",
        type=int,
        default=30,
        help=(
            "Maximum gameplay-distinct variants retained per normalized "
            "family. Zero keeps every variant."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--anchor-name",
        default="03_Sky_Striker_706646",
        help="Exact stem used by the native anchor-deck scheduler.",
    )
    return parser.parse_args()


def parse_ydk(path: Path) -> dict[str, tuple[int, ...]]:
    sections: dict[str, list[int]] = {
        SECTION_MAIN: [],
        SECTION_EXTRA: [],
        SECTION_SIDE: [],
    }
    current: str | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        marker = SECTION_MARKERS.get(line.lower())
        if marker is not None:
            current = marker
            continue
        if not line or line.startswith("#"):
            continue
        if current is None:
            raise ValueError(
                f"{path}:{line_number}: card appears before a section marker"
            )
        if not line.isdigit():
            raise ValueError(
                f"{path}:{line_number}: malformed card code {line!r}"
            )
        sections[current].append(int(line))
    if not sections[SECTION_MAIN]:
        raise ValueError(f"{path}: empty main deck")
    return {name: tuple(values) for name, values in sections.items()}


def multiset_payload(*sections: Iterable[int]) -> bytes:
    payload = [
        sorted(Counter(section).items())
        for section in sections
    ]
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_payload(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_family(stem: str) -> str:
    value = stem.lower().replace("_", "-")
    value = re.sub(r"^\d+-+", "", value)
    value = re.sub(r"-+\d+$", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "unknown"


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return value or "unknown"


def read_code_list(path: Path) -> set[int]:
    codes: set[int] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = raw_line.split()
        if not fields:
            continue
        try:
            codes.add(int(fields[0]))
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid code-list row"
            ) from exc
    return codes


def read_cdb_codes(path: Path) -> set[int]:
    with sqlite3.connect(path) as connection:
        return {
            int(row[0])
            for row in connection.execute("SELECT id FROM datas")
        }


def make_record(
    source: Path,
    source_root: Path,
    *,
    is_anchor: bool,
    anchor_name: str,
) -> DeckRecord:
    sections = parse_ydk(source)
    if is_anchor:
        relative = source.name
        format_name = "anchor"
        year = "fixed"
        family = normalize_family(anchor_name)
        output_name = f"{anchor_name}.ydk"
    else:
        relative_path = source.relative_to(source_root)
        relative = relative_path.as_posix()
        parts = relative_path.parts
        format_name = parts[0] if len(parts) >= 1 else "unknown"
        year = parts[1] if len(parts) >= 2 else "unknown"
        family = normalize_family(source.stem)
        output_name = "__".join(
            [
                safe_component(format_name),
                safe_component(year),
                safe_component(source.stem),
            ]
        ) + ".ydk"

    core_payload = multiset_payload(
        sections[SECTION_MAIN],
        sections[SECTION_EXTRA],
    )
    full_payload = multiset_payload(
        sections[SECTION_MAIN],
        sections[SECTION_EXTRA],
        sections[SECTION_SIDE],
    )
    return DeckRecord(
        source=source,
        source_relative=relative,
        format_name=format_name,
        year=year,
        family=family,
        main=sections[SECTION_MAIN],
        extra=sections[SECTION_EXTRA],
        side=sections[SECTION_SIDE],
        core_sha256=sha256_payload(core_payload),
        full_sha256=sha256_payload(full_payload),
        output_name=output_name,
        is_anchor=is_anchor,
    )


def deterministic_score(seed: int, *values: str) -> str:
    payload = "\0".join([str(seed), *values]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def choose_family_variants(
    records: list[DeckRecord],
    maximum: int,
    seed: int,
) -> tuple[list[DeckRecord], list[DeckRecord]]:
    if maximum <= 0 or len(records) <= maximum:
        return records, []
    ordered = sorted(
        records,
        key=lambda item: deterministic_score(
            seed,
            item.family,
            item.core_sha256,
            item.source_relative,
        ),
    )
    return ordered[:maximum], ordered[maximum:]


def assign_families(
    family_records: dict[str, list[DeckRecord]],
    validation_percent: int,
    seed: int,
) -> dict[str, str]:
    if not 0 < validation_percent < 100:
        raise ValueError("--validation-percent must be between 1 and 99")

    total = sum(len(records) for records in family_records.values())
    target_validation = round(total * validation_percent / 100)
    candidates = [
        (family, records)
        for family, records in family_records.items()
    ]
    candidates.sort(
        key=lambda item: (
            deterministic_score(seed, "split", item[0]),
            item[0],
        )
    )

    assignments = {
        family: "train"
        for family in family_records
    }
    validation_count = 0
    for family, records in candidates:
        size = len(records)
        current_error = abs(target_validation - validation_count)
        next_error = abs(target_validation - (validation_count + size))
        if validation_count < target_validation and (
            next_error <= current_error or validation_count == 0
        ):
            assignments[family] = "validation"
            validation_count += size
    return assignments


def write_deck(record: DeckRecord, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#main",
        *(str(code) for code in record.main),
        "#extra",
        *(str(code) for code in record.extra),
        "!side",
        *(str(code) for code in record.side),
        "",
    ]
    destination.write_bytes("\n".join(lines).encode("ascii"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for path in (
        args.source_root,
        args.anchor_deck,
        args.cards_cdb,
        args.code_list,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.token_deck is not None and not args.token_deck.is_file():
        raise FileNotFoundError(args.token_deck)

    source_files = sorted(args.source_root.rglob("*.ydk"))
    if not source_files:
        raise ValueError(f"no YDK files under {args.source_root}")

    anchor = make_record(
        args.anchor_deck,
        args.source_root,
        is_anchor=True,
        anchor_name=args.anchor_name,
    )
    records = [
        make_record(
            path,
            args.source_root,
            is_anchor=False,
            anchor_name=args.anchor_name,
        )
        for path in source_files
    ]

    cdb_codes = read_cdb_codes(args.cards_cdb)
    code_list_codes = read_code_list(args.code_list)
    used_codes = set(anchor.main + anchor.extra + anchor.side)
    for record in records:
        used_codes.update(record.main)
        used_codes.update(record.extra)
        used_codes.update(record.side)
    missing_cdb = sorted(used_codes.difference(cdb_codes))
    missing_code_list = sorted(used_codes.difference(code_list_codes))
    if missing_cdb:
        raise ValueError(f"deck cards absent from cards.cdb: {missing_cdb}")
    if missing_code_list:
        raise ValueError(
            f"deck cards absent from code list: {missing_code_list}"
        )

    core_seen = {anchor.core_sha256: anchor}
    duplicate_core: list[DeckRecord] = []
    unique_records: list[DeckRecord] = []
    for record in records:
        previous = core_seen.get(record.core_sha256)
        if previous is not None:
            duplicate_core.append(record)
            continue
        core_seen[record.core_sha256] = record
        unique_records.append(record)

    by_family: dict[str, list[DeckRecord]] = defaultdict(list)
    for record in unique_records:
        by_family[record.family].append(record)

    retained_by_family: dict[str, list[DeckRecord]] = {}
    capped_records: list[DeckRecord] = []
    for family, family_items in sorted(by_family.items()):
        retained, capped = choose_family_variants(
            family_items,
            args.max_variants_per_family,
            args.seed,
        )
        retained_by_family[family] = retained
        capped_records.extend(capped)

    assignments = assign_families(
        retained_by_family,
        args.validation_percent,
        args.seed,
    )

    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {args.output_root}"
        )
    train_dir = args.output_root / "train"
    validation_dir = args.output_root / "validation"
    manifest_dir = args.output_root / "manifests"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    write_deck(anchor, train_dir / anchor.output_name)
    output_names = {anchor.output_name}
    split_records: list[dict[str, object]] = []
    for family, family_items in sorted(retained_by_family.items()):
        split = assignments[family]
        output_dir = train_dir if split == "train" else validation_dir
        for record in sorted(
            family_items,
            key=lambda item: item.output_name,
        ):
            if record.output_name in output_names:
                raise ValueError(
                    f"flattened deck filename collision: {record.output_name}"
                )
            output_names.add(record.output_name)
            write_deck(record, output_dir / record.output_name)
            split_records.append(
                {
                    "source": record.source_relative,
                    "output_name": record.output_name,
                    "split": split,
                    "format": record.format_name,
                    "year": record.year,
                    "family": record.family,
                    "core_sha256": record.core_sha256,
                    "full_sha256": record.full_sha256,
                    "main": len(record.main),
                    "extra": len(record.extra),
                    "side": len(record.side),
                }
            )

    if args.token_deck is not None:
        for output_dir in (train_dir, validation_dir):
            shutil.copyfile(args.token_deck, output_dir / "_tokens.ydk")

    split_counts = Counter(
        item["split"]
        for item in split_records
    )
    split_counts["train"] += 1
    format_counts = Counter(
        f"{item['split']}:{item['format']}"
        for item in split_records
    )
    year_counts = Counter(
        f"{item['split']}:{item['year']}"
        for item in split_records
    )
    manifest = {
        "schema": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "source_root": str(args.source_root.resolve()),
            "anchor_deck": str(args.anchor_deck.resolve()),
            "anchor_name": args.anchor_name,
            "validation_percent": args.validation_percent,
            "max_variants_per_family": args.max_variants_per_family,
            "seed": args.seed,
            "cards_cdb_sha256": sha256_file(args.cards_cdb),
            "code_list_sha256": sha256_file(args.code_list),
        },
        "summary": {
            "source_decks": len(source_files),
            "source_unique_card_codes": len(used_codes),
            "gameplay_unique_decks_including_anchor": len(
                unique_records
            ) + 1,
            "duplicate_main_extra_decks_removed": len(duplicate_core),
            "family_cap_decks_removed": len(capped_records),
            "retained_families": len(retained_by_family),
            "train_decks_including_anchor": split_counts["train"],
            "validation_decks": split_counts["validation"],
            "anchor_forced_into_train": True,
            "anchor_family_for_other_decks": assignments.get(
                anchor.family
            ),
            "split_format_counts": dict(sorted(format_counts.items())),
            "split_year_counts": dict(sorted(year_counts.items())),
        },
        "anchor": {
            "output_name": anchor.output_name,
            "family": anchor.family,
            "core_sha256": anchor.core_sha256,
            "full_sha256": anchor.full_sha256,
        },
        "decks": split_records,
        "removed_duplicate_core_sources": [
            record.source_relative
            for record in duplicate_core
        ],
        "removed_by_family_cap": [
            record.source_relative
            for record in capped_records
        ],
    }
    manifest_path = manifest_dir / "deck_split_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "output_root": str(args.output_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        **manifest["summary"],
    }
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
