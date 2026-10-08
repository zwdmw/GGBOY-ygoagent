from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


SECTIONS = ("main", "extra", "side")
SIMILARITY_SECTION_WEIGHTS = {
    "main": 0.75,
    "extra": 0.25,
}
SOURCE_ID_RE = re.compile(r"^(?:\d+[_-])?(.*?)(?:[-_]\d+)?$", re.IGNORECASE)
FAMILY_TOKEN_RE = re.compile(r"[^a-z0-9]+")
GENERIC_FAMILY_TOKENS = {
    "deck",
    "duel",
    "final",
    "latest",
    "new",
    "ocg",
    "tcg",
    "top",
    "version",
    "v1",
    "v2",
    "v3",
}


@dataclass(frozen=True)
class ParsedDeck:
    main: collections.Counter[int]
    extra: collections.Counter[int]
    side: collections.Counter[int]

    def section(self, name: str) -> collections.Counter[int]:
        return getattr(self, name)

    def count(self, name: str) -> int:
        return sum(self.section(name).values())

    def canonical_sha256(self) -> str:
        payload = {
            name: sorted(self.section(name).items())
            for name in SECTIONS
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DeckRecord:
    index: int
    row: dict[str, object]
    source_path: Path
    deck: ParsedDeck
    canonical_sha256: str
    source_family: str | None

    @property
    def name(self) -> str:
        return str(self.row["deck_name"])

    @property
    def all_codes(self) -> set[int]:
        return set(self.deck.main) | set(self.deck.extra) | set(self.deck.side)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-split an existing canonical YGO deck dataset into uniform "
            "train, validation, and low-overlap final-test sets."
        )
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-code-list", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--validation-fraction",
        "--near-fraction",
        dest="validation_fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--test-fraction",
        "--far-fraction",
        dest="test_fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--cluster-threshold", type=float, default=0.80)
    parser.add_argument("--main-recall-threshold", type=float, default=0.90)
    parser.add_argument(
        "--separation-threshold",
        "--far-threshold",
        dest="separation_threshold",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--separation-main-recall-threshold",
        "--far-main-recall-threshold",
        dest="separation_main_recall_threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument("--candidate-topk", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument(
        "--pinned-train-deck-name",
        default="03_Sky_Striker_706646",
    )
    parser.add_argument(
        "--extra-official-validation-deck",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
                f"invalid code-list row at {path}:{line_number}"
            ) from exc
    if not codes:
        raise ValueError(f"empty code list: {path}")
    return codes


def parse_ydk(path: Path) -> ParsedDeck:
    sections = {
        name: collections.Counter()
        for name in SECTIONS
    }
    section = "main"
    for raw_line in path.read_text(
        encoding="utf-8-sig",
        errors="strict",
    ).splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        if lowered == "#main":
            section = "main"
        elif lowered == "#extra":
            section = "extra"
        elif lowered == "!side":
            section = "side"
        elif line and not line.startswith("#"):
            if not line.isdigit():
                raise ValueError(f"invalid YDK line {line!r}: {path}")
            sections[section][int(line)] += 1
    deck = ParsedDeck(**sections)
    if not 40 <= deck.count("main") <= 60:
        raise ValueError(f"invalid main-deck size in {path}")
    if deck.count("extra") > 15 or deck.count("side") > 15:
        raise ValueError(f"invalid extra/side size in {path}")
    return deck


def canonical_from_existing_format(deck: ParsedDeck) -> str:
    payload = "|".join(
        ",".join(
            str(code)
            for code, count in sorted(deck.section(section).items())
            for _ in range(count)
        )
        for section in SECTIONS
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def source_family(row: dict[str, object]) -> str | None:
    raw = str(row.get("representative_source") or "")
    stem = Path(raw).stem.casefold()
    match = SOURCE_ID_RE.match(stem)
    if match:
        stem = match.group(1)
    tokens = [
        token
        for token in FAMILY_TOKEN_RE.split(stem)
        if token and token not in GENERIC_FAMILY_TOKENS and not token.isdigit()
    ]
    if not tokens:
        return None
    return "-".join(tokens[:6])


def read_manifest(source_dataset: Path) -> tuple[list[DeckRecord], Path]:
    manifest_path = source_dataset / "manifests/decks.jsonl"
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[DeckRecord] = []
    seen_canonical: set[str] = set()
    for row in rows:
        relative = Path(str(row["output_path"]))
        path = source_dataset / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        deck = parse_ydk(path)
        canonical = canonical_from_existing_format(deck)
        expected = str(row["canonical_sha256"])
        if canonical != expected:
            raise ValueError(
                f"canonical hash mismatch for {path}: {canonical} != {expected}"
            )
        if canonical in seen_canonical:
            raise ValueError(f"duplicate canonical identity in source manifest: {canonical}")
        seen_canonical.add(canonical)
        records.append(
            DeckRecord(
                index=len(records),
                row=row,
                source_path=path,
                deck=deck,
                canonical_sha256=canonical,
                source_family=source_family(row),
            )
        )
    records.sort(key=lambda item: item.canonical_sha256)
    records = [
        DeckRecord(
            index=index,
            row=record.row,
            source_path=record.source_path,
            deck=record.deck,
            canonical_sha256=record.canonical_sha256,
            source_family=record.source_family,
        )
        for index, record in enumerate(records)
    ]
    return records, manifest_path


def multiset_intersection(
    left: collections.Counter[int],
    right: collections.Counter[int],
) -> int:
    return sum((left & right).values())


def multiset_union(
    left: collections.Counter[int],
    right: collections.Counter[int],
) -> int:
    return sum((left | right).values())


def multiset_jaccard(
    left: collections.Counter[int],
    right: collections.Counter[int],
) -> float:
    union = multiset_union(left, right)
    return multiset_intersection(left, right) / union if union else 1.0


def compute_section_idf(
    records: list[DeckRecord],
) -> dict[str, dict[int, float]]:
    document_count = len(records)
    if document_count == 0:
        raise ValueError("cannot compute IDF for an empty deck dataset")
    result: dict[str, dict[int, float]] = {}
    for section in SIMILARITY_SECTION_WEIGHTS:
        document_frequency: collections.Counter[int] = collections.Counter()
        for record in records:
            document_frequency.update(
                record.deck.section(section).keys()
            )
        result[section] = {
            code: math.log(
                (document_count + 1) / (frequency + 1)
            )
            + 1.0
            for code, frequency in document_frequency.items()
        }
    return result


def idf_weighted_multiset_jaccard(
    left: collections.Counter[int],
    right: collections.Counter[int],
    card_weights: dict[int, float],
) -> float:
    codes = set(left) | set(right)
    if not codes:
        return 1.0
    intersection = sum(
        card_weights.get(code, 1.0) * min(left[code], right[code])
        for code in codes
    )
    union = sum(
        card_weights.get(code, 1.0) * max(left[code], right[code])
        for code in codes
    )
    return intersection / union if union else 1.0


def weighted_similarity(
    left: ParsedDeck,
    right: ParsedDeck,
    section_idf: dict[str, dict[int, float]] | None = None,
) -> float:
    section_idf = section_idf or {}
    return sum(
        section_weight
        * idf_weighted_multiset_jaccard(
            left.section(section),
            right.section(section),
            section_idf.get(section, {}),
        )
        for section, section_weight in SIMILARITY_SECTION_WEIGHTS.items()
    )


def directional_recall(
    target: collections.Counter[int],
    candidate: collections.Counter[int],
) -> float:
    total = sum(target.values())
    return multiset_intersection(target, candidate) / total if total else 1.0


def pair_metrics(
    left: DeckRecord,
    right: DeckRecord,
    section_idf: dict[str, dict[int, float]] | None = None,
) -> dict[str, float]:
    return {
        "similarity": weighted_similarity(
            left.deck,
            right.deck,
            section_idf,
        ),
        "left_main_recall": directional_recall(left.deck.main, right.deck.main),
        "right_main_recall": directional_recall(right.deck.main, left.deck.main),
        "left_extra_recall": directional_recall(left.deck.extra, right.deck.extra),
        "right_extra_recall": directional_recall(right.deck.extra, left.deck.extra),
    }


def build_binary_matrix(
    records: list[DeckRecord],
    section: str,
    card_to_column: dict[int, int],
    card_weights: dict[int, float],
) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for record in records:
        for code in record.deck.section(section):
            rows.append(record.index)
            columns.append(card_to_column[code])
            values.append(math.sqrt(card_weights.get(code, 1.0)))
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(records), len(card_to_column)),
        dtype=np.float32,
    )


def candidate_pairs(
    records: list[DeckRecord],
    *,
    topk: int,
    batch_size: int,
    section_idf: dict[str, dict[int, float]] | None = None,
) -> set[tuple[int, int]]:
    section_idf = section_idf or compute_section_idf(records)
    cards = sorted(
        {
            code
            for record in records
            for section in SIMILARITY_SECTION_WEIGHTS
            for code in record.deck.section(section)
        }
    )
    card_to_column = {code: index for index, code in enumerate(cards)}
    matrices = {
        section: build_binary_matrix(
            records,
            section,
            card_to_column,
            section_idf[section],
        )
        for section in SIMILARITY_SECTION_WEIGHTS
    }
    count = len(records)
    topk = min(max(1, topk), max(1, count - 1))
    pairs: set[tuple[int, int]] = set()
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        scores = np.zeros((stop - start, count), dtype=np.float32)
        for section, section_weight in SIMILARITY_SECTION_WEIGHTS.items():
            intersection = (
                matrices[section][start:stop] @ matrices[section].T
            ).toarray()
            scores += (
                intersection.astype(np.float32, copy=False)
                * section_weight
            )
        local_rows = np.arange(stop - start)
        global_rows = np.arange(start, stop)
        scores[local_rows, global_rows] = -1
        indexes = np.argpartition(scores, -topk, axis=1)[:, -topk:]
        for local, neighbors in enumerate(indexes):
            left = start + local
            for right in neighbors:
                right = int(right)
                if scores[local, right] <= 0:
                    continue
                pairs.add((left, right) if left < right else (right, left))
        print(
            f"candidate_search rows={stop}/{count} pairs={len(pairs)}",
            flush=True,
        )

    by_existing_cluster: dict[int, list[int]] = collections.defaultdict(list)
    by_family: dict[str, list[int]] = collections.defaultdict(list)
    for record in records:
        cluster = int(record.row.get("near_duplicate_cluster", -1))
        by_existing_cluster[cluster].append(record.index)
        if record.source_family:
            by_family[record.source_family].append(record.index)
    add_group_pairs(pairs, by_existing_cluster.values())
    add_group_pairs(pairs, by_family.values(), maximum_group_size=256)
    return pairs


def add_group_pairs(
    pairs: set[tuple[int, int]],
    groups: Iterable[list[int]],
    *,
    maximum_group_size: int | None = None,
) -> None:
    for members in groups:
        if len(members) < 2:
            continue
        if maximum_group_size is not None and len(members) > maximum_group_size:
            continue
        ordered = sorted(members)
        for offset, left in enumerate(ordered[:-1]):
            for right in ordered[offset + 1 :]:
                pairs.add((left, right))


def evaluate_pairs(
    records: list[DeckRecord],
    pairs: set[tuple[int, int]],
    section_idf: dict[str, dict[int, float]] | None = None,
) -> dict[tuple[int, int], dict[str, float]]:
    evaluated: dict[tuple[int, int], dict[str, float]] = {}
    for number, pair in enumerate(sorted(pairs), 1):
        evaluated[pair] = pair_metrics(
            records[pair[0]],
            records[pair[1]],
            section_idf,
        )
        if number % 250_000 == 0:
            print(f"exact_pairs={number}/{len(pairs)}", flush=True)
    return evaluated


def build_base_clusters(
    records: list[DeckRecord],
    evaluated: dict[tuple[int, int], dict[str, float]],
    *,
    similarity_threshold: float,
    main_recall_threshold: float,
) -> tuple[list[list[int]], dict[int, int]]:
    groups = DisjointSet(len(records))
    for pair, metrics in evaluated.items():
        left, right = pair
        if (
            metrics["similarity"] >= similarity_threshold
            or metrics["left_main_recall"] >= main_recall_threshold
            or metrics["right_main_recall"] >= main_recall_threshold
        ):
            groups.union(left, right)

    clusters_by_root: dict[int, list[int]] = collections.defaultdict(list)
    for record in records:
        clusters_by_root[groups.find(record.index)].append(record.index)
    clusters = sorted(
        clusters_by_root.values(),
        key=lambda members: min(records[index].canonical_sha256 for index in members),
    )
    cluster_by_record: dict[int, int] = {}
    for cluster_id, members in enumerate(clusters):
        for index in members:
            cluster_by_record[index] = cluster_id
    return clusters, cluster_by_record


def cluster_neighbor_risk(
    clusters: list[list[int]],
    cluster_by_record: dict[int, int],
    evaluated: dict[tuple[int, int], dict[str, float]],
) -> dict[int, dict[str, float | int | None]]:
    risk: dict[int, dict[str, float | int | None]] = {
        cluster_id: {
            "max_similarity": 0.0,
            "max_main_recall": 0.0,
            "nearest_cluster": None,
        }
        for cluster_id in range(len(clusters))
    }
    for (left, right), metrics in evaluated.items():
        left_cluster = cluster_by_record[left]
        right_cluster = cluster_by_record[right]
        if left_cluster == right_cluster:
            continue
        similarity = float(metrics["similarity"])
        main_recall = max(
            float(metrics["left_main_recall"]),
            float(metrics["right_main_recall"]),
        )
        for current, neighbor in (
            (left_cluster, right_cluster),
            (right_cluster, left_cluster),
        ):
            row = risk[current]
            candidate = (
                main_recall,
                similarity,
                -neighbor,
            )
            existing = (
                float(row["max_main_recall"]),
                float(row["max_similarity"]),
                -int(row["nearest_cluster"])
                if row["nearest_cluster"] is not None
                else float("-inf"),
            )
            if candidate > existing:
                row["max_main_recall"] = main_recall
                row["max_similarity"] = similarity
                row["nearest_cluster"] = neighbor
    return risk


def build_far_components(
    clusters: list[list[int]],
    cluster_by_record: dict[int, int],
    evaluated: dict[tuple[int, int], dict[str, float]],
    *,
    far_threshold: float,
    far_main_recall_threshold: float,
) -> list[list[int]]:
    groups = DisjointSet(len(clusters))
    for pair, metrics in evaluated.items():
        left_cluster = cluster_by_record[pair[0]]
        right_cluster = cluster_by_record[pair[1]]
        if left_cluster == right_cluster:
            continue
        if (
            metrics["similarity"] > far_threshold
            or metrics["left_main_recall"] > far_main_recall_threshold
            or metrics["right_main_recall"] > far_main_recall_threshold
        ):
            groups.union(left_cluster, right_cluster)
    components: dict[int, list[int]] = collections.defaultdict(list)
    for cluster_id in range(len(clusters)):
        components[groups.find(cluster_id)].append(cluster_id)
    return sorted(components.values(), key=lambda item: tuple(item))


def deterministic_rank(seed: int, label: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{identity}".encode("ascii")).hexdigest()


def select_far_components(
    records: list[DeckRecord],
    clusters: list[list[int]],
    components: list[list[int]],
    *,
    target_decks: int,
    seed: int,
    excluded_clusters: set[int] | None = None,
) -> set[int]:
    excluded_clusters = excluded_clusters or set()
    items: list[tuple[int, str, list[int]]] = []
    for component in components:
        if excluded_clusters.intersection(component):
            continue
        size = sum(len(clusters[cluster_id]) for cluster_id in component)
        identity = min(
            records[index].canonical_sha256
            for cluster_id in component
            for index in clusters[cluster_id]
        )
        items.append(
            (
                size,
                deterministic_rank(seed, "far", identity),
                component,
            )
        )
    items.sort(key=lambda item: (item[0] > target_decks, item[0], item[1]))
    selected: set[int] = set()
    selected_count = 0
    remaining = list(items)
    while remaining and selected_count < target_decks:
        best_position = min(
            range(len(remaining)),
            key=lambda position: (
                abs(
                    target_decks
                    - (
                        selected_count
                        + remaining[position][0]
                    )
                ),
                remaining[position][0] > max(target_decks, target_decks * 2 - selected_count),
                remaining[position][1],
            ),
        )
        size, _rank, component = remaining.pop(best_position)
        current_error = abs(target_decks - selected_count)
        next_error = abs(target_decks - selected_count - size)
        if selected_count > 0 and next_error > current_error:
            continue
        selected.update(component)
        selected_count += size
    return selected


def primary_bucket(record: DeckRecord) -> str:
    buckets = [
        str(value)
        for value in record.row.get("source_buckets", [])
    ]
    return min(buckets, default="UNKNOWN")


def select_near_clusters(
    records: list[DeckRecord],
    clusters: list[list[int]],
    excluded_clusters: set[int],
    *,
    target_decks: int,
    seed: int,
    initial_selected: set[int] | None = None,
    cluster_risk: dict[
        int, dict[str, float | int | None]
    ] | None = None,
    label: str = "heldout",
) -> set[int]:
    initial_selected = initial_selected or set()
    cluster_risk = cluster_risk or {}
    if initial_selected & excluded_clusters:
        raise ValueError("initial validation clusters overlap exclusions")
    bucketed: dict[
        str,
        list[tuple[float, float, str, int, int]],
    ] = collections.defaultdict(list)
    for cluster_id, members in enumerate(clusters):
        if (
            cluster_id in excluded_clusters
            or cluster_id in initial_selected
        ):
            continue
        buckets = collections.Counter(primary_bucket(records[index]) for index in members)
        bucket = min(
            (
                value
                for value, count in buckets.items()
                if count == max(buckets.values())
            ),
            default="UNKNOWN",
        )
        identity = min(records[index].canonical_sha256 for index in members)
        risk = cluster_risk.get(cluster_id, {})
        bucketed[bucket].append(
            (
                float(risk.get("max_main_recall", 0.0)),
                float(risk.get("max_similarity", 0.0)),
                deterministic_rank(
                    seed,
                    f"{label}:{bucket}",
                    identity,
                ),
                cluster_id,
                len(members),
            )
        )

    total_available = sum(
        size
        for items in bucketed.values()
        for _recall, _similarity, _rank, _cluster_id, size in items
    )
    selected: set[int] = set(initial_selected)
    initial_size = sum(
        len(clusters[cluster_id])
        for cluster_id in selected
    )
    remaining_target = max(0, target_decks - initial_size)
    for bucket, items in sorted(bucketed.items()):
        bucket_total = sum(
            size
            for _recall, _similarity, _rank, _cluster_id, size in items
        )
        bucket_target = round(
            remaining_target * bucket_total / max(1, total_available)
        )
        current = 0
        for (
            _recall,
            _similarity,
            _rank,
            cluster_id,
            size,
        ) in sorted(items):
            if current >= bucket_target:
                break
            if abs(bucket_target - current - size) <= abs(bucket_target - current):
                selected.add(cluster_id)
                current += size

    def selected_size() -> int:
        return sum(len(clusters[cluster_id]) for cluster_id in selected)

    remaining = [
        cluster_id
        for cluster_id in range(len(clusters))
        if cluster_id not in excluded_clusters and cluster_id not in selected
    ]
    remaining.sort(
        key=lambda cluster_id: (
            float(
                cluster_risk.get(cluster_id, {}).get(
                    "max_main_recall",
                    0.0,
                )
            ),
            float(
                cluster_risk.get(cluster_id, {}).get(
                    "max_similarity",
                    0.0,
                )
            ),
            deterministic_rank(
                seed,
                f"{label}-fill",
                min(
                    records[index].canonical_sha256
                    for index in clusters[cluster_id]
                ),
            ),
        )
    )
    for cluster_id in remaining:
        if selected_size() >= target_decks:
            break
        size = len(clusters[cluster_id])
        if (
            abs(target_decks - selected_size() - size)
            <= abs(target_decks - selected_size())
        ):
            selected.add(cluster_id)
    return selected


def nearest_training_metrics(
    records: list[DeckRecord],
    split_by_record: dict[int, str],
    evaluated: dict[tuple[int, int], dict[str, float]],
) -> dict[int, dict[str, object]]:
    training = {
        index
        for index, split in split_by_record.items()
        if split == "train"
    }
    nearest: dict[int, dict[str, object]] = {}
    for pair, metrics in evaluated.items():
        left, right = pair
        if left in training and right not in training:
            validation_index, training_index = right, left
            main_recall = metrics["right_main_recall"]
            extra_recall = metrics["right_extra_recall"]
        elif right in training and left not in training:
            validation_index, training_index = left, right
            main_recall = metrics["left_main_recall"]
            extra_recall = metrics["left_extra_recall"]
        else:
            continue
        candidate = {
            "deck_name": records[training_index].name,
            "canonical_sha256": records[training_index].canonical_sha256,
            "similarity": metrics["similarity"],
            "main_recall": main_recall,
            "extra_recall": extra_recall,
        }
        current = nearest.get(validation_index)
        if current is None or (
            candidate["similarity"],
            candidate["main_recall"],
            candidate["extra_recall"],
            candidate["canonical_sha256"],
        ) > (
            current["similarity"],
            current["main_recall"],
            current["extra_recall"],
            current["canonical_sha256"],
        ):
            nearest[validation_index] = candidate
    return nearest


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    os.replace(temporary, path)


def validate_args(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if not args.official_code_list.is_file():
        raise FileNotFoundError(args.official_code_list)
    for path in args.extra_official_validation_deck:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.validation_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("validation fractions must be positive")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation fractions must leave a training split")
    if not (
        0
        < args.separation_threshold
        < args.cluster_threshold
        <= 1
    ):
        raise ValueError(
            "require 0 < separation_threshold < cluster_threshold <= 1"
        )
    if args.candidate_topk <= 0 or args.batch_size <= 0:
        raise ValueError("candidate-topk and batch-size must be positive")


def main() -> int:
    args = parse_args()
    args.source_dataset = args.source_dataset.resolve()
    args.output = args.output.resolve()
    args.official_code_list = args.official_code_list.resolve()
    args.extra_official_validation_deck = [
        path.resolve()
        for path in args.extra_official_validation_deck
    ]
    validate_args(args)

    records, source_manifest = read_manifest(args.source_dataset)
    official_codes = read_code_list(args.official_code_list)
    print(f"loaded_unique_decks={len(records)}", flush=True)
    section_idf = compute_section_idf(records)
    pairs = candidate_pairs(
        records,
        topk=args.candidate_topk,
        batch_size=args.batch_size,
        section_idf=section_idf,
    )
    print(f"candidate_pairs={len(pairs)}", flush=True)
    evaluated = evaluate_pairs(records, pairs, section_idf)
    clusters, cluster_by_record = build_base_clusters(
        records,
        evaluated,
        similarity_threshold=args.cluster_threshold,
        main_recall_threshold=args.main_recall_threshold,
    )
    print(f"base_clusters={len(clusters)}", flush=True)

    pinned_train_records = [
        record
        for record in records
        if record.name == args.pinned_train_deck_name
    ]
    if len(pinned_train_records) != 1:
        raise ValueError(
            "expected exactly one pinned training deck named "
            f"{args.pinned_train_deck_name!r}, found "
            f"{len(pinned_train_records)}"
        )
    pinned_train_clusters = {
        cluster_by_record[record.index]
        for record in pinned_train_records
    }
    official_compatible_by_record = {
        record.index: record.all_codes <= official_codes
        for record in records
    }
    official_validation_clusters = {
        cluster_by_record[record.index]
        for record in records
        if official_compatible_by_record[record.index]
    }
    if pinned_train_clusters & official_validation_clusters:
        raise ValueError(
            "pinned training cluster is also official-model compatible; "
            "the split contract would be ambiguous"
        )

    cluster_risk = cluster_neighbor_risk(
        clusters,
        cluster_by_record,
        evaluated,
    )

    test_target = round(len(records) * args.test_fraction)
    validation_target = round(
        len(records) * args.validation_fraction
    )
    separation_components = build_far_components(
        clusters,
        cluster_by_record,
        evaluated,
        far_threshold=args.separation_threshold,
        far_main_recall_threshold=(
            args.separation_main_recall_threshold
        ),
    )
    protected_clusters = (
        pinned_train_clusters | official_validation_clusters
    )
    eligible_separation_components = [
        component
        for component in separation_components
        if not protected_clusters.intersection(component)
    ]
    separation_component_sizes = [
        sum(len(clusters[cluster_id]) for cluster_id in component)
        for component in separation_components
    ]
    eligible_separation_decks = sum(
        sum(
            len(clusters[cluster_id])
            for cluster_id in component
        )
        for component in eligible_separation_components
    )
    print(
        "separation_components="
        f"{len(separation_components)}, "
        f"largest_decks={max(separation_component_sizes)}, "
        f"eligible_decks={eligible_separation_decks}",
        flush=True,
    )
    test_clusters = select_far_components(
        records,
        clusters,
        separation_components,
        target_decks=test_target,
        seed=args.seed + 1,
        excluded_clusters=protected_clusters,
    )
    validation_clusters = select_near_clusters(
        records,
        clusters,
        pinned_train_clusters | test_clusters,
        target_decks=validation_target,
        seed=args.seed,
        initial_selected=official_validation_clusters,
        cluster_risk=cluster_risk,
        label="validation",
    )
    if test_clusters & validation_clusters:
        raise AssertionError(
            "validation and test cluster selections overlap"
        )
    if pinned_train_clusters & (test_clusters | validation_clusters):
        raise AssertionError("pinned training cluster was held out")

    split_by_record: dict[int, str] = {}
    for record in records:
        cluster_id = cluster_by_record[record.index]
        if cluster_id in test_clusters:
            split = "test"
        elif cluster_id in validation_clusters:
            split = "validation"
        else:
            split = "train"
        split_by_record[record.index] = split
    nearest = nearest_training_metrics(records, split_by_record, evaluated)

    manifest_rows: list[dict[str, object]] = []
    destination_by_record: dict[int, Path] = {}
    split_counts = collections.Counter(split_by_record.values())
    canonical_by_split: dict[str, set[str]] = collections.defaultdict(set)
    for record in records:
        split = split_by_record[record.index]
        cluster_id = cluster_by_record[record.index]
        output_name = (
            f"cluster_{cluster_id:05d}__{record.source_path.name}"
        )
        destination = args.output / split / output_name
        materialize_file(record.source_path, destination, args.link_mode)
        destination_by_record[record.index] = destination
        canonical_by_split[split].add(record.canonical_sha256)
        nearest_row = nearest.get(record.index)
        row = {
            **record.row,
            "source_deck_name": record.name,
            "deck_name": destination.stem,
            "split": split,
            "output_path": destination.relative_to(args.output).as_posix(),
            "output_sha256": sha256_file(destination),
            "canonical_sha256": record.canonical_sha256,
            "content_cluster": cluster_id,
            "source_family": record.source_family,
            "official_model_compatible": (
                official_compatible_by_record[record.index]
            ),
            "pinned_train": cluster_id in pinned_train_clusters,
            "nearest_train": nearest_row,
        }
        manifest_rows.append(row)

    official_eval_rows: list[dict[str, object]] = []
    official_validation_root = args.output / "official_validation"
    for record in records:
        if not official_compatible_by_record[record.index]:
            continue
        if split_by_record[record.index] != "validation":
            raise AssertionError(
                "official-compatible source deck was not held out"
            )
        source = destination_by_record[record.index]
        destination = official_validation_root / source.name
        materialize_file(source, destination, args.link_mode)
        official_eval_rows.append(
            {
                "kind": "dataset_validation",
                "deck_name": destination.stem,
                "path": destination.relative_to(args.output).as_posix(),
                "sha256": sha256_file(destination),
                "canonical_sha256": record.canonical_sha256,
                "content_cluster": cluster_by_record[record.index],
            }
        )
    for external in args.extra_official_validation_deck:
        parsed = parse_ydk(external)
        codes = (
            set(parsed.main)
            | set(parsed.extra)
            | set(parsed.side)
        )
        missing = sorted(codes - official_codes)
        if missing:
            raise ValueError(
                f"external official-validation deck {external} contains "
                f"codes absent from the official model: {missing}"
            )
        destination = (
            official_validation_root / f"external__{external.name}"
        )
        materialize_file(external, destination, args.link_mode)
        official_eval_rows.append(
            {
                "kind": "external_fixed",
                "deck_name": destination.stem,
                "path": destination.relative_to(args.output).as_posix(),
                "sha256": sha256_file(destination),
                "canonical_sha256": parsed.canonical_sha256(),
                "content_cluster": None,
            }
        )

    split_names = sorted(canonical_by_split)
    for left_position, left in enumerate(split_names):
        for right in split_names[left_position + 1 :]:
            overlap = canonical_by_split[left] & canonical_by_split[right]
            if overlap:
                raise AssertionError(
                    f"canonical identities leaked across {left}/{right}: "
                    f"{sorted(overlap)[:5]}"
                )

    cross_split_cluster_violations = []
    for pair, metrics in evaluated.items():
        left, right = pair
        if split_by_record[left] == split_by_record[right]:
            continue
        if (
            metrics["similarity"] >= args.cluster_threshold
            or metrics["left_main_recall"]
            >= args.main_recall_threshold
            or metrics["right_main_recall"]
            >= args.main_recall_threshold
        ):
            cross_split_cluster_violations.append(
                {
                    "left": records[left].canonical_sha256,
                    "right": records[right].canonical_sha256,
                    "left_split": split_by_record[left],
                    "right_split": split_by_record[right],
                    "metrics": metrics,
                }
            )

    test_violations = []
    for record in records:
        if split_by_record[record.index] != "test":
            continue
        metrics = nearest.get(record.index)
        if metrics is None:
            continue
        if (
            float(metrics["similarity"])
            > args.separation_threshold
            or float(metrics["main_recall"])
            > args.separation_main_recall_threshold
        ):
            test_violations.append(
                {
                    "canonical_sha256": record.canonical_sha256,
                    "nearest_train": metrics,
                }
            )

    manifests = args.output / "manifests"
    all_manifest = manifests / "decks.jsonl"
    write_jsonl(all_manifest, manifest_rows)
    split_order = ("train", "validation", "test")
    for split in split_order:
        write_jsonl(
            manifests / f"{split}.jsonl",
            (row for row in manifest_rows if row["split"] == split),
        )
    cluster_rows = []
    for cluster_id, members in enumerate(clusters):
        splits = {split_by_record[index] for index in members}
        if len(splits) != 1:
            raise AssertionError(
                f"cluster {cluster_id} leaked across splits: {splits}"
            )
        cluster_rows.append(
            {
                "content_cluster": cluster_id,
                "split": next(iter(splits)),
                "size": len(members),
                "official_model_compatible_decks": sum(
                    official_compatible_by_record[index]
                    for index in members
                ),
                "pinned_train": cluster_id in pinned_train_clusters,
                "canonical_sha256": [
                    records[index].canonical_sha256
                    for index in members
                ],
            }
        )
    clusters_manifest = manifests / "clusters.jsonl"
    write_jsonl(clusters_manifest, cluster_rows)
    official_manifest = manifests / "official_validation.jsonl"
    write_jsonl(official_manifest, official_eval_rows)

    source_summary = args.source_dataset / "manifests/summary.json"
    source_contract = args.source_dataset / "manifests/training_contract.json"
    summary = {
        "schema": "ygo-uniform-cluster-split/v2",
        "seed": args.seed,
        "source_dataset": str(args.source_dataset),
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "source_summary_sha256": (
            sha256_file(source_summary) if source_summary.is_file() else None
        ),
        "source_contract_sha256": (
            sha256_file(source_contract) if source_contract.is_file() else None
        ),
        "unique_canonical_decks": len(records),
        "similarity_metric": {
            "name": "idf_weighted_multiset_jaccard",
            "idf_formula": "log((document_count + 1) / (document_frequency + 1)) + 1",
            "section_weights": SIMILARITY_SECTION_WEIGHTS,
            "side_deck_included": False,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "fractions": {
            split: split_counts[split] / len(records)
            for split in sorted(split_counts)
        },
        "base_cluster_count": len(clusters),
        "cluster_neighbor_risk": {
            "maximum_main_recall": max(
                float(row["max_main_recall"])
                for row in cluster_risk.values()
            ),
            "maximum_similarity": max(
                float(row["max_similarity"])
                for row in cluster_risk.values()
            ),
        },
        "separation_components": {
            "count": len(separation_components),
            "maximum_decks": max(separation_component_sizes),
            "eligible_count": len(eligible_separation_components),
            "eligible_decks": eligible_separation_decks,
        },
        "pinned_train_deck": args.pinned_train_deck_name,
        "pinned_train_clusters": sorted(pinned_train_clusters),
        "official_model": {
            "code_list": str(args.official_code_list),
            "code_list_sha256": sha256_file(args.official_code_list),
            "compatible_source_decks": sum(
                official_compatible_by_record.values()
            ),
            "heldout_clusters": len(official_validation_clusters),
            "evaluation_decks": len(official_eval_rows),
            "manifest": {
                "path": official_manifest.relative_to(
                    args.output
                ).as_posix(),
                "sha256": sha256_file(official_manifest),
            },
        },
        "thresholds": {
            "cluster_similarity": args.cluster_threshold,
            "cluster_main_recall": args.main_recall_threshold,
            "test_separation_similarity": (
                args.separation_threshold
            ),
            "test_separation_main_recall": (
                args.separation_main_recall_threshold
            ),
            "candidate_topk": args.candidate_topk,
        },
        "cross_split_cluster_violations": (
            cross_split_cluster_violations
        ),
        "test_separation_violations": test_violations,
        "manifest": {
            "path": all_manifest.relative_to(args.output).as_posix(),
            "sha256": sha256_file(all_manifest),
        },
        "split_manifests": {
            split: {
                "path": f"manifests/{split}.jsonl",
                "sha256": sha256_file(manifests / f"{split}.jsonl"),
            }
            for split in split_order
        },
        "clusters_manifest": {
            "path": clusters_manifest.relative_to(
                args.output
            ).as_posix(),
            "sha256": sha256_file(clusters_manifest),
        },
    }
    summary["contract_sha256"] = sha256_json(
        {
            "seed": args.seed,
            "similarity_metric": summary["similarity_metric"],
            "thresholds": summary["thresholds"],
            "split_manifests": summary["split_manifests"],
        }
    )
    write_json(manifests / "split_summary.json", summary)
    write_json(
        manifests / "overlap_audit.json",
        {
            "schema": "ygo-deck-overlap-audit/v1",
            "thresholds": summary["thresholds"],
            "cross_split_cluster_violations": (
                cross_split_cluster_violations
            ),
            "test_separation_violations": test_violations,
        },
    )
    write_json(
        manifests / "training_contract.json",
        {
            "schema": "ygo-uniform-training-contract/v2",
            "deck_root": "train",
            "validation_root": "validation",
            "test_root": "test",
            "official_validation_root": "official_validation",
            "sampling_unit": "content_cluster",
            "variant_sampling": "uniform_within_cluster",
            "pair_sampling": "independent_uniform_clusters",
            "deck_schedule": "cluster_uniform",
            "featured_deck": None,
            "featured_deck_probability": 0.0,
            "pinned_train_deck": args.pinned_train_deck_name,
            "same_deck_mirror_evaluation": True,
            "manifest_sha256": summary["manifest"]["sha256"],
            "contract_sha256": summary["contract_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "base_cluster_count": summary["base_cluster_count"],
                "separation_components": summary["separation_components"],
                "split_counts": summary["split_counts"],
                "fractions": summary["fractions"],
                "official_model": summary["official_model"],
                "cross_split_cluster_violation_count": len(
                    cross_split_cluster_violations
                ),
                "test_separation_violation_count": len(
                    test_violations
                ),
                "contract_sha256": summary["contract_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if not cross_split_cluster_violations
        and not test_violations
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
