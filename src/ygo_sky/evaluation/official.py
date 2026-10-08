from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ygoenv

from ygoai.rl.env import EnvPreprocess, RecordEpisodeStatistics
from ygoai.rl.jax.agent import RNNAgent
from ygoai.rl.jax.decision_checkpoint import restore_decision_checkpoint
from ygo_sky.semantics import load_structured_semantics, inject_semantic_constants
from ygo_sky.paths import project_root
from ygoai.utils import init_ygopro


ID_FIELDS = (
    ("cards_", 0),
    ("actions_", 1),
    ("h_actions_", 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the official ID-embedding model against the structured "
            "model with paired, seat-reversed self-play."
        )
    )
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--structured-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--deck",
        type=Path,
        default=Path("assets/deck/BlueEyesGPU.ydk"),
    )
    parser.add_argument(
        "--code-list",
        type=Path,
        default=Path("scripts/code_list.txt"),
    )
    parser.add_argument(
        "--official-code-list",
        type=Path,
        default=Path("scripts/official_code_list_864.txt"),
        help="Historical card-code order used to train the official model.",
    )
    parser.add_argument(
        "--semantic-cache",
        type=Path,
        default=Path("assets/structured/frozen_semantics_v1.npz"),
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=256,
        help="Number of paired seeds; total games are twice this value.",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--env-threads", type=int, default=16)
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-history-actions", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--allow-official-unknown", action="store_true")
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument(
        "--structured-policy",
        choices=("argmax", "sample"),
        default="argmax",
    )
    parser.add_argument(
        "--official-policy",
        choices=("argmax", "sample"),
        default="argmax",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("battle-mixed-blueeyes-result.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_semantic_shape(path: Path) -> tuple[int, int, int, int]:
    with np.load(path, allow_pickle=False) as payload:
        cdb_exact = payload["cdb_exact"]
        lua_effects = payload["lua_effects"]
        lua_mask = payload["lua_mask"]
        if lua_mask.shape != lua_effects.shape[:2]:
            raise ValueError(
                f"semantic mask {lua_mask.shape} does not match "
                f"effects {lua_effects.shape}"
            )
        if cdb_exact.shape[0] != lua_effects.shape[0]:
            raise ValueError(
                f"semantic row mismatch: {cdb_exact.shape[0]} and "
                f"{lua_effects.shape[0]}"
            )
        return (
            int(cdb_exact.shape[0]),
            int(cdb_exact.shape[1]),
            int(lua_effects.shape[1]),
            int(lua_effects.shape[2]),
        )


def read_code_list(path: Path) -> list[int]:
    codes: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields:
                codes.append(int(fields[0]))
    return codes


def read_deck_codes(path: Path) -> list[int]:
    codes: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value.isdigit():
                codes.append(int(value))
    return codes


def deck_index_stats(
    deck_path: Path,
    current_code_list_path: Path,
    official_code_list_path: Path,
) -> dict[str, Any]:
    current_code_to_id = {
        code: index
        for index, code in enumerate(
            read_code_list(current_code_list_path),
            start=1,
        )
    }
    official_code_to_id = {
        code: index
        for index, code in enumerate(
            read_code_list(official_code_list_path),
            start=1,
        )
    }
    deck_codes = read_deck_codes(deck_path)
    missing_current = sorted(set(deck_codes).difference(current_code_to_id))
    missing_official = sorted(set(deck_codes).difference(official_code_to_id))
    if missing_current:
        raise ValueError(
            f"deck cards missing from current code list: {missing_current}"
        )
    current_ids = [current_code_to_id[code] for code in deck_codes]
    official_ids = [official_code_to_id.get(code, 0) for code in deck_codes]
    return {
        "copies": len(deck_codes),
        "unique_cards": len(set(deck_codes)),
        "min_current_id": min(current_ids),
        "max_current_id": max(current_ids),
        "min_official_id": min(official_ids),
        "max_official_id": max(official_ids),
        "missing_current_codes": missing_current,
        "missing_official_codes": missing_official,
        "missing_official_copies": sum(code in missing_official for code in deck_codes),
    }


def build_official_id_lookup(
    current_code_list_path: Path,
    official_code_list_path: Path,
    official_max_id: int,
) -> tuple[np.ndarray, dict[str, int]]:
    current_codes = read_code_list(current_code_list_path)
    official_codes = read_code_list(official_code_list_path)
    if len(current_codes) != len(set(current_codes)):
        raise ValueError("current code list contains duplicate card codes")
    if len(official_codes) != len(set(official_codes)):
        raise ValueError("official code list contains duplicate card codes")
    if len(official_codes) > official_max_id:
        raise ValueError(
            f"official code list has {len(official_codes)} cards but the "
            f"checkpoint only accepts IDs through {official_max_id}"
        )

    official_code_to_id = {
        code: index
        for index, code in enumerate(official_codes, start=1)
    }
    lookup = np.zeros(len(current_codes) + 1, dtype=np.int32)
    for current_id, code in enumerate(current_codes, start=1):
        lookup[current_id] = official_code_to_id.get(code, 0)
    mapped = int(np.count_nonzero(lookup))
    return lookup, {
        "current_rows": len(current_codes),
        "official_rows": len(official_codes),
        "mapped_current_rows": mapped,
        "unmapped_current_rows": len(current_codes) - mapped,
    }


def make_env(
    args: argparse.Namespace,
    deck_name: str,
):
    base = ygoenv.make(
        task_id="YGOPro-v1",
        env_type="gymnasium",
        num_envs=args.pairs,
        num_threads=min(args.env_threads, args.pairs),
        seed=args.seed,
        player=-1,
        deck1=deck_name,
        deck2=deck_name,
        max_options=args.max_options,
        max_steps=args.max_steps,
        n_history_actions=args.n_history_actions,
        async_reset=False,
        greedy_reward=False,
        play_mode="self",
        timeout=args.timeout,
        oppo_info=False,
        record=False,
        thread_affinity_offset=-1,
        deck_schedule="independent",
        anchor_probability_percent=0,
    )
    base.num_envs = args.pairs
    env = EnvPreprocess(base, skip_mask=True)
    return RecordEpisodeStatistics(env)


def batched_sample(observation_space) -> dict[str, jax.Array]:
    return jax.tree.map(
        lambda value: jnp.asarray([value]),
        observation_space.sample(),
    )


def load_models(
    sample_observation: dict[str, jax.Array],
    official_checkpoint: Path,
    structured_checkpoint: Path,
    semantic_shape: tuple[int, int, int, int],
    seed: int,
):
    key = jax.random.PRNGKey(seed)
    official_agent = RNNAgent(structured=False)
    sidecar = json.loads(structured_checkpoint.with_name(
        structured_checkpoint.name + ".json").read_text())
    if sidecar["schema"] != "ygo-training-checkpoint/v2":
        raise ValueError("Unsupported checkpoint metadata")
    if sidecar.get("architecture") != "decision-v1" or sha256_file(structured_checkpoint) != sidecar["sha256"]:
        raise ValueError("Checkpoint identity mismatch")
    if tuple(sidecar["semantic_shape"]) != tuple(semantic_shape):
        raise ValueError("Checkpoint semantic shape mismatch")
    structured_agent = RNNAgent(
        **sidecar["model_args"], semantic_shape=semantic_shape,
        dtype=jnp.bfloat16 if sidecar["bfloat16"] else jnp.float32,
        switch=sidecar["switch"],
    )

    official_state = official_agent.init_rnn_state(1)
    official_variables = official_agent.init(
        key,
        sample_observation,
        official_state,
    )
    official_variables, added = restore_decision_checkpoint(
        official_variables,
        official_checkpoint.read_bytes(),
    )
    assert not added

    structured_state = structured_agent.init_rnn_state(1)
    structured_variables = structured_agent.init(
        key,
        sample_observation,
        structured_state,
    )
    structured_variables, added = restore_decision_checkpoint(
        structured_variables,
        structured_checkpoint.read_bytes(),
    )
    assert not added
    tables, checked_shape, _ = load_structured_semantics(
        str(project_root() / "resources/game/assets/structured/frozen_semantics_v1.npz"),
        str(project_root() / "resources/game/scripts/code_list.txt"))
    assert tuple(checked_shape) == tuple(semantic_shape)
    structured_variables = flax.core.unfreeze(structured_variables)
    inject_semantic_constants(structured_variables, tables)
    structured_variables = flax.core.freeze(structured_variables)
    print(f"strict_model_restore_ok architecture={sidecar['architecture']} "
          f"expert_step={sidecar['global_step']}", flush=True)

    official_variables = jax.device_put(official_variables)
    structured_variables = jax.device_put(structured_variables)
    embedding = official_variables["params"]["Encoder_0"]["Embed_0"][
        "embedding"
    ]
    official_max_id = int(embedding.shape[0] - 1)
    return (
        official_agent,
        official_variables,
        structured_agent,
        structured_variables,
        official_max_id,
    )


def remap_official_observation(
    observation: dict[str, jax.Array | None],
    id_lookup: jax.Array,
) -> dict[str, jax.Array | None]:
    remapped = dict(observation)
    for name, start in ID_FIELDS:
        values = observation[name]
        if values is None:
            continue
        card_ids = (
            values[..., start].astype(jnp.int32) * 256
            + values[..., start + 1].astype(jnp.int32)
        )
        in_range = card_ids < id_lookup.shape[0]
        bounded_ids = jnp.clip(card_ids, 0, id_lookup.shape[0] - 1)
        mapped_ids = id_lookup[bounded_ids]
        mapped_ids = jnp.where(in_range, mapped_ids, 0)
        values = values.at[..., start].set(
            (mapped_ids >> 8).astype(values.dtype)
        )
        values = values.at[..., start + 1].set(
            (mapped_ids & 0xFF).astype(values.dtype)
        )
        remapped[name] = values
    return remapped


def zero_done_state(state, done: jax.Array):
    return jax.tree.map(
        lambda value: jnp.where(done[:, None], 0, value),
        state,
    )


def choose_updated_state(old_state, new_state, active: jax.Array):
    return jax.tree.map(
        lambda old, new: jnp.where(active[:, None], new, old),
        old_state,
        new_state,
    )


def build_predictor(
    official_agent: RNNAgent,
    official_variables,
    structured_agent: RNNAgent,
    structured_variables,
    official_id_lookup: np.ndarray,
):
    official_id_lookup = jax.device_put(official_id_lookup)

    @jax.jit
    def predict(
        official_state,
        structured_state,
        observation,
        structured_active,
        previous_done,
    ):
        official_state = zero_done_state(official_state, previous_done)
        structured_state = zero_done_state(
            structured_state,
            previous_done,
        )
        official_observation = remap_official_observation(
            observation,
            official_id_lookup,
        )
        next_official_state, official_logits = official_agent.apply(
            official_variables,
            official_observation,
            official_state,
        )[:2]
        next_structured_state, structured_logits = structured_agent.apply(
            structured_variables,
            observation,
            structured_state,
        )[:2]

        official_state = choose_updated_state(
            official_state,
            next_official_state,
            ~structured_active,
        )
        structured_state = choose_updated_state(
            structured_state,
            next_structured_state,
            structured_active,
        )
        logits = jnp.where(
            structured_active[:, None],
            structured_logits,
            official_logits,
        )
        finite = jnp.asarray(
            [
                jnp.all(jnp.isfinite(official_logits)),
                jnp.all(jnp.isfinite(structured_logits)),
                jnp.all(jnp.isfinite(logits)),
            ]
        )
        return (
            official_state,
            structured_state,
            logits,
            finite,
        )

    return predict


def jax_observation(
    observation: dict[str, np.ndarray | None],
) -> dict[str, jax.Array | None]:
    return jax.tree.map(
        lambda value: None if value is None else jnp.asarray(value),
        observation,
        is_leaf=lambda value: value is None,
    )


def observation_hash(observation: dict[str, np.ndarray | None]) -> str:
    digest = hashlib.sha256()
    for name in sorted(observation):
        value = observation[name]
        digest.update(name.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def describe_valid_unmapped_ids(
    observation: dict[str, np.ndarray | None],
    structured_active: np.ndarray,
    official_id_lookup: np.ndarray,
    current_codes: list[int],
) -> list[dict[str, Any]]:
    code_by_id = np.asarray([0] + current_codes, dtype=np.int64)
    details: list[dict[str, Any]] = []
    official_active = ~structured_active
    for name, start in ID_FIELDS:
        values = observation[name]
        if values is None:
            continue
        values = np.asarray(values)
        card_ids = (
            values[..., start].astype(np.int32) * 256
            + values[..., start + 1].astype(np.int32)
        )
        in_range = card_ids < official_id_lookup.shape[0]
        bounded = np.clip(card_ids, 0, official_id_lookup.shape[0] - 1)
        mapped = official_id_lookup[bounded]
        mapped = np.where(in_range, mapped, 0)
        if name == "cards_":
            valid = values[..., 2] != 0
        else:
            valid = values[..., 3] != 0
        active_shape = (official_active.shape[0],) + (1,) * (
            card_ids.ndim - 1
        )
        bad = (
            (card_ids > 0)
            & (mapped == 0)
            & valid
            & official_active.reshape(active_shape)
        )
        for index in np.argwhere(bad)[:20]:
            location = tuple(int(value) for value in index)
            current_id = int(card_ids[location])
            code = (
                int(code_by_id[current_id])
                if 0 <= current_id < code_by_id.shape[0]
                else None
            )
            details.append(
                {
                    "field": name,
                    "index": location,
                    "current_id": current_id,
                    "card_code": code,
                    "raw_row": values[location].astype(int).tolist(),
                }
            )
    return details


def count_unmapped_ids(
    observation: dict[str, np.ndarray | None],
    structured_active: np.ndarray,
    official_id_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    all_counts = np.zeros(len(ID_FIELDS), dtype=np.int64)
    valid_counts = np.zeros(len(ID_FIELDS), dtype=np.int64)
    official_active = ~structured_active
    for field_index, (name, start) in enumerate(ID_FIELDS):
        values = observation[name]
        if values is None:
            continue
        values = np.asarray(values)
        card_ids = (
            values[..., start].astype(np.int32) * 256
            + values[..., start + 1].astype(np.int32)
        )
        in_range = card_ids < official_id_lookup.shape[0]
        bounded = np.clip(card_ids, 0, official_id_lookup.shape[0] - 1)
        mapped = official_id_lookup[bounded]
        mapped = np.where(in_range, mapped, 0)
        unmapped = (card_ids > 0) & (mapped == 0)
        if name == "cards_":
            valid = values[..., 2] != 0
        else:
            valid = values[..., 3] != 0
        active_shape = (official_active.shape[0],) + (1,) * (
            card_ids.ndim - 1
        )
        active = official_active.reshape(active_shape)
        all_counts[field_index] = int(np.count_nonzero(unmapped & active))
        valid_counts[field_index] = int(
            np.count_nonzero(unmapped & valid & active)
        )
    return all_counts, valid_counts


def outcome_from_reward(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def select_actions(
    logits: jax.Array,
    policy: str,
    temperature: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if policy == "argmax":
        return jnp.argmax(logits, axis=-1), key
    key, subkey = jax.random.split(key)
    scaled_logits = logits / temperature
    uniform = jax.random.uniform(subkey, shape=scaled_logits.shape)
    gumbel = -jnp.log(-jnp.log(uniform))
    return jnp.argmax(scaled_logits + gumbel, axis=-1), key


def run_phase(
    args: argparse.Namespace,
    env,
    predictor,
    official_agent: RNNAgent,
    structured_agent: RNNAgent,
    official_id_lookup: np.ndarray,
    current_codes: list[int],
    structured_seat: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observation, info = env.reset()
    initial_hash = observation_hash(observation)
    next_to_play = np.asarray(info["to_play"], dtype=np.int32)
    previous_done = np.zeros(args.pairs, dtype=np.bool_)
    collected = np.zeros(args.pairs, dtype=np.bool_)
    records: list[dict[str, Any] | None] = [None] * args.pairs
    official_state = official_agent.init_rnn_state(args.pairs)
    structured_state = structured_agent.init_rnn_state(args.pairs)
    official_key = jax.random.PRNGKey(args.seed ^ 0x13579BDF)
    structured_key = jax.random.PRNGKey(args.seed ^ 0x2468ACE0)

    step = 0
    model_time = 0.0
    env_time = 0.0
    all_unmapped_official_ids = np.zeros(
        len(ID_FIELDS),
        dtype=np.int64,
    )
    valid_unmapped_official_ids = np.zeros(
        len(ID_FIELDS),
        dtype=np.int64,
    )
    started = time.time()
    next_progress = args.progress_every
    next_heartbeat = time.monotonic() + 20
    unknown_codes_seen: set[int] = set()
    audited_decisions = [0, 0]

    def publish_progress():
        if not args.progress_output:
            return
        partial = [record for record in records if record is not None]
        temporary = args.progress_output.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "seat": structured_seat, "completed": int(collected.sum()),
            "expected": args.pairs, "records": partial, "updated_at": time.time(),
            "vector_step": step, "remaining_environment_indices": np.flatnonzero(~collected).tolist(),
            "elapsed_seconds": time.time() - started,
            "model_seconds": model_time, "environment_seconds": env_time,
            "device": str(jax.devices()[0]),
        }))
        temporary.replace(args.progress_output)

    publish_progress()
    while not bool(np.all(collected)):
        structured_active = next_to_play == structured_seat
        all_unmapped_counts, valid_unmapped_counts = count_unmapped_ids(
            observation,
            structured_active,
            official_id_lookup,
        )
        if np.any(valid_unmapped_counts):
            details = {
                name: int(count)
                for (name, _), count in zip(
                    ID_FIELDS,
                    valid_unmapped_counts,
                )
                if count
            }
            unmapped_rows = describe_valid_unmapped_ids(
                observation,
                structured_active,
                official_id_lookup,
                current_codes,
            )
            unknown_codes_seen.update(row["card_code"] for row in unmapped_rows)
            if not args.allow_official_unknown:
                raise ValueError(
                    "official model encountered valid observation IDs absent "
                    f"from its historical code list at step {step}: {details}; "
                    f"rows={unmapped_rows}"
                )
        if np.any(~collected):
            observed = {name: None if value is None else value[~collected]
                        for name, value in observation.items()}
            raw_counts, valid_counts = count_unmapped_ids(
                observed, structured_active[~collected], official_id_lookup)
            all_unmapped_official_ids += raw_counts
            valid_unmapped_official_ids += valid_counts
            audited_decisions[0] += int(np.count_nonzero(structured_active & ~collected))
            audited_decisions[1] += int(np.count_nonzero(~structured_active & ~collected))

        started_model = time.time()
        (
            official_state,
            structured_state,
            logits,
            finite,
        ) = predictor(
            official_state,
            structured_state,
            jax_observation(observation),
            jnp.asarray(structured_active),
            jnp.asarray(previous_done),
        )
        official_actions, official_key = select_actions(
            logits,
            args.official_policy,
            args.temperature,
            official_key,
        )
        structured_actions, structured_key = select_actions(
            logits,
            args.structured_policy,
            args.temperature,
            structured_key,
        )
        actions = np.asarray(
            jnp.where(
                jnp.asarray(structured_active),
                structured_actions,
                official_actions,
            ),
            dtype=np.int32,
        )
        finite = np.asarray(finite)
        model_time += time.time() - started_model
        if not bool(np.all(finite)):
            raise FloatingPointError(
                "non-finite logits at phase "
                f"seat={structured_seat}, step={step}: "
                f"official={bool(finite[0])}, "
                f"structured={bool(finite[1])}, "
                f"selected={bool(finite[2])}"
            )
        terminal_to_play = next_to_play.copy()
        terminal_structured_active = structured_active.copy()
        started_env = time.time()
        observation, _, previous_done, info = env.step(actions)
        env_time += time.time() - started_env
        previous_done = np.asarray(previous_done, dtype=np.bool_)
        next_to_play = np.asarray(info["to_play"], dtype=np.int32)
        step += 1

        completed_indices = np.flatnonzero(previous_done & ~collected)
        for index in completed_indices:
            episode_return = float(np.asarray(info["r"])[index])
            structured_return = episode_return * (
                1.0 if terminal_structured_active[index] else -1.0
            )
            incomplete = (int(np.asarray(info["l"])[index]) >= args.max_steps and structured_return == 0)
            records[index] = {
                "environment_index": int(index),
                "structured_seat": int(structured_seat),
                "outcome": "incomplete" if incomplete else outcome_from_reward(structured_return),
                "structured_return": structured_return,
                "environment_return": episode_return,
                "length": int(np.asarray(info["l"])[index]),
                "win_reason": int(np.asarray(info["win_reason"])[index]),
                "terminal_to_play": int(terminal_to_play[index]),
                "truncated": (
                    int(np.asarray(info["l"])[index]) >= args.max_steps
                    and structured_return == 0
                ),
            }
            collected[index] = True

        completed = int(collected.sum())
        if args.progress_output and len(completed_indices):
            publish_progress()
        if args.progress_every > 0 and (
            completed >= next_progress or completed == args.pairs
        ):
            partial = [record for record in records if record is not None]
            counts = Counter(record["outcome"] for record in partial)
            print(
                "progress "
                f"seat={structured_seat} "
                f"{completed}/{args.pairs} "
                f"W-D-L={counts['win']}-{counts['draw']}-"
                f"{counts['loss']}",
                flush=True,
            )
            next_progress = completed + args.progress_every
        if time.monotonic() >= next_heartbeat:
            publish_progress()
            print(f"heartbeat seat={structured_seat} completed={completed}/{args.pairs} "
                  f"vector_step={step}", flush=True)
            next_heartbeat = time.monotonic() + 20

        if step > args.max_steps + 10:
            missing = np.flatnonzero(~collected).tolist()
            raise RuntimeError(
                f"phase exceeded max steps; unfinished environments: {missing}"
            )

    finalized = [record for record in records if record is not None]
    if len(finalized) != args.pairs:
        raise AssertionError(
            f"collected {len(finalized)} records, expected {args.pairs}"
        )
    timings = {
        "wall_seconds": time.time() - started,
        "model_seconds": model_time,
        "environment_seconds": env_time,
        "vector_steps": step,
        "environment_decisions_including_resets": step * args.pairs,
        "initial_observation_sha256": initial_hash,
        "all_unmapped_official_ids": {
            name: int(count)
            for (name, _), count in zip(
                ID_FIELDS,
                all_unmapped_official_ids,
            )
        },
        "valid_unmapped_official_ids": {
            name: int(count)
            for (name, _), count in zip(
                ID_FIELDS,
                valid_unmapped_official_ids,
            )
        },
        "unknown_codes_seen": sorted(unknown_codes_seen),
        "recording_enabled": False,
        "counted_decisions": {"expert": audited_decisions[0], "official": audited_decisions[1]},
    }
    return finalized, timings


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["outcome"] for record in records)
    wins = counts["win"]
    draws = counts["draw"]
    losses = counts["loss"]
    total = len(records)
    lengths = np.asarray(
        [record["length"] for record in records],
        dtype=np.float64,
    )
    decisive = wins + losses
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "strict_win_rate": wins / total,
        "strict_win_rate_wilson_95": wilson_interval(wins, total),
        "score_rate_draw_half": (wins + 0.5 * draws) / total,
        "decisive_win_rate": wins / decisive if decisive else None,
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "p95_length": float(np.percentile(lengths, 95)),
        "win_reasons": dict(
            sorted(
                Counter(
                    str(record["win_reason"]) for record in records
                ).items()
            )
        ),
    }


def paired_summary(
    first_records: list[dict[str, Any]],
    second_records: list[dict[str, Any]],
) -> dict[str, Any]:
    label = {"win": "W", "draw": "D", "loss": "L"}
    paired_outcomes = Counter(
        label[first["outcome"]] + label[second["outcome"]]
        for first, second in zip(first_records, second_records)
    )
    pair_scores = Counter()
    value = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    for first, second in zip(first_records, second_records):
        score = value[first["outcome"]] + value[second["outcome"]]
        pair_scores[f"{score:.1f}"] += 1
    return {
        "outcomes_first_then_second": dict(sorted(paired_outcomes.items())),
        "structured_points_across_two_games": dict(
            sorted(pair_scores.items())
        ),
    }


def print_summary(name: str, summary: dict[str, Any]) -> None:
    interval = summary["strict_win_rate_wilson_95"]
    interval_text = (
        "n/a"
        if interval is None
        else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
    )
    print(
        f"{name}: "
        f"W-D-L={summary['wins']}-{summary['draws']}-"
        f"{summary['losses']}, "
        f"win_rate={summary['strict_win_rate']:.4f}, "
        f"score={summary['score_rate_draw_half']:.4f}, "
        f"95%CI={interval_text}, "
        f"mean_length={summary['mean_length']:.2f}",
        flush=True,
    )


from .paired import summarize, paired_summary, print_summary


def main() -> None:
    args = parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    for path in (
        args.official_checkpoint,
        args.structured_checkpoint,
        args.deck,
        args.code_list,
        args.official_code_list,
        args.semantic_cache,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    stats = deck_index_stats(args.deck, args.code_list, args.official_code_list)
    if stats["missing_official_codes"] and not args.allow_official_unknown:
        raise ValueError(f"Unknown official cards: {stats['missing_official_codes']}")
    print(f"deck_compatibility={json.dumps(stats)}", flush=True)

    from jax.experimental.compilation_cache import compilation_cache as cc

    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))
    semantic_shape = load_semantic_shape(args.semantic_cache)
    current_codes = read_code_list(args.code_list)
    deck_name = init_ygopro(
        "YGOPro-v1",
        "english",
        str(args.deck),
        str(args.code_list),
    )

    first_env = make_env(args, deck_name)
    try:
        (
            official_agent,
            official_variables,
            structured_agent,
            structured_variables,
            official_max_id,
        ) = load_models(
            jax_observation({name: None if value is None else value[:1]
                             for name, value in first_env.reset()[0].items()}),
            args.official_checkpoint,
            args.structured_checkpoint,
            semantic_shape,
            args.seed,
        )
        official_id_lookup, official_mapping_stats = \
            build_official_id_lookup(
                args.code_list,
                args.official_code_list,
                official_max_id,
            )
        predictor = build_predictor(
            official_agent,
            official_variables,
            structured_agent,
            structured_variables,
            official_id_lookup,
        )
        first_env.close()
        first_env = make_env(args, deck_name)
        print(
            "loaded models: "
            f"official_max_id={official_max_id}, "
            f"official_code_list_rows="
            f"{official_mapping_stats['official_rows']}, "
            f"semantic_shape={semantic_shape}, "
            f"device={jax.devices()[0]}",
            flush=True,
        )
        first_records, first_timings = run_phase(
            args,
            first_env,
            predictor,
            official_agent,
            structured_agent,
            official_id_lookup,
            current_codes,
            structured_seat=0,
        )
    finally:
        first_env.close()
    first_checkpoint = args.output.with_name("phase-seat0.json")
    first_checkpoint.write_text(json.dumps({"records": first_records, "timings": first_timings}, indent=2))

    second_env = make_env(args, deck_name)
    try:
        second_records, second_timings = run_phase(
            args,
            second_env,
            predictor,
            official_agent,
            structured_agent,
            official_id_lookup,
            current_codes,
            structured_seat=1,
        )
    finally:
        second_env.close()
    args.output.with_name("phase-seat1.json").write_text(
        json.dumps({"records": second_records, "timings": second_timings}, indent=2))

    first_summary = summarize(first_records)
    second_summary = summarize(second_records)
    overall_summary = summarize(first_records + second_records)
    mirrored_initial_state = (
        first_timings["initial_observation_sha256"]
        == second_timings["initial_observation_sha256"]
    )

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "paired_seeds": args.pairs,
            "total_games": args.pairs * 2,
            "seed": args.seed,
            "deck_for_both_players": str(args.deck.resolve()),
            "deck_sha256": sha256_file(args.deck),
            "code_list": str(args.code_list.resolve()),
            "code_list_sha256": sha256_file(args.code_list),
            "official_code_list": str(
                args.official_code_list.resolve()
            ),
            "official_code_list_sha256": sha256_file(
                args.official_code_list
            ),
            "semantic_cache": str(args.semantic_cache.resolve()),
            "max_options": args.max_options,
            "max_steps": args.max_steps,
            "n_history_actions": args.n_history_actions,
            "policy": (
                "deterministic_argmax"
                if (
                    args.structured_policy == "argmax"
                    and args.official_policy == "argmax"
                )
                else (
                    f"structured={args.structured_policy},"
                    f"official={args.official_policy}"
                )
            ),
            "structured_policy": args.structured_policy,
            "official_policy": args.official_policy,
            "temperature": args.temperature,
            "recording_enabled": False,
            "official_id_compatibility": (
                "map current observation IDs by card code into the "
                "historical official code-list order; unknown cards map to 0"
            ),
            "official_max_card_id": official_max_id,
            "official_mapping_stats": official_mapping_stats,
            "official_unknown_cards_allowed": args.allow_official_unknown,
            "expert_checkpoint_metadata": json.loads(args.structured_checkpoint.with_name(
                args.structured_checkpoint.name + ".json").read_text()),
            "structured_semantic_shape": semantic_shape,
            "mirrored_initial_observations_equal": mirrored_initial_state,
        },
        "models": {
            "official": {
                "path": str(args.official_checkpoint.resolve()),
                "sha256": sha256_file(args.official_checkpoint),
            },
            "structured": {
                "path": str(args.structured_checkpoint.resolve()),
                "sha256": sha256_file(args.structured_checkpoint),
            },
        },
        "deck_index_stats": deck_index_stats(
            args.deck,
            args.code_list,
            args.official_code_list,
        ),
        "structured_model": {
            "overall": overall_summary,
            "going_first_player_0": first_summary,
            "going_second_player_1": second_summary,
        },
        "paired": paired_summary(first_records, second_records),
        "timings": {
            "structured_going_first": first_timings,
            "structured_going_second": second_timings,
        },
        "games": {
            "structured_going_first": first_records,
            "structured_going_second": second_records,
        },
    }

    print_summary("structured overall", overall_summary)
    print_summary("structured going first", first_summary)
    print_summary("structured going second", second_summary)
    print(
        "paired initial observations equal: "
        f"{mirrored_initial_state}",
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
