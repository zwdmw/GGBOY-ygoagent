#!/usr/bin/env python3
"""Milestone evaluation of a structured checkpoint against a FROZEN structured
baseline (the 259M checkpoint), replacing the official ID-embedding opponent.

Why a separate evaluator: `evaluate_frozen.py` hard-codes the opponent slot as
`RNNAgent(structured=False)` (the official ID-embedding architecture) and remaps
observations through the official code list, so it cannot play a decision-v1
checkpoint as the opponent. This script keeps that evaluator's game loop,
seat-reversal pairing, seed scheme, outcome bookkeeping and Wilson intervals, but
loads BOTH sides as structured decision-v1 agents.

Output schema matches `evaluate_frozen.py` exactly (`structured_model`,
`models`, `configuration`, `paired`, `games`, `timings`), with
``models.official`` holding the frozen structured baseline. That keeps the
evaluation queue's assertions and the dashboard readers working unchanged.
"""
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
from ygoai.utils import init_ygopro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a structured checkpoint against a frozen structured baseline.")
    # kept named --official-checkpoint for drop-in compatibility with the queue
    parser.add_argument("--official-checkpoint", type=Path, required=True,
                        help="frozen structured baseline (the opponent)")
    parser.add_argument("--structured-checkpoint", type=Path, required=True,
                        help="checkpoint under test")
    parser.add_argument("--deck", type=Path, default=Path("assets/deck/BlueEyesGPU.ydk"))
    parser.add_argument("--code-list", type=Path, default=Path("scripts/code_list.txt"))
    parser.add_argument("--official-code-list", type=Path,
                        default=Path("scripts/official_code_list_864.txt"))
    parser.add_argument("--semantic-cache", type=Path,
                        default=Path("assets/structured/frozen_semantics_v1.npz"))
    parser.add_argument("--pairs", type=int, default=32,
                        help="paired seeds; total games are twice this value")
    parser.add_argument("--seed", type=int, default=2026092094)
    parser.add_argument("--env-threads", type=int, default=2)
    parser.add_argument("--max-options", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-history-actions", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--allow-official-unknown", action="store_true")
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--structured-policy", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--official-policy", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("baseline-vs-result.json"))
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
            raise ValueError(f"semantic mask {lua_mask.shape} != effects {lua_effects.shape}")
        if cdb_exact.shape[0] != lua_effects.shape[0]:
            raise ValueError("semantic row mismatch")
        return (int(cdb_exact.shape[0]), int(cdb_exact.shape[1]),
                int(lua_effects.shape[1]), int(lua_effects.shape[2]))


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


def deck_index_stats(deck_path: Path, current_code_list_path: Path,
                     official_code_list_path: Path) -> dict[str, Any]:
    current = {c: i for i, c in enumerate(read_code_list(current_code_list_path), start=1)}
    official = {c: i for i, c in enumerate(read_code_list(official_code_list_path), start=1)}
    deck_codes = read_deck_codes(deck_path)
    missing_current = sorted(set(deck_codes).difference(current))
    missing_official = sorted(set(deck_codes).difference(official))
    if missing_current:
        raise ValueError(f"deck cards missing from current code list: {missing_current}")
    current_ids = [current[c] for c in deck_codes]
    official_ids = [official.get(c, 0) for c in deck_codes]
    return {
        "copies": len(deck_codes), "unique_cards": len(set(deck_codes)),
        "min_current_id": min(current_ids), "max_current_id": max(current_ids),
        "min_official_id": min(official_ids), "max_official_id": max(official_ids),
        "missing_current_codes": missing_current,
        "missing_official_codes": missing_official,
        "missing_official_copies": sum(c in missing_official for c in deck_codes),
    }


def make_env(args, deck_name: str):
    base = ygoenv.make(
        task_id="YGOPro-v1", env_type="gymnasium",
        num_envs=args.pairs, num_threads=min(args.env_threads, args.pairs),
        seed=args.seed, player=-1, deck1=deck_name, deck2=deck_name,
        max_options=args.max_options, max_steps=args.max_steps,
        n_history_actions=args.n_history_actions, async_reset=False,
        greedy_reward=False, play_mode="self", timeout=args.timeout,
        oppo_info=False, record=False, thread_affinity_offset=-1,
        deck_schedule="independent", anchor_probability_percent=0,
    )
    base.num_envs = args.pairs
    return RecordEpisodeStatistics(EnvPreprocess(base, skip_mask=True))


def build_structured(name, checkpoint: Path, sample_observation, semantic_shape,
                     tables, key):
    sidecar_path = checkpoint.with_name(checkpoint.name + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing sidecar for {name}: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar["schema"] != "ygo-training-checkpoint/v2":
        raise ValueError(f"{name}: unsupported checkpoint metadata")
    if sidecar.get("architecture") != "decision-v1" or sha256_file(checkpoint) != sidecar["sha256"]:
        raise ValueError(f"{name}: architecture or checksum mismatch")
    if tuple(sidecar["semantic_shape"]) != tuple(semantic_shape):
        raise ValueError(f"{name}: semantic shape mismatch")
    agent = RNNAgent(**sidecar["model_args"], semantic_shape=semantic_shape,
                     dtype=jnp.bfloat16 if sidecar["bfloat16"] else jnp.float32,
                     switch=sidecar["switch"])
    variables = agent.init(key, sample_observation, agent.init_rnn_state(1))
    variables, added = restore_decision_checkpoint(variables, checkpoint.read_bytes())
    if added:
        raise ValueError(f"{name}: partial decision checkpoint ({added})")
    variables = flax.core.unfreeze(variables)
    inject_semantic_constants(variables, tables)
    print(f"loaded {name}: architecture={sidecar['architecture']} "
          f"step={sidecar['global_step']} sha={sidecar['sha256'][:12]}", flush=True)
    return agent, jax.device_put(flax.core.freeze(variables)), sidecar


def build_predictor(official_agent, official_variables, structured_agent,
                    structured_variables, shared_model=False):
    @jax.jit
    def predict(official_state, structured_state, observation,
                structured_active, previous_done):
        def zero(state):
            return jax.tree.map(lambda v: jnp.where(previous_done[:, None], 0, v), state)
        official_state = zero(official_state)
        structured_state = zero(structured_state)
        if shared_model:
            active_state = jax.tree.map(
                lambda official, structured: jnp.where(
                    structured_active[:, None], structured, official
                ),
                official_state, structured_state,
            )
            next_active_state, logits = official_agent.apply(
                official_variables, observation, active_state
            )[:2]
            official_state = jax.tree.map(
                lambda old, new: jnp.where(
                    structured_active[:, None], old, new
                ),
                official_state, next_active_state,
            )
            structured_state = jax.tree.map(
                lambda old, new: jnp.where(
                    structured_active[:, None], new, old
                ),
                structured_state, next_active_state,
            )
            finite_logits = jnp.all(jnp.isfinite(logits))
            finite = jnp.asarray([finite_logits, finite_logits, finite_logits])
            return official_state, structured_state, logits, finite
        # Both sides are structured decision-v1, so both consume the raw
        # observation; there is no official ID remapping in this evaluator.
        next_official_state, official_logits = official_agent.apply(
            official_variables, observation, official_state)[:2]
        next_structured_state, structured_logits = structured_agent.apply(
            structured_variables, observation, structured_state)[:2]
        official_state = jax.tree.map(
            lambda old, new: jnp.where(structured_active[:, None], old, new),
            official_state, next_official_state)
        structured_state = jax.tree.map(
            lambda old, new: jnp.where(structured_active[:, None], new, old),
            structured_state, next_structured_state)
        logits = jnp.where(structured_active[:, None], structured_logits, official_logits)
        finite = jnp.asarray([
            jnp.all(jnp.isfinite(official_logits)),
            jnp.all(jnp.isfinite(structured_logits)),
            jnp.all(jnp.isfinite(logits)),
        ])
        return official_state, structured_state, logits, finite
    return predict


def jax_observation(observation):
    return jax.tree.map(lambda v: None if v is None else jnp.asarray(v), observation,
                        is_leaf=lambda v: v is None)


def observation_hash(observation) -> str:
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


def outcome_from_reward(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def select_actions(logits, policy: str, temperature: float, key):
    if policy == "argmax":
        return jnp.argmax(logits, axis=-1), key
    key, subkey = jax.random.split(key)
    scaled = logits / temperature
    uniform = jax.random.uniform(subkey, shape=scaled.shape)
    return jnp.argmax(scaled - jnp.log(-jnp.log(uniform)), axis=-1), key


def run_phase(args, env, predictor, official_agent, structured_agent, structured_seat):
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
    model_time = env_time = 0.0
    started = time.time()
    next_progress = args.progress_every
    next_heartbeat = time.monotonic() + 20

    def publish_progress():
        if not args.progress_output:
            return
        partial = [r for r in records if r is not None]
        temporary = args.progress_output.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "seat": structured_seat, "completed": int(collected.sum()),
            "expected": args.pairs, "records": partial, "updated_at": time.time(),
            "vector_step": step,
            "remaining_environment_indices": np.flatnonzero(~collected).tolist(),
            "elapsed_seconds": time.time() - started,
            "model_seconds": model_time, "environment_seconds": env_time,
            "device": str(jax.devices()[0]), "opponent": "structured-baseline",
        }))
        temporary.replace(args.progress_output)

    publish_progress()
    while not bool(np.all(collected)):
        structured_active = next_to_play == structured_seat
        started_model = time.time()
        official_state, structured_state, logits, finite = predictor(
            official_state, structured_state, jax_observation(observation),
            jnp.asarray(structured_active), jnp.asarray(previous_done))
        official_actions, official_key = select_actions(
            logits, args.official_policy, args.temperature, official_key)
        structured_actions, structured_key = select_actions(
            logits, args.structured_policy, args.temperature, structured_key)
        actions = np.asarray(jnp.where(jnp.asarray(structured_active),
                                       structured_actions, official_actions), dtype=np.int32)
        finite = np.asarray(finite)
        model_time += time.time() - started_model
        if not bool(np.all(finite)):
            raise FloatingPointError(f"non-finite logits at seat={structured_seat} step={step}")
        terminal_structured_active = structured_active.copy()
        started_env = time.time()
        observation, _, previous_done, info = env.step(actions)
        env_time += time.time() - started_env
        previous_done = np.asarray(previous_done, dtype=np.bool_)
        next_to_play = np.asarray(info["to_play"], dtype=np.int32)
        step += 1
        for index in np.flatnonzero(previous_done & ~collected):
            episode_return = float(np.asarray(info["r"])[index])
            structured_return = episode_return * (
                1.0 if terminal_structured_active[index] else -1.0)
            incomplete = (int(np.asarray(info["l"])[index]) >= args.max_steps and structured_return == 0)
            records[index] = {
                "environment_index": int(index),
                "structured_seat": int(structured_seat),
                "outcome": "incomplete" if incomplete else outcome_from_reward(structured_return),
                "structured_return": structured_return,
                "environment_return": episode_return,
                "length": int(np.asarray(info["l"])[index]),
                "win_reason": int(np.asarray(info["win_reason"])[index]),
                "terminal_to_play": int(next_to_play[index]),
                "truncated": incomplete,
            }
            collected[index] = True
        completed = int(collected.sum())
        if args.progress_output and len(np.flatnonzero(previous_done & ~collected)) >= 0:
            publish_progress()
        if args.progress_every > 0 and (completed >= next_progress or completed == args.pairs):
            partial = [r for r in records if r is not None]
            counts = Counter(r["outcome"] for r in partial)
            print(f"progress seat={structured_seat} {completed}/{args.pairs} "
                  f"W-D-L={counts['win']}-{counts['draw']}-{counts['loss']}", flush=True)
            next_progress = completed + args.progress_every
        if time.monotonic() >= next_heartbeat:
            publish_progress()
            print(f"heartbeat seat={structured_seat} completed={completed}/{args.pairs} "
                  f"vector_step={step}", flush=True)
            next_heartbeat = time.monotonic() + 20
        if step > args.max_steps + 10:
            raise RuntimeError(f"phase exceeded max steps; unfinished: "
                               f"{np.flatnonzero(~collected).tolist()}")
    finalized = [r for r in records if r is not None]
    if len(finalized) != args.pairs:
        raise AssertionError(f"collected {len(finalized)}, expected {args.pairs}")
    return finalized, {
        "wall_seconds": time.time() - started, "model_seconds": model_time,
        "environment_seconds": env_time, "vector_steps": step,
        "environment_decisions_including_resets": step * args.pairs,
        "initial_observation_sha256": initial_hash,
        "recording_enabled": False, "opponent_architecture": "decision-v1",
    }


def wilson_interval(successes: int, total: int):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [centre - radius, centre + radius]


def summarize(records) -> dict[str, Any]:
    completed = [r for r in records if r["outcome"] != "incomplete"]
    counts = Counter(r["outcome"] for r in completed)
    wins, draws, losses = counts["win"], counts["draw"], counts["loss"]
    total = len(completed)
    lengths = np.asarray([r["length"] for r in records], dtype=np.float64)
    decisive = wins + losses
    return {
        "games": len(records), "completed": total, "incomplete": len(records) - total,
        "wins": wins, "draws": draws, "losses": losses,
        "strict_win_rate": wins / total if total else None,
        "strict_win_rate_wilson_95": wilson_interval(wins, total),
        "score_rate_draw_half": (wins + 0.5 * draws) / total if total else None,
        "decisive_win_rate": wins / decisive if decisive else None,
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "p95_length": float(np.percentile(lengths, 95)),
        "win_reasons": dict(sorted(Counter(str(r["win_reason"]) for r in records).items())),
    }


def paired_summary(first_records, second_records) -> dict[str, Any]:
    complete_pairs = [(a, b) for a, b in zip(first_records, second_records)
                      if a["outcome"] != "incomplete" and b["outcome"] != "incomplete"]
    label = {"win": "W", "draw": "D", "loss": "L"}
    paired = Counter(label[a["outcome"]] + label[b["outcome"]]
                     for a, b in complete_pairs)
    scores = Counter()
    value = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    differences = []
    for a, b in complete_pairs:
        scores[f"{value[a['outcome']] + value[b['outcome']]:.1f}"] += 1
        differences.append(value[a["outcome"]] + value[b["outcome"]] - 1.0)
    mean = float(np.mean(differences)) if differences else None
    radius = 1.96 * float(np.std(differences, ddof=1)) / math.sqrt(len(differences)) if len(differences) > 1 else None
    return {"completed_pairs": len(complete_pairs), "paired_score_difference": mean,
            "paired_difference_normal_95": [mean - radius, mean + radius] if radius is not None else None,
            "outcomes_first_then_second": dict(sorted(paired.items())),
            "structured_points_across_two_games": dict(sorted(scores.items()))}


def print_summary(name: str, summary: dict[str, Any]) -> None:
    ci = summary["strict_win_rate_wilson_95"]
    text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    print(f"{name}: W-D-L={summary['wins']}-{summary['draws']}-{summary['losses']}, "
          f"win_rate={summary['strict_win_rate']}, "
          f"score={summary['score_rate_draw_half']}, incomplete={summary['incomplete']}, 95%CI={text}, "
          f"mean_length={summary['mean_length']:.2f}", flush=True)


def main() -> None:
    args = parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    for path in (args.official_checkpoint, args.structured_checkpoint, args.deck,
                 args.code_list, args.official_code_list, args.semantic_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    stats = deck_index_stats(args.deck, args.code_list, args.official_code_list)
    print(f"deck_compatibility={json.dumps(stats)}", flush=True)

    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))
    semantic_shape = load_semantic_shape(args.semantic_cache)
    deck_name = init_ygopro("YGOPro-v1", "english", str(args.deck), str(args.code_list))
    tables, checked_shape, _ = load_structured_semantics(
        str(args.semantic_cache), str(args.code_list))
    assert tuple(checked_shape) == tuple(semantic_shape)

    probe = make_env(args, deck_name)
    try:
        sample_observation = jax_observation(
            {k: None if v is None else v[:1] for k, v in probe.reset()[0].items()})
    finally:
        probe.close()

    official_sha256 = sha256_file(args.official_checkpoint)
    structured_sha256 = sha256_file(args.structured_checkpoint)
    shared_model = official_sha256 == structured_sha256
    official_agent, official_variables, official_sidecar = build_structured(
        "baseline(official slot)", args.official_checkpoint, sample_observation,
        semantic_shape, tables, jax.random.PRNGKey(args.seed))
    if shared_model:
        structured_agent = official_agent
        structured_variables = official_variables
        structured_sidecar = official_sidecar
        print("shared identical-model inference path enabled", flush=True)
    else:
        structured_agent, structured_variables, structured_sidecar = build_structured(
            "structured", args.structured_checkpoint, sample_observation,
            semantic_shape, tables, jax.random.PRNGKey(args.seed + 1))
    predictor = build_predictor(official_agent, official_variables,
                                structured_agent, structured_variables,
                                shared_model=shared_model)

    first_env = make_env(args, deck_name)
    try:
        first_records, first_timings = run_phase(
            args, first_env, predictor, official_agent, structured_agent, structured_seat=0)
    finally:
        first_env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("phase-seat0.json").write_text(
        json.dumps({"records": first_records, "timings": first_timings}, indent=2))
    second_env = make_env(args, deck_name)
    try:
        second_records, second_timings = run_phase(
            args, second_env, predictor, official_agent, structured_agent, structured_seat=1)
    finally:
        second_env.close()
    args.output.with_name("phase-seat1.json").write_text(
        json.dumps({"records": second_records, "timings": second_timings}, indent=2))

    first_summary = summarize(first_records)
    second_summary = summarize(second_records)
    overall_summary = summarize(first_records + second_records)
    mirrored = (first_timings["initial_observation_sha256"]
                == second_timings["initial_observation_sha256"])

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "paired_seeds": args.pairs, "total_games": args.pairs * 2, "seed": args.seed,
            "deck_for_both_players": str(args.deck.resolve()),
            "deck_sha256": sha256_file(args.deck),
            "code_list": str(args.code_list.resolve()),
            "code_list_sha256": sha256_file(args.code_list),
            "official_code_list": str(args.official_code_list.resolve()),
            "official_code_list_sha256": sha256_file(args.official_code_list),
            "semantic_cache": str(args.semantic_cache.resolve()),
            "max_options": args.max_options, "max_steps": args.max_steps,
            "n_history_actions": args.n_history_actions,
            "structured_policy": args.structured_policy,
            "official_policy": args.official_policy,
            "temperature": args.temperature, "recording_enabled": False,
            "opponent_kind": "frozen-structured-baseline",
            "opponent_note": "Both sides are decision-v1; no official ID remapping.",
            "shared_identical_model_inference": shared_model,
            "expert_checkpoint_metadata": json.loads(
                args.structured_checkpoint.with_name(
                    args.structured_checkpoint.name + ".json").read_text()),
            "structured_semantic_shape": semantic_shape,
            "mirrored_initial_observations_equal": mirrored,
        },
        "models": {
            "official": {"path": str(args.official_checkpoint.resolve()),
                         "sha256": official_sha256,
                         "architecture": official_sidecar["architecture"],
                         "global_step": official_sidecar["global_step"]},
            "structured": {"path": str(args.structured_checkpoint.resolve()),
                           "sha256": structured_sha256},
        },
        "deck_index_stats": deck_index_stats(args.deck, args.code_list, args.official_code_list),
        # `structured` is the checkpoint under test; outcomes are its perspective.
        "structured_model": {
            "overall": overall_summary,
            "going_first_player_0": first_summary,
            "going_second_player_1": second_summary,
        },
        "paired": paired_summary(first_records, second_records),
        "timings": {"structured_going_first": first_timings,
                    "structured_going_second": second_timings},
        "games": {"structured_going_first": first_records,
                  "structured_going_second": second_records},
    }

    print_summary("structured overall", overall_summary)
    print_summary("structured going first", first_summary)
    print_summary("structured going second", second_summary)
    print(f"paired initial observations equal: {mirrored}", flush=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
