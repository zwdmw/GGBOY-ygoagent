#!/usr/bin/env python3
"""Native observation versus unchanged deployed network encoder.

Uses an isolated native TCP bridge, not the Aliyun WASM engine or transport.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import sys
import threading
import time
import uuid

DEFAULT_EXPERIMENT = Path(__file__).resolve().parents[1]
DEFAULT_ROOM = DEFAULT_EXPERIMENT


def atomic_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def peer_filename(batch_offset, native_seat, port):
    return f"peer-{batch_offset}-{native_seat}-{port}-{uuid.uuid4().hex}.json"


def audit_peers(games, peers):
    errors = [peer for peer in peers if "error" in peer]
    if errors or len(peers) != len(games):
        raise RuntimeError(f"Peer audit failed: {len(errors)} errors; {len(peers)} peers / {len(games)} games")
    expected = Counter((game["batch_seed"], game["network_seat"], game["network_outcome"])
                       for game in games)
    actual = Counter((peer["batch_seed"], peer["network_seat"],
                      "win" if peer["result"]["won"] is True else
                      "loss" if peer["result"]["won"] is False else "draw") for peer in peers)
    if actual != expected:
        raise ValueError(f"Native/network batch-seat outcomes disagree: {expected} vs {actual}")
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--room-root", type=Path, default=DEFAULT_ROOM)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--parity-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--env-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1468023700)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--trace-control", action="store_true")
    parser.add_argument("--replay-control", type=Path)
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args()


def peer_main():
    options = dict(argument.split("=", 1) for argument in sys.argv[1:] if "=" in argument)
    root = Path(os.environ["INPUT_EVAL_ROOM"])
    sys.path.insert(0, str(root / "src"))
    from ygo_sky.duel.network.client import ExternalClient
    from ygo_sky.duel.network.cards import CardPool
    from ygo_sky.duel.network.structured import StructuredEncoder
    from ygo_sky.duel.network.vendor.netduel.cards import load_ydk
    from ygo_sky.evaluation.ipc import encode_observation, receive_frame, send_frame

    port = int(options["Port"])
    output = Path(os.environ["INPUT_EVAL_OUTPUT"])
    pool = CardPool(options["DbPath"], os.environ["INPUT_EVAL_CODES"])
    main, extra, side = load_ydk(options["DeckFile"])
    encoder = StructuredEncoder(pool)
    rpc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    rpc.settimeout(120)
    rpc.connect(os.environ["INPUT_EVAL_SOCKET"])
    batch_offset = int(os.environ["INPUT_EVAL_BATCH_OFFSET"])
    native_seat = int(os.environ["INPUT_EVAL_NATIVE_SEAT"])
    stats = {"port": port, "source": "unchanged_deployed_network_encoder", "prompts": {},
             "batch_offset": batch_offset, "batch_seed": int(os.environ["INPUT_EVAL_BATCH_SEED"]),
             "network_seat": 1 - native_seat}

    class Policy:
        def bind(self, bound):
            self.client = bound
            if bound.result.our_player != stats["network_seat"]:
                raise ValueError("Network peer was assigned the wrong seat")
            encoder.reset(bound.result.our_player, int(options["Seed"]))

        def observe_game_message(self, msg, body):
            bound = self.client
            encoder.observe_message(msg, body, bound.result.turns, bound.ctx.current_phase)

        def choose(self, state):
            observation = encoder.encode_state(state, self.client.selector)
            send_frame(rpc, encode_observation(state.our_player, state.n, observation))
            result = json.loads(receive_frame(rpc))
            if "error" in result:
                raise RuntimeError(result["error"])
            index = result["action"]
            if type(index) is not int or not 0 <= index < state.n:
                raise ValueError("Invalid inference action")
            key = str(state.msg)
            stats["prompts"][key] = stats["prompts"].get(key, 0) + 1
            return index

        def record(self, *arguments):
            encoder.record(*arguments)

    client = ExternalClient(
        "127.0.0.1", port, "InputEvalNetwork", main, extra, Policy(),
        side=side, seed=int(options["Seed"]), preference="first", card_pool=pool,
        max_options=128, version=0x133E, require_deck_info=True)
    try:
        client.connect()
        while not client.terminal_seen:
            opcode, payload = client.stream.recv(timeout=120)
            client._handle(opcode, payload)
        stats.update(result=client.result.as_dict(), initial_counts=client.initial_deck_counts,
                     row_overflows=encoder.row_overflows,
                     deck_disagreements=encoder.deck_disagreements)
    except Exception as error:
        stats["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(output / peer_filename(batch_offset, native_seat, port), stats)
        client.close()
        rpc.close()


def summarize(games):
    counts = Counter(game["network_outcome"] for game in games)
    total = len(games)
    wins, losses, draws = (counts[key] for key in ("win", "loss", "draw"))
    decisive = wins + losses
    interval = None
    if decisive:
        proportion, zscore = wins / decisive, 1.959963984540054
        denominator = 1 + zscore ** 2 / decisive
        centre = (proportion + zscore ** 2 / (2 * decisive)) / denominator
        radius = zscore * (proportion * (1 - proportion) / decisive +
                          zscore ** 2 / (4 * decisive ** 2)) ** 0.5 / denominator
        interval = [centre - radius, centre + radius]
    return {"games": total, "network_wins": wins, "native_wins": losses,
            "draws": draws, "truncated": sum(game["truncated"] for game in games),
            "network_strict_win_rate": wins / total if total else None,
            "native_strict_win_rate": losses / total if total else None,
            "network_decisive_win_rate": wins / decisive if decisive else None,
            "network_decisive_wilson_95": interval,
            "network_score_rate": (wins + draws / 2) / total if total else None}


def main():
    args = parse_args()
    for name in ("experiment", "project_root", "room_root", "checkpoint", "native_checkpoint",
                 "deck", "native_source", "parity_proof", "output", "replay_control"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    if args.pairs < 1 or not 1 <= args.batch_size <= 32 or args.env_threads < 1:
        raise ValueError("Invalid pair count or batch size")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.trace_control:
        if not args.control:
            raise ValueError("Tensor traces are restricted to isolated native controls")
        (args.output / "trace").mkdir()
    if args.replay_control and not args.control:
        raise ValueError("Action-prefix replay is restricted to isolated native controls")
    project = args.project_root or args.experiment / "resources/game"
    native_source = args.native_source or args.experiment / "native/ygoenv"
    checkpoint = args.checkpoint or args.experiment / "resources/models/base-463001600.flax_model"
    native_checkpoint = args.native_checkpoint or checkpoint
    if args.control and sha256(checkpoint) != sha256(native_checkpoint):
        raise ValueError("Native control requires the same checkpoint on both seats")
    deck = args.deck or args.experiment / "resources/decks/sky-striker.ydk"
    os.environ.update(YGO_EVALUATION_EXPERIMENT=str(args.experiment),
                      YGO_PROJECT_ROOT=str(project), YGOENV_SOURCE_ROOT=str(native_source),
                      XLA_PYTHON_CLIENT_PREALLOCATE="false")
    sys.path[:0] = [str(native_source), str(args.experiment / "src")]
    os.environ.update(YGO_SKY_HOME=str(args.experiment), YGO_GAME_ROOT=str(project))
    import numpy as np
    import ygoenv
    if args.inspect:
        print(ygoenv.__file__, flush=True)
        print(ygoenv.make_spec(task_id="YGOPro-v1").config, flush=True)
        return
    import jax
    import jax.numpy as jnp
    from jax.experimental.compilation_cache import compilation_cache
    from ygoai.rl.env import EnvPreprocess
    from ygoai.utils import init_ygopro
    from ygo_sky.evaluation.paired import build_structured, load_semantic_shape
    from ygo_sky.semantics import load_structured_semantics
    from ygo_sky.evaluation.ipc import (decode_observation, observation_schema, receive_frame,
                                     send_frame, validate_observation)

    if jax.default_backend() != "gpu":
        raise RuntimeError("This evaluation requires GPU inference")
    compilation_cache.set_cache_dir(str(args.experiment / "jax-cache"))
    os.chdir(project)
    code_list = project / "scripts/code_list.txt"
    cards_db = project / "assets/locale/en/cards.cdb"
    semantic_cache = project / "assets/structured/frozen_semantics_v1.npz"
    assets = {str(path.relative_to(args.experiment)): sha256(path) for path in
              (code_list, cards_db, semantic_cache, args.experiment / "src/ygoai/rl/jax/agent.py")}
    deck_name = init_ygopro("YGOPro-v1", "english", str(deck), str(code_list))
    common = dict(task_id="YGOPro-v1", env_type="gymnasium",
                  num_threads=min(args.env_threads, args.batch_size),
                  thread_affinity_offset=-1, deck1=deck_name, deck2=deck_name,
                  deck_schedule="independent", anchor_probability_percent=0,
                  max_cards=80, max_options=128, n_history_actions=32,
                  max_steps=args.max_steps, async_reset=False, greedy_reward=False,
                  timeout=120, oppo_info=False, record=False)
    probe_base = ygoenv.make(**{**common, "num_threads": 1},
                            num_envs=1, seed=args.seed, player=-1, play_mode="self")
    probe_base.num_envs = 1
    probe = EnvPreprocess(probe_base, skip_mask=True)
    try:
        sample = probe.reset()[0]
    finally:
        probe.close()
    tables, semantic_shape, _ = load_structured_semantics(str(semantic_cache), str(code_list))
    if tuple(semantic_shape) != tuple(load_semantic_shape(semantic_cache)):
        raise ValueError("Semantic table shape mismatch")
    agent, variables, sidecar = build_structured(
        "network", checkpoint, jax.tree.map(jnp.asarray, sample),
        semantic_shape, tables, jax.random.PRNGKey(args.seed))
    if sha256(native_checkpoint) == sha256(checkpoint):
        native_agent, native_variables, native_sidecar = agent, variables, sidecar
    else:
        native_agent, native_variables, native_sidecar = build_structured(
            "native", native_checkpoint, jax.tree.map(jnp.asarray, sample),
            semantic_shape, tables, jax.random.PRNGKey(args.seed + 1))
    native_path = Path(ygoenv.__file__).parent / "ygopro/ygopro_ygoenv.cpython-311-x86_64-linux-gnu.so"
    binary_matches = sha256(native_path) == native_sidecar["native_module_sha256"]
    if not binary_matches and not args.control:
        if not args.parity_proof:
            raise ValueError("Different native binary requires a tensor parity proof")
        proof = json.loads(args.parity_proof.read_text())
        if not (proof["passed"] and proof["training_native_sha256"] == native_sidecar["native_module_sha256"]
                and proof["bridge_native_sha256"] == sha256(native_path)
                and proof["checkpoint_sha256"] == sha256(native_checkpoint)
                and proof["deck_sha256"] == sha256(deck)):
            raise ValueError("Tensor parity proof does not match evaluation assets")
        if "assets" in proof and proof["assets"] != assets:
            raise ValueError("Tensor parity proof was generated with different assets")
    if (sha256(checkpoint) != sidecar["sha256"] or
            sha256(native_checkpoint) != native_sidecar["sha256"]):
        raise ValueError("Checkpoint hash mismatch")
    model_lock = threading.Lock()

    def batch_observations(observations):
        padded = observations + [sample] * (args.batch_size - len(observations))
        return {key: None if sample[key] is None else np.concatenate(
            [observation[key] for observation in padded], axis=0) for key in sample}

    def build_predictor(model, model_variables):
        @jax.jit
        def infer(states, observation):
            next_states, logits = model.apply(model_variables, observation, states)[:2]
            return next_states, jnp.argmax(logits, axis=-1), jnp.all(jnp.isfinite(logits))

        def predict(states, observations):
            with model_lock:
                next_states, actions, finite = jax.device_get(
                    infer(states, batch_observations(observations)))
            if not bool(finite):
                raise FloatingPointError("Non-finite policy logits")
            return next_states, actions
        return predict

    predict = build_predictor(agent, variables)
    native_predict = (predict if native_agent is agent else
                      build_predictor(native_agent, native_variables))
    blank_states = jax.device_get(native_agent.init_rnn_state(args.batch_size))
    schema = observation_schema(sample)
    predict(jax.device_get(agent.init_rnn_state(args.batch_size)), [sample])
    if native_predict is not predict:
        native_predict(blank_states, [sample])
    stop = threading.Event()
    requests = queue.Queue()
    threads = []
    address = Path(f"/tmp/ygo-input-eval-{os.getpid()}.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(address))
    os.chmod(address, 0o600)
    listener.listen(64)
    listener.settimeout(0.5)

    def serve_connection(connection):
        state = jax.device_get(agent.init_rnn_state(1))
        try:
            while not stop.is_set():
                actor, count, observation = decode_observation(receive_frame(connection))
                validate_observation(observation, schema)
                future = queue.Queue(maxsize=1)
                requests.put((state, observation, count, future))
                state, result = future.get(timeout=120)
                send_frame(connection, json.dumps(result).encode("ascii"))
        except (EOFError, ConnectionError, OSError, queue.Empty):
            pass
        finally:
            connection.close()

    def accept_connections():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            connection.settimeout(120)
            thread = threading.Thread(target=serve_connection, args=(connection,), daemon=True)
            threads.append(thread)
            thread.start()

    def inference_loop():
        while not stop.is_set():
            try:
                first = requests.get(timeout=0.5)
            except queue.Empty:
                continue
            pending = [first]
            deadline = time.monotonic() + 0.003
            while len(pending) < args.batch_size:
                try:
                    pending.append(requests.get(timeout=max(0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            states = jax.tree.map(lambda *values: np.concatenate(values),
                                  *[request[0] for request in pending],
                                  *[agent.init_rnn_state(1) for _ in range(args.batch_size - len(pending))])
            try:
                updated, actions = predict(states, [request[1] for request in pending])
                for index, (_, _, count, future) in enumerate(pending):
                    action = int(actions[index])
                    result = ({"action": action} if 0 <= action < count else
                              {"error": "GPU returned action outside legal range"})
                    future.put((jax.tree.map(lambda value: value[index:index + 1], updated), result))
            except Exception as error:
                for state, _, _, future in pending:
                    future.put((state, {"error": f"{type(error).__name__}: {error}"}))

    for target in (accept_connections, inference_loop):
        thread = threading.Thread(target=target, daemon=True)
        threads.append(thread)
        thread.start()
    binary_dir = args.output / "bin"
    binary_dir.mkdir()
    launcher = binary_dir / "mono"
    launcher.write_text("#!/bin/sh\nexec " + sys.executable + ' "$@"\n')
    launcher.chmod(0o700)
    os.environ.update(INPUT_EVAL_ROOM=str(args.room_root), INPUT_EVAL_OUTPUT=str(args.output),
                      INPUT_EVAL_SOCKET=str(address), INPUT_EVAL_CODES=str(code_list),
                      PATH=str(binary_dir) + ":" + os.environ["PATH"])
    os.environ.pop("AGENT_SPECTATOR_FULL", None)
    games = []
    started = time.monotonic()
    metadata = {"schema": "network-input-ab/v2", "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint), "native_sha256": sha256(native_path),
                "native_checkpoint": str(native_checkpoint),
                "native_checkpoint_sha256": sha256(native_checkpoint),
                "models": {
                    "network": {"checkpoint": str(checkpoint), "sha256": sidecar["sha256"],
                                "global_step": sidecar["global_step"],
                                "sidecar_sha256": sha256(checkpoint.with_name(checkpoint.name + ".json")),
                                "input": "native_training" if args.control else "deployed_network",
                                "device": "gpu"},
                    "native": {"checkpoint": str(native_checkpoint), "sha256": native_sidecar["sha256"],
                               "global_step": native_sidecar["global_step"],
                               "sidecar_sha256": sha256(native_checkpoint.with_name(native_checkpoint.name + ".json")),
                               "input": "native_training", "device": "gpu",
                               "observation_revision": native_sidecar.get("observation_revision"),
                               "training_native_sha256": native_sidecar["native_module_sha256"]}},
                "assets": assets,
                "project_root": str(project),
                "deck_sha256": sha256(deck), "seed": args.seed, "requested_games": 2 * args.pairs,
                "policy": "argmax", "device": str(jax.devices()), "control": args.control,
                "control_action_source": str(args.replay_control) if args.replay_control else "gpu_policy",
                "batch_size": args.batch_size, "env_threads": common["num_threads"],
                "network_protocol_version": "0x133e",
                "network_encoder_sha256": sha256(args.room_root / "src/ygo_sky/duel/network/structured.py"),
                "network_client_sha256": sha256(args.room_root / "src/ygo_sky/duel/network/client.py"),
                "native_binary_matches_training": binary_matches,
                "native_tensor_parity_proof": str(args.parity_proof) if args.parity_proof else None,
                "transport": "native_isolated_tcp_bridge",
                "aliyun_end_to_end": False,
                "limitations": ["Aliyun WASM core and refresh timings are not reproduced",
                                "Bridge supplies healthy authoritative initial deck counts"],
                "private_opponent_information_sent_to_network": False}
    atomic_json(args.output / "configuration.json", metadata)

    def publish():
        document = {"configuration": metadata, "summary": summarize(games),
                    "by_network_seat": {str(seat): summarize([game for game in games
                        if game["network_seat"] == seat]) for seat in (0, 1)},
                    "wall_seconds": time.monotonic() - started, "games": games}
        atomic_json(args.output / "progress.json", document)
        print(json.dumps({"event": "progress", **document["summary"],
                          "seconds": round(document["wall_seconds"], 1)}), flush=True)
        return document

    try:
        for offset in range(0, args.pairs, args.batch_size):
            count = min(args.batch_size, args.pairs - offset)
            for native_seat in (0, 1):
                os.environ.update(INPUT_EVAL_BATCH_OFFSET=str(offset),
                                  INPUT_EVAL_NATIVE_SEAT=str(native_seat),
                                  INPUT_EVAL_BATCH_SEED=str(args.seed + offset))
                options = dict(common, num_envs=count, seed=args.seed + offset,
                               player=-1 if args.control else native_seat,
                               play_mode="self" if args.control else "windbot")
                if not args.control:
                    options.update(windbot_root=str(project), windbot_executable=str(Path(__file__).resolve()),
                                   windbot_executor="InputEval", windbot_deck_file=str(deck),
                                   windbot_cards_db=str(cards_db),
                                   windbot_log_path=str(args.output / f"peer-{offset}-{native_seat}"),
                                   windbot_seed=args.seed + offset, windbot_timeout_ms=120000)
                env = ygoenv.make(**options)
                lengths = np.zeros(count, dtype=np.int32)
                digests = [hashlib.sha256() for _ in range(count)]
                active = np.arange(count, dtype=np.int32)
                states = [jax.tree.map(lambda value: value.copy(), blank_states) for _ in (0, 1)]
                try:
                    observation, info = env.reset()
                    observation["mask_"] = None
                    while active.size:
                        actor = np.asarray(info["to_play"], dtype=np.int32)
                        if not args.control and np.any(actor != native_seat):
                            raise ValueError("Native bridge exposed opponent's private decision")
                        observations = [{key: None if value is None else value[index:index + 1]
                                         for key, value in observation.items()} for index in range(len(active))]
                        for identity, player, row in zip(active, actor, observations):
                            digest = digests[identity]
                            digest.update(bytes([int(player)]))
                            for key in sorted(row):
                                value = row[key]
                                digest.update(key.encode("ascii"))
                                digest.update(b"none" if value is None else value.tobytes())
                        current = jax.tree.map(lambda *values: np.concatenate(values),
                            *[jax.tree.map(lambda value: value[identity:identity + 1], states[int(player)])
                              for identity, player in zip(active, actor)],
                            *[native_agent.init_rnn_state(1) for _ in range(args.batch_size - len(active))])
                        updated, actions = native_predict(current, observations)
                        if args.replay_control:
                            actions = actions.copy()
                            for index, identity in enumerate(active):
                                trace_name = f"{offset}-{native_seat}-{identity}-{int(lengths[identity]):04d}.npz"
                                with np.load(args.replay_control / trace_name) as reference:
                                    if int(reference["actor"]) != int(actor[index]):
                                        raise ValueError(f"Replay actor mismatch: {trace_name}")
                                    for key, value in observations[index].items():
                                        if value is not None and not np.array_equal(reference[key], value):
                                            raise ValueError(f"Replay tensor mismatch: {trace_name}/{key}")
                                    actions[index] = int(reference["action"])
                        if args.trace_control:
                            for index, identity in enumerate(active):
                                trace_path = args.output / "trace" / (
                                    f"{offset}-{native_seat}-{identity}-{int(lengths[identity]):04d}.npz")
                                np.savez_compressed(trace_path,
                                    actor=np.asarray(actor[index]), action=np.asarray(actions[index]),
                                    **{key: value for key, value in observations[index].items()
                                       if value is not None})
                        for index, (identity, player) in enumerate(zip(active, actor)):
                            jax.tree.map(lambda target, source: target.__setitem__(identity, source[index]),
                                         states[int(player)], updated)
                        legal_count = np.asarray(info["num_options"])
                        if np.any(actions[:len(active)] >= legal_count):
                            raise ValueError("Native policy returned illegal action")
                        for identity, action in zip(active, actions):
                            digests[identity].update(bytes([int(action)]))
                        observation, reward, terminated, truncated, info = env.step(
                            actions[:len(active)].astype(np.int32), env_id=active)
                        if not np.array_equal(np.asarray(info["env_id"]), active):
                            raise ValueError("Environment batch ordering changed")
                        observation["mask_"] = None
                        done = np.asarray(terminated) | np.asarray(truncated)
                        lengths[active] += 1
                        for index in np.flatnonzero(done):
                            native_reward = float(reward[index])
                            if args.control:
                                native_reward *= 1 if actor[index] == native_seat else -1
                            games.append({"pair_index": offset + int(active[index]),
                                          "batch_seed": args.seed + offset,
                                          "environment_index": int(active[index]),
                                          "network_seat": 1 - native_seat,
                                          "network_outcome": "loss" if native_reward > 0 else
                                              "win" if native_reward < 0 else "draw",
                                          "length": int(lengths[active[index]]),
                                          "observation_action_sha256": digests[active[index]].hexdigest(),
                                          "win_reason": int(info["win_reason"][index]),
                                          "truncated": bool(truncated[index])})
                        alive = ~np.asarray(done, dtype=np.bool_)
                        active = active[alive]
                        observation = {key: None if value is None else value[alive]
                                       for key, value in observation.items()}
                        info = {key: np.asarray(info[key])[alive]
                                for key in ("to_play", "num_options", "win_reason", "env_id")}
                finally:
                    env.close()
                publish()
        deadline = time.monotonic() + 30
        while not args.control and len(list(args.output.glob("peer-*.json"))) < len(games):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        peers = [json.loads(path.read_text()) for path in args.output.glob("peer-*.json")]
        errors = [] if args.control else audit_peers(games, peers)
        document = publish()
        document.update(status="complete", peer_errors=errors, peer_count=len(peers),
                        network_prompts=dict(sum((Counter(peer["prompts"]) for peer in peers), Counter())))
        atomic_json(args.output / "result.json", document)
    except Exception as error:
        document = publish()
        document.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(args.output / "result.json", document)
        raise
    finally:
        stop.set()
        listener.close()
        for thread in threads:
            thread.join(timeout=2)
        address.unlink(missing_ok=True)


if __name__ == "__main__":
    if any(argument.startswith("Host=") for argument in sys.argv[1:]):
        peer_main()
    else:
        main()
