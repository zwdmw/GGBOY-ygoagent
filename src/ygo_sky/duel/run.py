import hashlib
import json
import os
import time

from ..model import Predictor
from ..observation import template
from ..paths import contained, write_json
from .network.cards import CardPool
from .network.client import ExternalClient
from .network.structured import StructuredEncoder, OBSERVATION_VERSION
from .network.vendor.netduel.cards import load_ydk


def run(root, config, output, device="cpu", predictor=None):
    if config.get("schema") != "ygo-sky-duel/v1":
        raise ValueError("Unsupported duel configuration")
    if type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
        raise ValueError("Invalid duel port")
    shared = predictor is not None
    predictor = predictor or Predictor(root, config["model"], device)
    session = "duel:" + str(output.resolve())
    if not shared:
        predictor.choose(template(), 1, session + ":warmup")
        predictor.reset(session + ":warmup")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = predictor.manifest["artifacts"]
    pool = CardPool(contained(root, artifacts["cards"]["path"]), contained(root, artifacts["code_list"]["path"]))
    pool.semantic_pending = set()
    random_deck = config.get("deck_source", "local") == "server_random"
    deck = contained(root, config["deck"])
    main, extra, side = ([], [], []) if random_deck else load_ydk(deck)
    encoder = StructuredEncoder(pool)
    started = time.monotonic()
    replay_entries = []
    decisions = 0
    client = None
    write_json(output / "session.json", {"config": config, "model_sha256": predictor.model_id,
               "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
               "network_observation": OBSERVATION_VERSION, "device": device})

    class Policy:
        def bind(self, bound):
            self.client = bound
            predictor.reset(session)
            encoder.reset(bound.result.our_player, config["seed"])

        def observe_game_message(self, msg, body):
            if self.client.result.our_player in (0, 1):
                encoder.observe_message(msg, body, self.client.result.turns, self.client.ctx.current_phase)

        def choose(self, state):
            nonlocal decisions
            observation = encoder.encode_state(state, self.client.selector)
            answer = predictor.choose(observation, int(state.n), session)
            decisions += 1
            decision_stream.write(json.dumps({**answer, "game": len(self.client.games) + 1,
                "turn": state.turn, "msg": state.msg}) + "\n")
            decision_stream.flush()
            return answer["action"]

        def record(self, state, action, selector, index, forced):
            encoder.record(state, action, selector, index, forced)

    def journal(direction, opcode, payload, **fields):
        trace.write(json.dumps({"direction": direction, "opcode": int(opcode),
                               "payload": payload.hex(), **fields}) + "\n")
        trace.flush()

    def status(state, **fields):
        write_json(output / "status.json", {"state": state, **fields})

    def save_replay(raw):
        if raw[:4] not in (b"yrp1", b"yrp2") or not 32 <= len(raw) <= 64 * 1024 * 1024:
            raise ValueError("Invalid original server replay")
        digest = hashlib.sha256(raw).hexdigest()
        if any(row["sha256"] == digest for row in replay_entries):
            return
        path = output / f"server-{len(replay_entries) + 1:03}.yrp"
        path.write_bytes(raw)
        replay_entries.append({"file": path.name, "sha256": digest, "bytes": len(raw),
                               "container": raw[:4].decode("ascii")})

    with (output / "decisions.jsonl").open("w", encoding="utf-8") as decision_stream, \
            (output / "network.jsonl").open("w", encoding="utf-8") as trace:
        outcome, error = "incomplete", ""
        try:
            client = ExternalClient(config["host"], config["port"], config["nickname"],
                main, extra, Policy(), side=side, version=config["version"], seed=config["seed"],
                timeout=config.get("timeout", 120), max_options=128, card_pool=pool,
                password=os.environ.get(config.get("room_env", "YGO_ROOM"), ""),
                account_password=os.environ.get(config.get("account_password_env", "YGO_ACCOUNT_PASSWORD"), ""),
                auth_mode=config.get("auth_mode", "none"), deck_source=config.get("deck_source", "local"),
                preference=config.get("preference", "auto"), journal=journal, state_change=status,
                save_replay=save_replay, require_deck_info=True,
                connect_window=config.get("connect_window", 30),
                reconnect_window=config.get("reconnect_window", 30),
                transition_idle=config.get("transition_idle", 120))
            result = client.run()
            outcome, error = client.outcome(), result.error
        except (Exception, KeyboardInterrupt) as exc:
            error = type(exc).__name__ + ": " + str(exc)[:250]
            for variable in (config.get("room_env", "YGO_ROOM"),
                             config.get("account_password_env", "YGO_ACCOUNT_PASSWORD")):
                secret = os.environ.get(variable, "")
                if secret:
                    error = error.replace(secret, "[REDACTED]")
        finally:
            if client is not None:
                client.close()
            predictor.reset(session)
            report = {"outcome": outcome, "error": error, "decisions": decisions,
                      "games": client.games if client else [],
                      "replays": replay_entries, "wall_seconds": time.monotonic() - started,
                      "model_sha256": predictor.model_id,
                      "replay_errors": client.replay_errors if client else [],
                      "deck_sync": client.deck_sync if client else {}}
            write_json(output / "result.json", report)
    return report
