"""Isolated process launch for native training and paired evaluation."""
import os
import subprocess
import sys
import time
from pathlib import Path

from .paths import contained, read_json, write_json
from .resources import verify, sha256


def native_runtime(root):
    runtime = read_json(root / "native/runtime.json")
    path = contained(root, runtime["module"])
    if not path.is_relative_to(root / "native/ygoenv") or sha256(path) != runtime["sha256"]:
        raise ValueError("Native runtime identity mismatch")
    if runtime.get("observation_revision") != "sky-selfplay-own-deck-candidates-v2":
        raise ValueError("Unsupported native observation contract")
    return runtime


def environment(root, platform="gpu"):
    return {**os.environ, "YGO_SKY_HOME": str(root),
            "YGO_GAME_ROOT": str(root / "resources/game"),
            "PYTHONPATH": os.pathsep.join(map(str, (root / "src", root / "native/ygoenv"))),
            "JAX_PLATFORMS": "cuda" if platform == "gpu" else platform,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def flags(values):
    result = []
    for key, value in values.items():
        if value is None:
            continue
        key = key.replace("_", "-")
        if isinstance(value, bool):
            parts = key.rsplit(".", 1)
            flag = key if value else ".".join([*parts[:-1], "no-" + parts[-1]])
            result.append("--" + flag)
        else:
            result.append("--" + key)
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def training(root, config, output, platform="gpu", dry_run=False, pool=None):
    config = read_json(config)
    if config.get("schema") != "ygo-sky-training/v1":
        raise ValueError("Unsupported training configuration")
    if config["trainer"] not in ("cleanba", "cleanba_pool", "cleanba_mirror_pool"):
        raise ValueError("Unknown trainer")
    manifest, metadata, hashes = verify(root, config["model"])
    args = dict(config["args"])
    args.update({"checkpoint": str(contained(root, manifest["artifacts"]["checkpoint"]["path"])),
                 "tb_offset": metadata["global_step"],
                 "deck": str(contained(root, config["deck"])),
                 "semantic_file": str(contained(root, manifest["artifacts"]["semantics"]["path"])),
                 "code_list_file": str(contained(root, manifest["artifacts"]["code_list"]["path"])),
                 "ckpt_dir": str(output / "checkpoints"), "tb_dir": str(output / "tensorboard"),
                 "cluster_stats_dir": str(output / "cluster-stats"),
                 "run_name": f"{config['trainer']}__{args['seed']}__{int(time.time())}"})
    args.update({"m1." + key: value for key, value in metadata["model_args"].items()})
    if config["trainer"] != "cleanba":
        selected = pool or config.get("opponent_pool")
        if not selected:
            raise ValueError("This recipe requires --pool with historical checkpoints and sidecars")
        args["opponent_pool_dir"] = str(Path(selected).resolve())
    batch = args["local_num_envs"] * args["num_actor_threads"] * args["num_steps"] * len(args["actor_device_ids"])
    if args["total_timesteps"] < batch or args["total_timesteps"] % batch:
        raise ValueError(f"Additional training steps must be a positive multiple of rollout batch {batch}")
    command = [sys.executable, "-m", "ygo_sky.training." + config["trainer"], *flags(args)]
    if dry_run:
        return {"command": command, "cwd": str(root / "resources/game"), "optimizer_restored": False}
    runtime = native_runtime(root)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"config": config, "resolved_args": args,
               "artifacts": hashes, "native_runtime": runtime, "optimizer_restored": False, "platform": platform})
    code = subprocess.call(command, cwd=root / "resources/game", env=environment(root, platform))
    write_json(output / "exit.json", {"exit_code": code})
    if code:
        raise RuntimeError(f"Training exited with code {code}; see {output}")
    return {"output": str(output), "exit_code": code}


def evaluation(root, config, output, platform="gpu", pairs=None, model=None, opponent=None):
    config = read_json(config)
    if config.get("schema") != "ygo-sky-evaluation/v1":
        raise ValueError("Unsupported evaluation configuration")
    candidate, _, _ = verify(root, model or config["model"])
    official_mode = config.get("mode") == "official"
    if official_mode:
        baseline = {"artifacts": config["opponent_artifacts"]}
        for item in baseline["artifacts"].values():
            if sha256(contained(root, item["path"])) != item["sha256"]:
                raise ValueError("Official opponent resource checksum mismatch")
    else:
        baseline, _, _ = verify(root, opponent or config["opponent"])
    runtime = native_runtime(root)
    values = {
        "structured_checkpoint": contained(root, candidate["artifacts"]["checkpoint"]["path"]),
        "official_checkpoint": contained(root, baseline["artifacts"]["checkpoint"]["path"]),
        "deck": contained(root, config["deck"]),
        "code_list": contained(root, candidate["artifacts"]["code_list"]["path"]),
        "official_code_list": contained(root, baseline["artifacts"]["code_list"]["path"]),
        "semantic_cache": contained(root, candidate["artifacts"]["semantics"]["path"]),
        "pairs": pairs or config["pairs"], "seed": config["seed"],
        "max_steps": config["max_steps"], "output": output / "result.json",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"config": config, "native_runtime": runtime, "resolved_args": {k: str(v) if isinstance(v, Path) else v for k, v in values.items()}, "platform": platform})
    if official_mode:
        values["allow_official_unknown"] = True
    module = "official" if official_mode else "paired"
    code = subprocess.call([sys.executable, "-m", "ygo_sky.evaluation." + module, *flags(values)],
                           cwd=root / "resources/game", env=environment(root, platform))
    if code:
        raise RuntimeError(f"Evaluation exited with code {code}")
    return read_json(output / "result.json")
