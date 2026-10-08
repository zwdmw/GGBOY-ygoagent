"""Replay recorded legal actions and compare every input byte after rebuilding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("record", "replay"))
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--device", default="gpu")
parser.add_argument("--seed", type=int, default=2026100803)
args = parser.parse_args()
args.trace = args.trace.resolve()
args.binary = args.binary.resolve()
from ygo_sky.paths import project_root, write_json
root = project_root()
os.environ.update(JAX_PLATFORMS="cpu" if args.device == "cpu" else "cuda",
                  XLA_PYTHON_CLIENT_PREALLOCATE="false", YGO_GAME_ROOT=str(root / "resources/game"))
import tempfile
temporary = tempfile.TemporaryDirectory(prefix="sky-native-regression-")
package = Path(temporary.name) / "ygoenv"
shutil.copytree(root / "native/ygoenv/ygoenv", package,
                ignore=shutil.ignore_patterns("*.so", "__pycache__"))
shutil.copy2(args.binary, package / "ygopro/ygopro_ygoenv.cpython-311-x86_64-linux-gnu.so")
sys.path.insert(0, temporary.name)
import numpy as np
import ygoenv
from ygoai.utils import init_ygopro
from ygoai.rl.env import EnvPreprocess
from ygo_sky.model import Predictor

os.chdir(root / "resources/game")
deck = init_ygopro("YGOPro-v1", "english", str(root / "resources/decks/sky-striker.ydk"),
                   str(root / "resources/game/scripts/code_list.txt"))
base = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1,
                   seed=args.seed, deck1=deck, deck2=deck, play_mode="self", player=-1,
                   max_options=128, n_history_actions=32, oppo_info=False, async_reset=False,
                   max_steps=1000, timeout=120)
base.num_envs = 1
env = EnvPreprocess(base, skip_mask=True)
predictor = Predictor(root, device=args.device) if args.mode == "record" else None
trace = args.trace.resolve()
if args.mode == "record":
    trace.mkdir(parents=True, exist_ok=False)
steps, done, digest = 0, False, hashlib.sha256()
try:
    observation, info = env.reset()
    while not done:
        actor, count = int(info["to_play"][0]), int(info["num_options"][0])
        arrays = {key: value for key, value in observation.items() if value is not None}
        path = trace / f"{steps:04d}.npz"
        if predictor:
            action = predictor.choose(observation, count, f"seat-{actor}")["action"]
            np.savez_compressed(path, actor=actor, legal_count=count, action=action, **arrays)
        else:
            with np.load(path, allow_pickle=False) as saved:
                if actor != int(saved["actor"]) or count != int(saved["legal_count"]):
                    raise ValueError(f"Actor/legal count mismatch at {steps}")
                if set(saved.files) != {*arrays, "actor", "legal_count", "action"}:
                    raise ValueError("Observation keys mismatch")
                for key, value in arrays.items():
                    if not np.array_equal(saved[key], value):
                        raise ValueError(f"Tensor mismatch at {steps}: {key}")
                action = int(saved["action"])
        for key, value in sorted(arrays.items()):
            digest.update(key.encode())
            digest.update(value.tobytes())
        observation, reward, terminal, truncated, info = env.step(np.asarray([action], dtype=np.int32))
        steps += 1
        done = bool(terminal[0] or truncated[0])
        if bool(truncated[0]):
            raise ValueError("Regression ended before a complete game")
        if steps > 1000:
            raise ValueError("Regression exceeded native step limit")
    report = {"passed": True, "steps": steps, "tensor_sha256": digest.hexdigest(),
              "reward": float(reward[0]), "seed": args.seed,
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "mode": args.mode}
    if args.mode == "replay":
        reference = json.loads((trace / "record.json").read_text())
        for key in ("steps", "tensor_sha256", "reward"):
            if report[key] != reference[key]:
                raise ValueError(f"Terminal mismatch: {key}")
    write_json(trace / f"{args.mode}.json", report)
    print(json.dumps(report))
finally:
    env.close()
    temporary.cleanup()
