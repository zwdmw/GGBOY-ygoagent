"""Two model clients in one private 233 room; no public matchmaking."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ygo_sky.paths import project_root, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="gpu")
parser.add_argument("--deck")
args = parser.parse_args()
if not os.environ.get("YGO_ROOM"):
    raise ValueError("Set YGO_ROOM to a private room identifier")
os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from ygo_sky.model import Predictor
from ygo_sky.duel.run import run
from ygo_sky.observation import template

root = project_root()
predictor = Predictor(root, device=args.device)
predictor.choose(template(), 1, "warmup")
predictor.reset("warmup")
config = read_json(root / "configs/duel/233.json")
if args.deck:
    config["deck"] = args.deck
config.update(timeout=180, transition_idle=180)
args.output.mkdir(parents=True, exist_ok=False)
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(run, root, {**config, "nickname": f"Sky463M_{seat}",
                                          "seed": config["seed"] + seat},
                               args.output / f"seat-{seat}", args.device, predictor) for seat in (0, 1)]
    results = [future.result() for future in futures]
(args.output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
