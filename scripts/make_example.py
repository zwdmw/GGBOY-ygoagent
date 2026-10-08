"""Capture a real native decision for inference acceptance."""
import os
from pathlib import Path

from ygo_sky.paths import project_root

root = project_root()
os.environ["YGO_GAME_ROOT"] = str(root / "resources/game")
os.chdir(root / "resources/game")
import numpy as np
import ygoenv
from ygoai.rl.env import EnvPreprocess
from ygoai.utils import init_ygopro

deck = init_ygopro("YGOPro-v1", "english", str(root / "resources/decks/sky-striker.ydk"),
                   str(root / "resources/game/scripts/code_list.txt"))
base = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1,
                   seed=2026100801, deck1=deck, deck2=deck, play_mode="self", player=-1,
                   max_options=128, n_history_actions=32, oppo_info=False, async_reset=False,
                   max_steps=1000, timeout=120)
base.num_envs = 1
env = EnvPreprocess(base, skip_mask=True)
try:
    observation, info = env.reset()
    np.savez_compressed(root / "resources/example-observation.npz",
                        **{key: value for key, value in observation.items() if value is not None})
    print({"legal_count": int(info["num_options"][0]), "to_play": int(info["to_play"][0])})
finally:
    env.close()
