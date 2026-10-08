#!/usr/bin/env python3
"""Verify complete-decision tensor hashes before using the bridged native."""
import argparse
import json
from pathlib import Path


def compare(training, bridge):
    if training["status"] != "complete" or bridge["status"] != "complete":
        raise ValueError("Both controls must be complete")
    for field in ("checkpoint_sha256", "deck_sha256", "seed", "requested_games"):
        if training["configuration"][field] != bridge["configuration"][field]:
            raise ValueError(f"Different control configuration: {field}")
    for field in ("native_checkpoint_sha256", "assets"):
        if training["configuration"].get(field) != bridge["configuration"].get(field):
            raise ValueError(f"Different control configuration: {field}")
    def keyed(document):
        return {(game["pair_index"], game["network_seat"]): game for game in document["games"]}
    baseline, candidate = keyed(training), keyed(bridge)
    if set(baseline) != set(candidate) or len(baseline) != training["configuration"]["requested_games"]:
        raise ValueError("Control game coverage differs")
    fields = ("observation_action_sha256", "network_outcome", "length", "win_reason", "truncated")
    differences = [{"game": list(identity), "field": field}
                   for identity in baseline for field in fields
                   if baseline[identity][field] != candidate[identity][field]]
    proof = {"passed": not differences, "differences": differences,
            "games": len(baseline), "decisions": sum(game["length"] for game in baseline.values()),
            "training_native_sha256": training["configuration"]["native_sha256"],
            "bridge_native_sha256": bridge["configuration"]["native_sha256"],
            "checkpoint_sha256": training["configuration"]["checkpoint_sha256"],
            "deck_sha256": training["configuration"]["deck_sha256"],
            "method": "all model input tensor bytes, actor and chosen action at every decision",
            "seed": training["configuration"]["seed"]}
    if "assets" in training["configuration"]:
        proof["assets"] = training["configuration"]["assets"]
    proof["bridge_control_action_source"] = bridge["configuration"].get(
        "control_action_source", "gpu_policy")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=Path)
    parser.add_argument("bridge", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    proof = compare(json.loads(args.training.read_text()), json.loads(args.bridge.read_text()))
    proof.update(training_result=str(args.training), bridge_result=str(args.bridge))
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)
    if not proof["passed"]:
        raise SystemExit(1)
