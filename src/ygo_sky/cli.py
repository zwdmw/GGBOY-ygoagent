import argparse
import json
import os
from pathlib import Path

from .paths import project_root, read_json


def main():
    parser = argparse.ArgumentParser(prog="ygo-sky")
    parser.add_argument("--root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--model", default="base-463m")
    native = sub.add_parser("install-native")
    register = sub.add_parser("register-model")
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--name", required=True)
    archive = sub.add_parser("install-resources")
    archive.add_argument("archive")
    archive.add_argument("--sha256", required=True)
    infer = sub.add_parser("infer")
    infer.add_argument("--model", default="base-463m")
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--observation", type=Path, required=True)
    infer.add_argument("--legal-count", type=int, required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--model", default="base-463m")
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    duel = sub.add_parser("duel")
    duel.add_argument("--config", type=Path)
    duel.add_argument("--host")
    duel.add_argument("--port", type=int)
    duel.add_argument("--model")
    duel.add_argument("--deck")
    duel.add_argument("--nickname")
    duel.add_argument("--version", type=lambda value: int(value, 0))
    duel.add_argument("--device", default="cpu")
    duel.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    train.add_argument("--pool", type=Path)
    train.add_argument("--dry-run", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    evaluate.add_argument("--pairs", type=int)
    evaluate.add_argument("--model")
    evaluate.add_argument("--opponent")
    args = parser.parse_args()
    root = project_root(args.root)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.command in ("infer", "serve", "duel"):
        os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
    if args.command == "verify":
        from .resources import verify
        manifest, _, rows = verify(root, args.model)
        result = {"model": args.model, "step": manifest["global_step"], "artifacts": rows}
    elif args.command == "install-native":
        from .resources import install_native
        result = {"native": str(install_native(root))}
    elif args.command == "register-model":
        from .resources import register
        result = register(root, args.checkpoint.resolve(), args.name)
    elif args.command == "install-resources":
        from .resources import install_archive
        result = install_archive(root, args.archive, args.sha256)
    elif args.command == "infer":
        import numpy as np
        from .model import Predictor
        with np.load(args.observation, allow_pickle=False) as data:
            observation = {key: data[key] for key in data.files}
        result = Predictor(root, args.model, args.device).choose(observation, args.legal_count)
    elif args.command == "serve":
        from .service import serve
        serve(root, args.model, args.device, args.host, args.port)
        return
    elif args.command == "duel":
        from .duel.run import run
        config = read_json(args.config or root / "configs/duel/233.json")
        config.update({key: getattr(args, key) for key in ("host", "port", "model", "deck", "nickname", "version")
                       if getattr(args, key) is not None})
        result = run(root, config, args.output.resolve(), args.device)
    elif args.command == "train":
        from .runner import training
        result = training(root, args.config, args.output.resolve(), args.platform, args.dry_run, args.pool)
    else:
        from .runner import evaluation
        result = evaluation(root, args.config or root / "configs/eval/native-mirror.json",
                            args.output.resolve(), args.platform, args.pairs, args.model, args.opponent)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
