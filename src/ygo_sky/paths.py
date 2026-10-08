import json
import os
from pathlib import Path


def project_root(value=None):
    if value or os.environ.get("YGO_SKY_HOME"):
        root = Path(value or os.environ["YGO_SKY_HOME"]).expanduser().resolve()
    else:
        root = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()), None)
        if root is None:
            raise ValueError("Set YGO_SKY_HOME or pass --root to the extracted project/resources directory")
    if not (root / "configs/model").is_dir():
        raise ValueError(f"Missing model configuration directory in {root}")
    return root


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Resource path escapes the project root")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
