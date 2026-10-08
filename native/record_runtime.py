import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
modules = list((root / "native/ygoenv/ygoenv/ygopro").glob("ygopro_ygoenv*.so"))
if len(modules) != 1:
    raise RuntimeError(f"Expected one native module, found {modules}")
module = modules[0]
with module.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
value = {"schema": "ygo-sky-runtime/v1", "kind": "source-build", "module": str(module.relative_to(root)),
         "sha256": digest, "observation_revision": "sky-selfplay-own-deck-candidates-v2",
         "engine_commit": "f96929650ff8685b82fd48670126eae406366734"}
(root / "native/runtime.json").write_text(json.dumps(value, indent=2) + "\n")
