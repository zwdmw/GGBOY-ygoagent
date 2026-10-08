import hashlib
import json
import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp

def load_structured_semantics(path: str, code_list_file: str):
    cache_path = os.path.abspath(path)
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {"cdb_exact", "lua_effects", "lua_mask", "metadata"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                f"semantic cache is missing arrays: {sorted(missing)}"
            )
        cdb_exact = np.asarray(payload["cdb_exact"], dtype=np.float32)
        lua_effects = np.asarray(payload["lua_effects"], dtype=np.float16)
        lua_mask = np.asarray(payload["lua_mask"], dtype=np.uint8)
        metadata = json.loads(str(payload["metadata"].item()))

    with open(code_list_file, "r", encoding="utf-8") as handle:
        codes = [int(line.split()[0]) for line in handle if line.split()]
    expected_rows = len(codes) + 1
    if cdb_exact.ndim != 2:
        raise ValueError(f"cdb_exact must be rank 2, got {cdb_exact.shape}")
    if lua_effects.ndim != 3:
        raise ValueError(
            f"lua_effects must be rank 3, got {lua_effects.shape}"
        )
    if lua_mask.shape != lua_effects.shape[:2]:
        raise ValueError(
            f"lua_mask shape {lua_mask.shape} does not match "
            f"lua_effects {lua_effects.shape}"
        )
    if cdb_exact.shape[0] != expected_rows or lua_effects.shape[0] != expected_rows:
        raise ValueError(
            f"semantic row count must be {expected_rows}, got "
            f"{cdb_exact.shape[0]} and {lua_effects.shape[0]}"
        )
    code_list_hash = __import__("hashlib").sha256(
        "\n".join(str(code) for code in codes).encode()
    ).hexdigest()
    if metadata.get("code_list_sha256") != code_list_hash:
        raise ValueError("semantic cache code_list hash does not match")
    semantic_shape = (
        cdb_exact.shape[0],
        cdb_exact.shape[1],
        lua_effects.shape[1],
        lua_effects.shape[2],
    )
    tables = {
        "cdb_exact": cdb_exact,
        "lua_effects": lua_effects,
        "lua_mask": lua_mask,
    }
    return tables, semantic_shape, metadata

def inject_semantic_constants(tree, tables):
    matches = []

    def visit(node):
        if not isinstance(node, dict):
            return
        if set(tables).issubset(node):
            matches.append(node)
        for value in node.values():
            visit(value)

    visit(tree)
    if len(matches) != 1:
        raise ValueError(
            f"expected one structured semantic constant collection, "
            f"found {len(matches)}"
        )
    destination = matches[0]
    for name, value in tables.items():
        if tuple(destination[name].shape) != tuple(value.shape):
            raise ValueError(
                f"semantic tensor {name} shape mismatch: "
                f"{destination[name].shape} != {value.shape}"
            )
        destination[name] = jnp.asarray(value)
