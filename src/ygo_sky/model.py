import inspect
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np

from ygoai.rl.jax.agent import RNNAgent
from ygoai.rl.jax.decision_checkpoint import restore_decision_checkpoint
from .observation import template, validate
from .paths import contained
from .resources import verify
from .semantics import inject_semantic_constants, load_structured_semantics


class Predictor:
    def __init__(self, root, model="base-463m", device="cpu"):
        self.manifest, metadata, _ = verify(root, model)
        self.model_id = self.manifest["artifacts"]["checkpoint"]["sha256"]
        artifacts = self.manifest["artifacts"]
        tables, shape, _ = load_structured_semantics(str(contained(root, artifacts["semantics"]["path"])),
                                                    str(contained(root, artifacts["code_list"]["path"])))
        if list(shape) != metadata["semantic_shape"]:
            raise ValueError("Semantic shape differs from checkpoint")
        accepted = set(inspect.signature(RNNAgent).parameters)
        unknown = set(metadata["model_args"]) - accepted
        if unknown:
            raise ValueError(f"Unsupported checkpoint model arguments: {sorted(unknown)}")
        platform, _, index = device.partition(":")
        devices = jax.devices("gpu" if platform in ("cuda", "gpu") else platform)
        self.device = devices[int(index or "0")]
        self.agent = RNNAgent(**metadata["model_args"], semantic_shape=shape,
                              dtype=jnp.bfloat16 if metadata["bfloat16"] else jnp.float32,
                              switch=metadata["switch"])
        with jax.default_device(self.device):
            sample = jax.tree.map(lambda x: None if x is None else jnp.asarray(x), template(), is_leaf=lambda x: x is None)
            variables = flax.core.unfreeze(self.agent.init(jax.random.PRNGKey(0), sample, self.agent.init_rnn_state(1)))
            inject_semantic_constants(variables, tables)
            checkpoint = contained(root, artifacts["checkpoint"]["path"])
            variables, added = restore_decision_checkpoint(variables, checkpoint.read_bytes())
            if added:
                raise ValueError("Partial checkpoint migration is forbidden in inference")
            inject_semantic_constants(variables, tables)
            self.variables = jax.device_put(flax.core.freeze(variables), self.device)
        self._infer = jax.jit(lambda state, observation: self.agent.apply(self.variables, observation, state)[:2], device=self.device)
        self.states = {}

    def new_state(self, batch=1):
        return jax.device_put(self.agent.init_rnn_state(batch), self.device)

    def reset(self, session="default"):
        self.states.pop(session, None)

    def evaluate(self, state, observation):
        return self._infer(state, jax.device_put(observation, self.device))

    def choose(self, observation, count, session="default"):
        observation = validate(observation)
        if type(count) is not int or not 1 <= count <= 128:
            raise ValueError("Legal action count must be between 1 and 128")
        if session not in self.states and len(self.states) >= 1024:
            raise ValueError("Too many policy sessions; reset completed sessions")
        started = time.monotonic()
        previous = self.states[session] if session in self.states else self.new_state()
        state, logits = self.evaluate(previous, observation)
        scores = np.asarray(logits)[0]
        if not np.isfinite(scores[:count]).all():
            raise ValueError("Non-finite legal action logits")
        index = int(np.argmax(scores[:count]))
        if int(np.argmax(scores)) >= count:
            raise ValueError("Model selected padding; legal action count/encoding mismatch")
        self.states[session] = state
        return {"action": index, "legal_count": count, "inference_ms": (time.monotonic() - started) * 1000,
                "model_sha256": self.model_id, "global_step": self.manifest["global_step"]}
