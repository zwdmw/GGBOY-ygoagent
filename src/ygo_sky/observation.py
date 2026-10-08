"""Training model input geometry; all dimensions exclude the batch axis."""
import numpy as np

SHAPES = {
    "cards_": (160, 41), "global_": (23,), "actions_": (128, 12),
    "h_actions_": (32, 14), "action_ir_": (128, 24),
    "action_single_refs_": (128, 4), "action_group_refs_": (128, 5, 8),
    "action_group_mask_": (128, 5, 8), "selection_": (12,),
    "public_events_": (32, 16), "public_event_refs_": (32, 4),
}


def template(batch=1):
    result = {k: np.zeros((batch, *shape), dtype=np.uint8) for k, shape in SHAPES.items()}
    result["actions_"][:, 0, 3] = 1
    result["mask_"] = None
    return result


def validate(observation, batch=1):
    if set(observation) - {*SHAPES, "mask_"}:
        raise ValueError("Unexpected observation fields")
    for key, shape in SHAPES.items():
        value = observation.get(key)
        if value is None or np.shape(value) != (batch, *shape) or np.asarray(value).dtype != np.uint8:
            raise ValueError(f"Invalid {key}: expected uint8 {(batch, *shape)}")
    if observation.get("mask_") is not None:
        raise ValueError("This observable model requires mask_=None")
    return {**observation, "mask_": None}
