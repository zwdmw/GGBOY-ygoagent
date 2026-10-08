import io
import json
import socket
import struct
import zipfile

import numpy as np

LIMIT = 2 * 1024 * 1024


def exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Inference channel closed")
        data.extend(chunk)
    return bytes(data)


def send_frame(sock, payload):
    if not 0 < len(payload) <= LIMIT:
        raise ValueError("Invalid inference frame size")
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def receive_frame(sock):
    size, = struct.unpack("<I", exact(sock, 4))
    if not 0 < size <= LIMIT:
        raise ValueError("Invalid inference frame size")
    return exact(sock, size)


def encode_observation(actor, count, observation):
    arrays = {key: value for key, value in observation.items() if value is not None}
    arrays["__meta"] = np.asarray([actor, count], dtype=np.int32)
    arrays["__none"] = np.asarray([key for key, value in observation.items() if value is None], dtype="U64")
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def decode_observation(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) > 32 or sum(row.file_size for row in archive.infolist()) > LIMIT:
            raise ValueError("Oversized observation")
        if any(row.compress_type != zipfile.ZIP_STORED for row in archive.infolist()):
            raise ValueError("Compressed observation is unsupported")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        meta = data["__meta"]
        if meta.shape != (2,) or meta.dtype != np.int32:
            raise ValueError("Invalid observation metadata")
        actor, count = map(int, meta)
        if actor not in (0, 1) or not 1 <= count <= 128:
            raise ValueError("Invalid actor/legal action count")
        observation = {key: data[key] for key in data.files if not key.startswith("__")}
        for key in data["__none"].tolist():
            if key in observation or key.startswith("__"):
                raise ValueError("Invalid null observation field")
            observation[key] = None
    return actor, count, observation


def observation_schema(observation):
    return {key: None if value is None else (value.shape, value.dtype) for key, value in observation.items()}


def validate_observation(observation, schema):
    if observation_schema(observation) != schema:
        raise ValueError("Observation differs from the warmed schema")
