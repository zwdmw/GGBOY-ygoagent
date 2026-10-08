"""HTTP policy service with bounded requests and isolated recurrent sessions."""
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from .model import Predictor
from .observation import validate


def serve(root, model, device, host, port):
    token = os.environ.get("YGO_API_TOKEN", "")
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError("Non-loopback service requires YGO_API_TOKEN")
    predictor = Predictor(root, model, device)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass

        def reply(self, code, data):
            raw = json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"error": "unknown endpoint"})
                return
            self.reply(200, {"ready": True, "model_sha256": predictor.model_id,
                        "global_step": predictor.manifest["global_step"]})

        def do_POST(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "unauthorized"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise ValueError("Invalid body size")
                body = json.loads(self.rfile.read(size))
                session = body["session"]
                if not isinstance(session, str) or not 1 <= len(session) <= 128:
                    raise ValueError("Invalid session")
                if self.path == "/reset":
                    predictor.reset(session)
                    self.reply(200, {"reset": session})
                elif self.path == "/infer":
                    observation = {}
                    for key, value in body["observation"].items():
                        if value is None:
                            observation[key] = None
                            continue
                        array = np.asarray(value)
                        if not np.issubdtype(array.dtype, np.integer) or (array < 0).any() or (array > 255).any():
                            raise ValueError("Observation values must be integers in [0,255]")
                        observation[key] = array.astype(np.uint8)
                    self.reply(200, predictor.choose(validate(observation), body["legal_count"], session))
                else:
                    self.reply(404, {"error": "unknown endpoint"})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    server = HTTPServer((host, port), Handler)
    server.timeout = 30
    print(json.dumps({"listening": f"{host}:{port}", "model_sha256": predictor.model_id}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
