"""Episode-level uniform routing over isolated, frozen environment services."""

import hashlib
import json
import random
import secrets
import socket
import socketserver
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import replace

from robocode.runtime.environment import environment_service


class MixtureServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, bindings, directory):
        self.bindings = bindings
        self.token = secrets.token_hex(16)
        self.directory = directory
        self.log_lock = threading.Lock()
        super().__init__(("0.0.0.0", 0), MixtureHandler)

    def selected(self, seed):
        # Fixed seeds reproduce the revision as well as its initial state.
        rng = (
            random.Random(f"uniform-archive:{seed}")
            if seed is not None
            else random.SystemRandom()
        )
        return rng.randrange(len(self.bindings))


class MixtureHandler(socketserver.StreamRequestHandler):
    def handle(self):
        upstream = ExitStack()
        stream = None
        binding = None
        connection = secrets.token_hex(8)
        try:
            for line in self.rfile:
                request = json.loads(line)
                if request.pop("token", None) != self.server.token:
                    self.wfile.write(b'{"error":"invalid token"}\n')
                    return
                if request["cmd"] == "close":
                    return
                if request["cmd"] == "reset" or stream is None:
                    upstream.close()
                    index = self.server.selected(request.get("seed"))
                    binding = self.server.bindings[index]
                    sock = upstream.enter_context(
                        socket.create_connection(
                            (binding.host, binding.port), timeout=30
                        )
                    )
                    stream = upstream.enter_context(sock.makefile("rwb"))
                    entry = {
                        "connection": connection,
                        "seed": request.get("seed"),
                        "options": request.get("options"),
                        "revision_index": index,
                        "environment_artifact_id": binding.artifact_id,
                    }
                    with self.server.log_lock:
                        with (self.server.directory / "samples.jsonl").open("a") as log:
                            log.write(json.dumps(entry) + "\n")
                request["token"] = binding.token
                stream.write(json.dumps(request).encode() + b"\n")
                stream.flush()
                response = stream.readline()
                if not response:
                    raise RuntimeError("Mixture member closed its connection")
                self.wfile.write(response)
        finally:
            upstream.close()


@contextmanager
def uniform_mixture_service(artifacts, service_dir, **kwargs):
    """Route requests only; simulation and source remain in member containers."""
    service_dir.mkdir(parents=True)
    with ExitStack() as stack:
        bindings = []
        for index, artifact in enumerate(artifacts):
            member_kwargs = dict(kwargs)
            if member_kwargs.get("telemetry_run_id"):
                member_kwargs["telemetry_run_id"] += f"-revision{index}"
            bindings.append(
                stack.enter_context(
                    environment_service(
                        json.dumps(artifact.manifest["config"]),
                        service_dir / f"revision{index}",
                        artifact=artifact,
                        **member_kwargs,
                    )
                )
            )
        for binding in bindings:
            for key in ("action_space", "observation_space", "counts"):
                assert binding.metadata.get(key) == bindings[0].metadata.get(key), key
        manifest = {
            "distribution": "uniform_per_reset",
            "members": [a.artifact_id for a in artifacts],
        }
        identity = (
            "sha256:"
            + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        )
        (service_dir / "mixture.json").write_text(
            json.dumps(dict(manifest, mixture_id=identity), indent=2) + "\n"
        )
        with MixtureServer(bindings, service_dir) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield replace(
                    bindings[-1],
                    service_id="uniform-mixture",
                    host="127.0.0.1",
                    port=server.server_address[1],
                    token=server.token,
                    artifact_id=identity,
                )
            finally:
                server.shutdown()
                thread.join()
