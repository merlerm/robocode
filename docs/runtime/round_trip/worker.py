"""Manual revision check: a service and fixed-policy client inside Apptainer."""

import argparse
import hashlib
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from runpy import run_path
from types import FrameType
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf

from robocode.runtime.artifacts import load_environment_artifact
from robocode.utils.env_client import BlackboxEnv
from robocode.utils.env_server import encode, serialize_space

ROOT = Path("/sandbox")


def save(name: str, value: Any) -> None:
    """Write an inspectable JSON record."""
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def serve(expected_id: str) -> None:
    """Run the existing strict RPC server from the requested artifact."""
    from robocode.utils.env_server_runtime import (  # pylint: disable=import-outside-toplevel
        _EnvServer,
    )

    artifact = load_environment_artifact(Path("/artifact"), expected_id=expected_id)
    config = artifact.manifest["config"]
    env = instantiate(OmegaConf.create(config))
    metadata: dict[str, Any] = {
        "artifact_id": artifact.artifact_id,
        "source_commit": artifact.manifest["source"]["commit"],
        "description": env.env_description_blackbox,
        "observation_space": serialize_space(env.observation_space),
        "action_space": serialize_space(env.action_space),
        "max_steps": 200,
        "strict": True,
    }
    env.close()
    save("metadata.json", metadata)
    save("imports-start.json", artifact.verify_imports())
    # This token is local to a disposable, network-isolated verification container.
    with _EnvServer(config, "round-trip-probe", strict=True) as server:
        metadata.update(
            host="127.0.0.1", port=server.server_address[1], token="round-trip-probe"
        )
        save("ready.json", metadata)
        print(
            f"SERVICE_READY artifact={artifact.artifact_id} "
            f'commit={metadata["source_commit"]}',
            flush=True,
        )

        def stop(_signum: int, _frame: FrameType | None) -> None:
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, stop)
        try:
            server.serve_forever()
        finally:
            save("imports-end.json", artifact.verify_imports())
            print("SERVICE_STOPPED", flush=True)


def evaluate(expected_id: str) -> None:
    """Use the same policy and seeds through the service for each revision."""
    policy_class = run_path("/probe/approach.py")["GeneratedApproach"]

    artifact = load_environment_artifact(Path("/artifact"), expected_id=expected_id)
    policy_hash = hashlib.sha256(Path("/probe/approach.py").read_bytes()).hexdigest()
    log = (ROOT / "server.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-B", "/probe/worker.py", "--server", expected_id],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 60
        while not (ROOT / "ready.json").exists():
            if proc.poll() is not None:
                raise RuntimeError((ROOT / "server.log").read_text())
            if time.monotonic() > deadline:
                raise TimeoutError("Environment service startup timed out")
            time.sleep(0.1)
        metadata = json.loads((ROOT / "ready.json").read_text())
        client = BlackboxEnv(metadata, sandbox_root=ROOT)
        episodes = []
        renders = {}
        try:
            with (ROOT / "transitions.jsonl").open("w") as trace:
                for seed in range(10):
                    obs, info = client.reset(seed=seed)
                    policy = policy_class(
                        client.action_space, client.observation_space, {}
                    )
                    policy.reset(obs, info)
                    trace.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "step": 0,
                                "obs": encode(obs),
                                "info": encode(info),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    if seed == 0:
                        renders["initial"] = client.render_state(
                            state=obs, label="initial"
                        )
                    total_reward = 0.0
                    for step in range(1, 201):
                        action = policy.get_action(obs)
                        obs, reward, terminated, truncated, info = client.step(action)
                        total_reward += reward
                        trace.write(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "step": step,
                                    "action": encode(action),
                                    "obs": encode(obs),
                                    "reward": reward,
                                    "terminated": terminated,
                                    "truncated": truncated,
                                    "info": encode(info),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        if seed == 0 and step == 30:
                            renders["step30"] = client.render_state(
                                state=obs, label="step30"
                            )
                        if terminated or truncated:
                            break
                    if seed == 0:
                        renders["final"] = client.render_state(state=obs, label="final")
                    row = {
                        "seed": seed,
                        "steps": step,
                        "terminated": terminated,
                        "truncated": truncated,
                        "return": total_reward,
                    }
                    episodes.append(row)
                    print("EPISODE " + json.dumps(row, sort_keys=True), flush=True)
        finally:
            client.close()
        save(
            "result.json",
            {
                "artifact_id": artifact.artifact_id,
                "policy_sha256": policy_hash,
                "episodes": episodes,
                "renders": renders,
            },
        )
        load_environment_artifact(Path("/artifact"), expected_id=expected_id)
        print("POST_RUN_ARTIFACT_VERIFIED", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
    assert proc.returncode == 0, (ROOT / "server.log").read_text()
    print("SERVICE_PROCESS_REAPED", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("expected_id")
    args = parser.parse_args()
    if args.server:
        serve(args.expected_id)
    else:
        evaluate(args.expected_id)
