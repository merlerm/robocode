"""Compare mixture episodes with direct calls to their selected frozen revision."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiments.mixture_service import uniform_mixture_service
from robocode.runtime.environment import environment_service
from robocode.runtime.artifacts import load_environment_artifact
from robocode.utils.env_client import BlackboxEnv, _encode

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("runtime", type=Path)
parser.add_argument("artifacts", type=Path, nargs="+")
args = parser.parse_args()
artifacts = [load_environment_artifact(path) for path in args.artifacts]
with uniform_mixture_service(
    artifacts,
    args.output,
    strict=True,
    container_backend="docker",
    image="robocode-sandbox:multi-agent-runtime",
    runtime_source=args.runtime,
) as binding:
    metadata = dict(
        binding.metadata,
        host=binding.host,
        port=binding.port,
        token=binding.token,
        strict=True,
        max_steps=200,
        primitives=[],
    )
    client = BlackboxEnv(metadata)
    visited = set()
    with ExitStack() as stack:
        direct_bindings = [
            stack.enter_context(
                environment_service(
                    json.dumps(artifact.manifest["config"]),
                    args.output / f"direct-{index}",
                    artifact=artifact,
                    strict=True,
                    container_backend="docker",
                    image="robocode-sandbox:multi-agent-runtime",
                    runtime_source=args.runtime,
                )
            )
            for index, artifact in enumerate(artifacts)
        ]
        try:
            for seed in range(20):
                options = {"object_count": seed % 5}
                observation, _ = client.reset(seed=seed, options=options)
                sample = json.loads(
                    (args.output / "samples.jsonl").read_text().splitlines()[-1]
                )
                index = sample["revision_index"]
                visited.add(index)
                direct = direct_bindings[index]
                other = BlackboxEnv(
                    dict(
                        metadata, host=direct.host, port=direct.port, token=direct.token
                    )
                )
                try:
                    expected, _ = other.reset(seed=seed, options=options)
                    assert _encode(observation) == _encode(expected)
                    action = np.zeros(client.action_space.shape, dtype=np.float32)
                    assert _encode(client.step(action)) == _encode(other.step(action))
                finally:
                    other.close()
                again, _ = client.reset(seed=seed, options=options)
                assert _encode(again) == _encode(observation)
            assert visited == set(range(len(artifacts))), visited
            try:
                client.get_state()
            except RuntimeError as error:
                assert (
                    "strict" in str(error).lower()
                    or "unsupported" in str(error).lower()
                )
            else:
                raise AssertionError("Strict mixture exposed get_state")
        finally:
            client.close()
print(
    "PASS: 20 reset/step comparisons, deterministic revision selection, all members visited, strict access preserved."
)
