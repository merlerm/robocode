"""Exercise real launchers with a deterministic program in place of a paid CLI."""

import asyncio

# pylint: disable=missing-function-docstring,unused-argument,use-dict-literal
import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.container_evaluation import evaluate_in_container
from robocode.runtime.environment import environment_service
from robocode.runtime.remote_environment import RemoteEnvironment
from robocode.runtime.whitebox import load_whitebox_source
from robocode.utils import apptainer_sandbox, docker_sandbox
from robocode.utils.backends import create_backend
from robocode.utils.env_server import write_env_spaces
from robocode.utils.model_broker import BrokerUpstream
from robocode.utils.sandbox_types import _StreamParseResult

_PROBE = """import asyncio
import json
from pathlib import Path
import numpy as np
import kinder
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from env_client import make_env
assert kinder.__file__.startswith('/whitebox/source/')
for name in ['/whitebox/source/src/kinder/__init__.py', '/robocode/src/robocode/environments/kinder_geom2d_env.py']:
    try:
        with open(name, 'a') as f: f.write('# forbidden')
    except OSError:
        pass
    else:
        raise AssertionError('source writable: ' + name)
assert not Path('/whitebox/source/demos').exists()
assert not Path('/robocode/src/robocode/oracles').exists()
local = KinderGeom2DEnv('kinder/Motion2D-p0-v0')
remote = make_env()
a, _ = local.reset(seed=42)
b, _ = remote.reset(seed=42)
np.testing.assert_array_equal(a, b)
a, *_ = local.step(np.array([.05, 0, 0, 0, 0]))
b, *_ = remote.step([.05, 0, 0, 0, 0])
np.testing.assert_array_equal(a, b)
assert local.render().shape == (750, 750, 3)
remote.render_state(state=b, label='pinned-remote')
local.close()
remote.close()
Path('/sandbox/approach.py').write_text('class GeneratedApproach:\\n    def __init__(self, action_space, observation_space, primitives): pass\\n    def reset(self, state, info): pass\\n    def get_action(self, state): return [0, 0, 0, 0, 0]\\n')
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
url = json.loads(Path('/sandbox/.mcp/mcp_config.json').read_text())['mcpServers']['robocode-tools']['url']
async def check_tools():
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert {t.name for t in (await session.list_tools()).tools} == {'render_state', 'render_policy'}
            for tool, args in [('render_state', {'seed': 42}), ('render_policy', {'approach_dir': '/sandbox', 'seed': 42, 'max_steps': 2, 'max_frames': 2})]:
                result = await session.call_tool(tool, args)
                assert not result.isError, result
asyncio.run(check_tools())
assert len(list(Path('/sandbox/mcp_renders').rglob('*.png'))) >= 2
print('REAL_MCP_STATE_AND_POLICY_RENDER_OK', flush=True)
print('PINNED_SOURCE_READONLY_LOCAL_AND_REMOTE_MATCH_RENDER_OK', flush=True)
"""


@pytest.fixture(params=["docker", "apptainer"], name="runtime")
def fixture_runtime(request):
    directory = os.environ.get("ROBOCODE_WHITEBOX_TEST_ARTIFACTS")
    if directory is None:
        pytest.skip("Set ROBOCODE_WHITEBOX_TEST_ARTIFACTS for live source-view checks")
    root = Path(directory)
    backend = request.param
    path = root / (backend + "-whitebox")
    identity = json.loads((path / "view.json").read_text())["source_id"]
    repo = Path(__file__).resolve().parents[2]
    return {
        "backend": backend,
        "artifact": load_environment_artifact(root / (backend + "-artifact")),
        "view": load_whitebox_source(path, identity),
        "runtime_source": root / "service-runtime",
        "image": (
            "robocode-sandbox:multi-agent-runtime"
            if backend == "docker"
            else str(repo / "robocode-sandbox.sif")
        ),
    }


class ProbeBackend:
    """Keep provider setup, replacing only inference and stream parsing."""

    def __init__(self, name):
        self.name = name
        self.backend = create_backend(OmegaConf.create({"backend": name}))

    def build_cli_cmd(self, config, **kwargs):
        self.backend.build_cli_cmd(config, **kwargs)
        return ["/robocode/.venv/bin/python", "-B", "/sandbox/probe.py"]

    def setup_sandbox_files(self, *args, **kwargs):
        self.backend.setup_sandbox_files(*args, **kwargs)

    def stdin_text(self, config):
        return ""

    def build_env(self, config, extra=None):
        return dict(os.environ)

    def parse_stream(self, proc, stream_log_path=None, stderr_file=None):
        stdout = proc.stdout.read()
        proc.wait(timeout=60)
        stderr_file.seek(0)
        output = stdout + stderr_file.read()
        stream_log_path.write_text(output)
        return _StreamParseResult(
            bool(proc.returncode), output if proc.returncode else None, 1, 0.0
        )


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_source_view_in_synthesis(runtime, tmp_path, monkeypatch, provider):
    backend, artifact = runtime["backend"], runtime["artifact"]
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)
    (tmp_path / "env_config.json").write_text(json.dumps(artifact.manifest["config"]))
    client = Path(__file__).resolve().parents[2] / "src/robocode/utils/env_client.py"
    with environment_service(
        json.dumps(artifact.manifest["config"]),
        tmp_path / "service",
        strict=True,
        artifact=artifact,
        container_backend=backend,
        image=runtime["image"],
        runtime_source=runtime["runtime_source"],
    ) as binding:
        env = RemoteEnvironment(binding, 200)
        write_env_spaces(
            sandbox,
            container_backend=backend,
            binding=binding,
            observation_space=env.observation_space,
            action_space=env.action_space,
            max_steps=200,
        )
        options = dict(
            sandbox_dir=sandbox,
            init_files={"probe.py": probe, "env_client.py": client},
            whitebox_source=runtime["view"],
            output_filename="approach.py",
            telemetry=True,
            mcp_tools=("render_state", "render_policy"),
            max_budget_usd=0,
        )
        if backend == "docker":
            monkeypatch.setattr(
                docker_sandbox,
                "_build_docker_auth_args",
                lambda _: nullcontext(([], {})),
            )
            config = docker_sandbox.DockerSandboxConfig(
                **options, docker_image=runtime["image"]
            )
            result = asyncio.run(
                docker_sandbox.run_agent_in_docker_sandbox(
                    config, ProbeBackend(provider)
                )
            )
        else:
            monkeypatch.setattr(
                apptainer_sandbox,
                "load_broker_upstream",
                lambda _: BrokerUpstream("responses", "api.openai.com", "/v1", {}),
            )
            monkeypatch.setattr(
                apptainer_sandbox, "_model_client", lambda _, command: (command, {})
            )
            config = apptainer_sandbox.ApptainerSandboxConfig(
                **options, sif_path=Path(runtime["image"]), env_server_port=binding.port
            )
            result = asyncio.run(
                apptainer_sandbox.run_agent_in_apptainer_sandbox(
                    config, ProbeBackend(provider)
                )
            )
        assert result.success, result.error
        provenance = json.loads((sandbox / "source-provenance.json").read_text())
        assert provenance["artifact_id"] == artifact.artifact_id
        assert provenance["whitebox_source_id"] == runtime["view"].source_id
        assert (
            "PINNED_SOURCE_READONLY_LOCAL_AND_REMOTE_MATCH_RENDER_OK"
            in (tmp_path / "stream.jsonl").read_text()
        )
        assert (tmp_path / "telemetry/events.jsonl").stat().st_size > 0
        env.close()


def test_source_view_in_evaluation(runtime, tmp_path):
    policy = tmp_path / "submission/approach.py"
    policy.parent.mkdir()
    policy.write_text("""import kinder
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
assert kinder.__file__.startswith('/whitebox/source/')
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.local = KinderGeom2DEnv('kinder/Motion2D-p0-v0')
    def reset(self, state, info):
        self.local.reset(seed=42)
    def get_action(self, state):
        self.local.step([0, 0, 0, 0, 0])
        return [0, 0, 0, 0, 0]
""")
    results = evaluate_in_container(
        policy,
        runtime["artifact"],
        tmp_path / "evaluation",
        backend=runtime["backend"],
        environment_image=runtime["image"],
        evaluation_image=runtime["image"],
        eval_seeds=[42],
        replicate_seed=42,
        max_steps=2,
        timeout=5,
        render=True,
        determinism_episodes=1,
        runtime_source=runtime["runtime_source"],
        whitebox_source=runtime["view"],
    )
    assert results["whitebox_source_id"] == runtime["view"].source_id
    assert results["environment_artifact_id"] == runtime["artifact"].artifact_id
    assert results["per_episode"][0]["num_steps"] == 2
    assert not results["per_episode"][0].get("crashed")
    assert results["whitebox_imports"]["kinder"][0].startswith("/whitebox/source/")
