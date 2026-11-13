"""Persistent editor turns through real launchers, with no model inference."""

# pylint: disable=missing-function-docstring,protected-access,import-outside-toplevel
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from robocode.runtime.agent_session import AgentSession
from robocode.runtime.editor import EditorBinding, prepare_editor_workspace
from robocode.runtime.policy_service import policy_service
from robocode.runtime.whitebox import load_whitebox_source
from robocode.utils import apptainer_sandbox, docker_sandbox
from robocode.utils.model_broker import BrokerUpstream
from tests.runtime.test_agent_turn_containers import FixtureBackend


class EditorBackend(FixtureBackend):
    """Keep both providers' launch arguments, session handling and stream parsers."""

    def build_cli_cmd(self, config, **kwargs):
        command = self.delegate.build_cli_cmd(config, **kwargs)
        return [
            "/robocode/.venv/bin/python",
            "-B",
            "/sandbox/editor_cli.py",
            self.name,
            *command,
        ]


@pytest.mark.parametrize("environment", ["motion", "variable"])
@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
@pytest.mark.parametrize("provider", ["claude", "codex"])
@pytest.mark.parametrize("access", ["blackbox", "whitebox"])
def test_editor_turns(tmp_path, monkeypatch, runtime, provider, access, environment):
    assets = os.environ.get("ROBOCODE_EDITOR_TEST_ARTIFACTS")
    if not assets:
        pytest.skip("Set ROBOCODE_EDITOR_TEST_ARTIFACTS to prepared checkpoint assets")
    root = Path(__file__).resolve().parents[2]
    path = Path(assets) / environment / (runtime + "-whitebox")
    view = load_whitebox_source(
        path, json.loads((path / "view.json").read_text())["source_id"]
    )
    session = AgentSession(tmp_path / "editor")
    prepare_editor_workspace(view, session.workspace)
    (session.workspace / "access-mode.txt").write_text(access)
    policy = tmp_path / "policy"
    policy.mkdir()
    from robocode.utils.env_server import serialize_space

    env_config = (
        OmegaConf.load(
            root / "experiments/conf/environment/obstruction2d_generalized.yaml"
        )
        if environment == "variable"
        else OmegaConf.create(
            {
                "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
                "env_id": "kinder/Motion2D-p0-v0",
            }
        )
    )
    (session.workspace / "probe-env.json").write_text(
        json.dumps(OmegaConf.to_container(env_config))
    )
    env = instantiate(env_config)
    spaces = {
        "action_space": serialize_space(env.action_space),
        "observation_space": serialize_space(env.observation_space),
    }
    env.close()
    files = {
        name: root / "tests/utils/fixtures" / name
        for name in ("editor_cli.py", "editor_probe.py")
    }
    if runtime == "docker":
        monkeypatch.setattr(
            docker_sandbox, "_build_docker_auth_args", lambda _: nullcontext(([], {}))
        )
        config_type = docker_sandbox.DockerSandboxConfig
        options = {"docker_image": "robocode-sandbox:multi-agent-runtime"}
        image = "robocode-evaluation:multi-agent"
    else:
        monkeypatch.setattr(
            apptainer_sandbox,
            "load_broker_upstream",
            lambda _: BrokerUpstream("responses", "api.openai.com", "/v1", {}),
        )
        monkeypatch.setattr(
            apptainer_sandbox, "_model_client", lambda _, cmd: (cmd, {})
        )
        config_type = apptainer_sandbox.ApptainerSandboxConfig
        options = {"sif_path": root / "robocode-sandbox.sif"}
        image = str(root / "robocode-evaluation.sif")
    turns = []
    bindings = []
    for turn_id, horizontal_action in (("e1", 0.04), ("e2", 0.02)):
        (policy / "approach.py").write_text(f"""class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): pass
    def reset(self, state, info): self.updates = 0
    def get_action(self, state): return [{horizontal_action} if not self.updates else .03, 0, 0, 0, 0]
    def update(self, state, reward, done, info): self.updates += 1
""")
        service = tmp_path / ("service-" + turn_id)
        with policy_service(
            policy / "approach.py",
            spaces,
            service,
            backend=runtime,
            image=image,
            blackbox_strict=True,
            seed=0,
        ) as binding:
            bindings.append(binding)
            editor = EditorBinding(
                view, binding, service / "policy" if access == "whitebox" else None
            )
            config = config_type(
                sandbox_dir=session.workspace,
                init_files=files,
                environment_editor=editor,
                model="fixture-model",
                telemetry=True,
                max_budget_usd=2,
                **options,
            )
            turns.append(
                AgentSession(session.directory).run_turn(
                    turn_id,
                    config,
                    EditorBackend(provider),
                    environment_id=view.manifest["artifact_id"],
                )
            )
    first, second = turns[0], turns[1]
    assert bindings[0].token != bindings[1].token
    assert first["policy_artifact_id"] != second["policy_artifact_id"]
    assert first["outcome"] == second["outcome"] == "submitted", (first, second)
    assert first["commit"] != second["commit"]
    assert second["resumed"]
    assert second["policy_artifact_id"] == binding.policy_artifact_id
    assert second["policy_access"] == access
    assert second["source_view_id"] == view.source_id
    assert "policy_service.json" not in second["artifact"]["files"]
    assert "approach.py" not in second["artifact"]["files"]
    assert not any("/.git/" in name for name in second["artifact"]["files"])
    assert json.loads((session.workspace / "e2-probe.json").read_text())[
        "transition_delta"
    ] == pytest.approx(0.005, abs=1e-6)
    assert (session.workspace / "e1.png").is_file()
    assert (session.workspace / "e2.png").is_file()
    assert (
        subprocess.check_output(
            ["git", "-C", str(session.workspace), "status", "--porcelain"], text=True
        )
        == ""
    )
    load_whitebox_source(view.path, view.source_id)
