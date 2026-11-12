"""Exercise real Claude tool execution against a scripted local API, no model calls."""

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.backends.claude import ClaudeBackend
from robocode.utils.sandbox_types import SandboxConfig


def test_claude_settings_stop_timed_out_commands(tmp_path: Path) -> None:
    """Keep the production timeout policy checked without a container runtime."""
    ClaudeBackend(DEFAULT_BACKEND_CFG).setup_sandbox_files(
        SandboxConfig(sandbox_dir=tmp_path)
    )
    settings = json.loads((tmp_path / ".claude/settings.json").read_text())
    assert settings["env"]["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"


@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
@pytest.mark.parametrize("fixed", [False, True], ids=["control", "stop-on-timeout"])
def test_claude_command_timeout(tmp_path: Path, runtime: str, fixed: bool) -> None:
    """A timed-out wait dies, normal parallel shell work survives, and CLI exits."""
    if not os.environ.get("ROBOCODE_TEST_CLAUDE_TIMEOUT"):
        pytest.skip("Set ROBOCODE_TEST_CLAUDE_TIMEOUT=1 for real CLI/container checks")
    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests/utils/fixtures/claude_timeout_probe.py"
    backend = ClaudeBackend(DEFAULT_BACKEND_CFG)
    backend.setup_sandbox_files(SandboxConfig(sandbox_dir=tmp_path))
    if not fixed:
        # The control is identical apart from the newly added native setting.
        settings = tmp_path / ".claude/settings.json"
        data = json.loads(settings.read_text())
        data.get("env", {}).pop("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", None)
        settings.write_text(json.dumps(data))
    name = f"robocode-timeout-{uuid.uuid4().hex[:10]}"
    if runtime == "docker":
        cmd = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--name",
            name,
            "--entrypoint",
            "/opt/robocode-strict/bin/python",
            "-v",
            f"{tmp_path}:/sandbox",
            "-v",
            f"{fixture}:/probe.py:ro",
            os.environ.get(
                "ROBOCODE_STRICT_IMAGE", "robocode-strict-blackbox:multi-agent-runtime"
            ),
            "/probe.py",
        ]
    else:
        cmd = [
            "apptainer",
            "exec",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--bind",
            f"{tmp_path}:/sandbox",
            "--bind",
            f"{fixture}:/probe.py:ro",
            "--pwd",
            "/sandbox",
            str(root / "robocode-strict-blackbox.sif"),
            "/opt/robocode-strict/bin/python",
            "/probe.py",
        ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=45, check=False
        )
        (tmp_path / "container.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["fixed"] == fixed
        assert summary["cli_exited"] == fixed
        assert summary["wait_shell_alive"] != fixed
    finally:
        if runtime == "docker":
            subprocess.run(
                ["docker", "rm", "--force", name], capture_output=True, check=False
            )
