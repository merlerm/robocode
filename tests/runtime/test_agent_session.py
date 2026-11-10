"""Turn boundaries, immutable handoffs and accounting without model inference."""

# pylint: disable=missing-function-docstring,protected-access
import json
import subprocess
from dataclasses import replace

import pytest
from omegaconf import OmegaConf

from robocode.runtime.agent_session import SUBMISSION_FILE, AgentSession
from robocode.utils import rate_limit, sandbox
from robocode.utils.backends import create_backend
from robocode.utils.sandbox_types import GenerationMetrics, SandboxConfig, SandboxResult


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def setup(tmp_path):
    session = AgentSession(tmp_path / "agent")
    config = SandboxConfig(
        sandbox_dir=session.workspace,
        output_filename="approach.py",
        max_budget_usd=2,
    )
    backend = create_backend(OmegaConf.create({"backend": "claude"}))
    return session, config, backend


def write_submission(config, *, retained=False):
    sandbox._setup_sandbox_dir(config)
    store = config.sandbox_dir / ".agent_home/projects"
    store.mkdir(parents=True, exist_ok=True)
    (store / "session.jsonl").write_text("{}\n")
    if not retained:
        (config.sandbox_dir / "approach.py").write_text(f"turn = {config.turn_id!r}\n")
        git(config.sandbox_dir, "add", "-A")
        git(config.sandbox_dir, "commit", "-qm", config.turn_id)
    commit = git(config.sandbox_dir, "rev-parse", "HEAD")
    (config.sandbox_dir / SUBMISSION_FILE).write_text(
        json.dumps(
            {
                "turn_id": config.turn_id,
                "commit": commit,
                "retained": retained,
            }
        )
    )
    return SandboxResult(
        True, config.sandbox_dir / "approach.py", None, 0.25, session_id="fixture"
    )


def test_recreate_coordinator_and_retain_submission(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    calls = []

    def run(active, _backend):
        calls.append(active)
        return write_submission(active, retained=bool(len(calls) > 1))

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    first = session.run_turn("p0", config, backend, environment_id="e0")
    second = AgentSession(session.directory).run_turn(
        "p1",
        replace(config, max_budget_usd=3),
        backend,
        environment_id="e1",
    )
    assert [c.resume_previous_session for c in calls] == [False, True]
    assert [c.max_budget_usd for c in calls] == [2, 3]
    assert second["retained"] and second["commit"] == first["commit"]
    assert second["cumulative_cost_usd"] == 0.5
    assert second["environment_id"] == "e1"
    assert (
        session.directory / "turns/p0/artifact/approach.py"
    ).read_text() == "turn = 'p0'\n"


def test_new_turn_does_not_reuse_stale_submission(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    monkeypatch.setattr(
        rate_limit, "_run_active_sandbox", lambda c, b: write_submission(c)
    )
    session.run_turn("p0", config, backend, environment_id="e0")
    monkeypatch.setattr(
        rate_limit,
        "_run_active_sandbox",
        lambda c, b: SandboxResult(
            True, c.sandbox_dir / "approach.py", None, 0.1, session_id="fixture"
        ),
    )
    result = session.run_turn("p1", config, backend, environment_id="e0")
    assert result["outcome"] == "unsubmitted" and "commit" not in result
    assert not (session.workspace / SUBMISSION_FILE).exists()


@pytest.mark.parametrize(
    "problem", ["turn", "commit", "retained", "dirty", "missing", "empty"]
)
def test_reject_invalid_handoff(tmp_path, monkeypatch, problem):
    session, config, backend = setup(tmp_path)

    def run(active, _backend):
        result = write_submission(active)
        path = active.sandbox_dir / SUBMISSION_FILE
        content = json.loads(path.read_text())
        if problem == "turn":
            content["turn_id"] = "old"
        if problem == "commit":
            content["commit"] = "f" * 40
        if problem == "retained":
            content["retained"] = True
        if problem == "dirty":
            (active.sandbox_dir / "approach.py").write_text("changed")
        if problem == "empty":
            (active.sandbox_dir / "approach.py").write_text("")
            git(active.sandbox_dir, "add", "approach.py")
            git(active.sandbox_dir, "commit", "-qm", "empty output")
            content["commit"] = git(active.sandbox_dir, "rev-parse", "HEAD")
        if problem == "missing":
            git(active.sandbox_dir, "rm", "approach.py")
            git(active.sandbox_dir, "commit", "-qm", "remove output")
            content["commit"] = git(active.sandbox_dir, "rev-parse", "HEAD")
        path.write_text(json.dumps(content))
        return result

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    with pytest.raises(ValueError):
        session.run_turn("p0", config, backend, environment_id="e0")
    with pytest.raises(ValueError, match="pending"):
        AgentSession(session.directory).run_turn(
            "p1", config, backend, environment_id="e0"
        )


def test_retry_accounting_uses_each_attempt_once(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    calls = []

    def run(active, _backend):
        calls.append(active)
        if len(calls) == 1:
            sandbox._setup_sandbox_dir(active)
            (active.sandbox_dir / "approach.py").write_text("preserve this edit")
            return SandboxResult(
                False,
                None,
                "token cap",
                0.4,
                output_token_limit_hit=True,
                generation_metrics=GenerationMetrics(),
                session_id="fixture",
            )
        assert (active.sandbox_dir / "approach.py").read_text() == "preserve this edit"
        return write_submission(active)

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    result = session.run_turn("p0", config, backend, environment_id="e0")
    assert [c.max_budget_usd for c in calls] == [2, 1.6]
    assert result["cost_usd"] == pytest.approx(0.65)
    assert len(result["attempts"]) == 2


def test_unknown_spend_blocks_automatic_continuation(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    monkeypatch.setattr(
        rate_limit,
        "_run_active_sandbox",
        lambda c, b: SandboxResult(False, None, "lost usage", None),
    )
    with pytest.raises(ValueError, match="cost"):
        session.run_turn("p0", config, backend, environment_id="e0")
    with pytest.raises(ValueError, match="pending"):
        session.run_turn("p1", config, backend, environment_id="e0")


def test_budget_stop_preserves_last_saved_work(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)

    def run(active, _backend):
        write_submission(active)
        (active.sandbox_dir / SUBMISSION_FILE).unlink()
        (active.sandbox_dir / "approach.py").write_text("latest = True\n")
        return SandboxResult(
            True,
            active.sandbox_dir / "approach.py",
            "budget",
            2.1,
            generation_metrics=GenerationMetrics(stop_reason="error_max_budget_usd"),
            session_id="fixture",
        )

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    result = session.run_turn("p0", config, backend, environment_id="e0")
    assert result["outcome"] == "best_effort"
    assert (
        session.directory / "turns/p0/artifact/approach.py"
    ).read_text() == "latest = True\n"
    assert result["cost_usd"] == 2.1


def test_resume_scaffolding_preserves_files_and_git_head(tmp_path):
    original = tmp_path / "initial.py"
    original.write_text("initial")
    workspace = tmp_path / "workspace"
    config = SandboxConfig(sandbox_dir=workspace, init_files={"approach.py": original})
    sandbox._setup_sandbox_dir(config)
    sandbox._initial_commit(workspace)
    head = git(workspace, "rev-parse", "HEAD")
    (workspace / "approach.py").write_text("agent edit")
    sandbox._setup_sandbox_dir(replace(config, resume_previous_session=True))
    assert (workspace / "approach.py").read_text() == "agent edit"
    assert git(workspace, "rev-parse", "HEAD") == head


def test_session_rejects_changed_access_and_duplicate_turn(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    monkeypatch.setattr(
        rate_limit, "_run_active_sandbox", lambda c, b: write_submission(c)
    )
    session.run_turn("p0", config, backend, environment_id="e0")
    with pytest.raises(ValueError, match="access contract"):
        session.run_turn(
            "p1", replace(config, blackbox=True), backend, environment_id="e0"
        )
    with pytest.raises(FileExistsError):
        session.run_turn("p0", config, backend, environment_id="e0")


def test_missing_session_id_fails_loud(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    monkeypatch.setattr(
        rate_limit,
        "_run_active_sandbox",
        lambda c, b: SandboxResult(True, None, None, 0.1),
    )
    with pytest.raises(ValueError, match="session ID"):
        session.run_turn("p0", config, backend, environment_id="e0")


def test_environment_editor_does_not_need_a_policy(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    config = replace(config, output_filename="environment.py")

    def run(active, _backend):
        result = write_submission(active)
        git(active.sandbox_dir, "mv", "approach.py", "environment.py")
        git(active.sandbox_dir, "commit", "-qm", "environment edit")
        (active.sandbox_dir / SUBMISSION_FILE).write_text(
            json.dumps(
                {
                    "turn_id": active.turn_id,
                    "commit": git(active.sandbox_dir, "rev-parse", "HEAD"),
                    "retained": False,
                }
            )
        )
        return result

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    record = session.run_turn("e0", config, backend, environment_id="base")
    assert record["outcome"] == "submitted"
    assert (session.directory / "turns/e0/artifact/environment.py").is_file()
    assert not (session.directory / "turns/e0/artifact/approach.py").exists()


def test_lock_prevents_overlapping_turns(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)

    def run(active, _backend):
        with pytest.raises(BlockingIOError):
            AgentSession(session.directory).run_turn(
                "p1", config, backend, environment_id="e1"
            )
        return write_submission(active)

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    assert (
        session.run_turn("p0", config, backend, environment_id="e0")["outcome"]
        == "submitted"
    )


def test_missing_native_store_cannot_silently_start_fresh(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    monkeypatch.setattr(
        rate_limit, "_run_active_sandbox", lambda c, b: write_submission(c)
    )
    session.run_turn("p0", config, backend, environment_id="e0")
    (session.workspace / ".agent_home/projects/session.jsonl").unlink()
    with pytest.raises(ValueError, match="store is missing"):
        session.run_turn("p1", config, backend, environment_id="e0")


def test_independent_agent_contexts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rate_limit, "_run_active_sandbox", lambda c, b: write_submission(c)
    )
    for number in range(3):
        session, config, backend = setup(tmp_path / str(number))
        record = session.run_turn("p0", config, backend, environment_id="e0")
        assert record["cumulative_cost_usd"] == 0.25 and not record["resumed"]
