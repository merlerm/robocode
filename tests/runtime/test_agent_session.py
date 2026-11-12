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


def read_state(session):
    return json.loads((session.directory / "session.json").read_text())


def assert_finalised(session, record):
    """Check the turn was accounted for exactly like an accepted turn."""
    state = read_state(session)
    ledgers = [
        json.loads((turn / "turn.json").read_text())
        for turn in (session.directory / "turns").iterdir()
    ]
    assert "pending_turn" not in state and state["turns"][-1] == record["turn_id"]
    assert state["session_id"] == "fixture"
    assert state["cost_usd"] == pytest.approx(sum(t["cost_usd"] for t in ledgers))
    assert record["cumulative_cost_usd"] == state["cost_usd"]
    saved = json.loads(
        (session.directory / "turns" / record["turn_id"] / "turn.json").read_text()
    )
    assert saved == record
    return state


@pytest.mark.parametrize(
    "problem, reason",
    [
        ("turn", "names turn"),
        ("commit", "not HEAD"),
        ("short_sha", "not HEAD"),
        ("extra_commit", "not HEAD"),
        ("retained", "retained=false"),
        ("no_retained", "retained=false"),
        ("dirty", "uncommitted"),
        ("symlink", "symlink"),
        ("malformed", "not valid JSON"),
        ("not_object", "JSON object"),
        ("missing", "non-empty approach.py"),
        ("empty", "non-empty approach.py"),
    ],
)
def test_invalid_handoff_is_recorded_and_session_continues(
    tmp_path, monkeypatch, problem, reason
):
    session, config, backend = setup(tmp_path)

    def run(active, _backend):
        result = write_submission(active)
        path = active.sandbox_dir / SUBMISSION_FILE
        content = json.loads(path.read_text())
        if problem == "turn":
            content["turn_id"] = "old"
        if problem == "commit":
            content["commit"] = "f" * 40
        if problem == "short_sha":
            content["commit"] = content["commit"][:7]
        if problem == "extra_commit":
            (active.sandbox_dir / "approach.py").write_text("post-submission fix")
            git(active.sandbox_dir, "commit", "-qam", "post-submission fix")
        if problem == "retained":
            content["retained"] = True
        if problem == "no_retained":
            del content["retained"]
        if problem == "dirty":
            (active.sandbox_dir / "approach.py").write_text("changed")
        if problem == "empty":
            (active.sandbox_dir / "approach.py").write_text("")
            git(active.sandbox_dir, "add", "approach.py")
            git(active.sandbox_dir, "commit", "-qm", "empty output")
            content["commit"] = git(active.sandbox_dir, "rev-parse", "HEAD")
        if problem == "missing":
            git(active.sandbox_dir, "rm", "-q", "approach.py")
            git(active.sandbox_dir, "commit", "-qm", "remove output")
            content["commit"] = git(active.sandbox_dir, "rev-parse", "HEAD")
        path.write_text(json.dumps(content))
        if problem == "malformed":
            path.write_text(json.dumps(content)[:-1])
        if problem == "not_object":
            path.write_text(json.dumps([content]))
        if problem == "symlink":
            target = tmp_path / "handoff.json"
            path.rename(target)
            path.symlink_to(target)
        return result

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    record = session.run_turn("p0", config, backend, environment_id="e0")
    assert record["outcome"] == "rejected" and reason in record["rejection"]
    assert "commit" not in record and "artifact" not in record
    assert "retained" not in record
    assert not (session.directory / "turns/p0/artifact").exists()
    assert assert_finalised(session, record)["commit"] is None

    monkeypatch.setattr(
        rate_limit, "_run_active_sandbox", lambda c, b: write_submission(c)
    )
    following = AgentSession(session.directory).run_turn(
        "p1", config, backend, environment_id="e0"
    )
    assert following["outcome"] == "submitted" and following["resumed"]
    assert not following["retained"] and following["previous_commit"] is None
    state = assert_finalised(session, following)
    assert state["turns"] == ["p0", "p1"] and state["cost_usd"] == 0.5
    assert state["commit"] == following["commit"]


def test_rejected_turn_keeps_last_accepted_commit(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    calls = []

    def run(active, _backend):
        calls.append(active)
        if active.turn_id == "p0":
            return write_submission(active)
        # p1 makes one new commit; p2 and p3 re-submit it without committing.
        result = write_submission(active, retained=active.turn_id != "p1")
        path = active.sandbox_dir / SUBMISSION_FILE
        content = json.loads(path.read_text())
        content["turn_id"] = {"p1": "wrong", "p2": "p2", "p3": "p3"}[active.turn_id]
        content["retained"] = active.turn_id == "p2"
        path.write_text(json.dumps(content))
        return result

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    first = session.run_turn("p0", config, backend, environment_id="e0")
    rejected = session.run_turn("p1", config, backend, environment_id="e0")
    assert rejected["outcome"] == "rejected"
    assert read_state(session)["commit"] == first["commit"]
    head = git(session.workspace, "rev-parse", "HEAD")
    assert head != first["commit"]
    # HEAD differs from the last accepted commit, so it is new work.
    wrong_retained = session.run_turn("p2", config, backend, environment_id="e0")
    assert wrong_retained["outcome"] == "rejected"
    assert "retained=false" in wrong_retained["rejection"]
    accepted = session.run_turn("p3", config, backend, environment_id="e0")
    assert accepted["outcome"] == "submitted" and not accepted["retained"]
    assert accepted["commit"] == head
    assert accepted["previous_commit"] == first["commit"]
    assert [c.resume_previous_session for c in calls] == [False, True, True, True]
    state = assert_finalised(session, accepted)
    assert state["commit"] == head and state["cost_usd"] == 1.0


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


def test_budget_stop_without_output_is_unsubmitted(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)

    def run(active, _backend):
        write_submission(active)
        (active.sandbox_dir / SUBMISSION_FILE).unlink()
        git(active.sandbox_dir, "rm", "-q", "approach.py")
        (active.sandbox_dir / "helper.py").write_text("def act(obs):\n    return 0\n")
        return SandboxResult(
            True,
            None,
            "budget",
            2.0,
            generation_metrics=GenerationMetrics(stop_reason="error_max_budget_usd"),
            session_id="fixture",
        )

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    result = session.run_turn("p0", config, backend, environment_id="e0")
    assert result["outcome"] == "unsubmitted" and "commit" not in result
    assert "Best-effort" in result["rejection"]
    assert not (session.directory / "turns/p0/artifact").exists()
    state = assert_finalised(session, result)
    assert state["commit"] is None and state["cost_usd"] == 2.0
    monkeypatch.setattr(
        rate_limit, "_run_active_sandbox", lambda c, b: write_submission(c)
    )
    following = session.run_turn("p1", config, backend, environment_id="e0")
    assert following["outcome"] == "submitted" and following["resumed"]
    assert assert_finalised(session, following)["cost_usd"] == 2.25


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
