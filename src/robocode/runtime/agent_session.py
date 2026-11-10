"""Persistent, sequential agent turns with explicit committed submissions."""

from __future__ import annotations

import fcntl
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from robocode.runtime.artifacts import export_git_commit
from robocode.runtime.snapshots import file_manifest
from robocode.utils.apptainer_sandbox import ApptainerSandboxConfig
from robocode.utils.backends import AgentBackend
from robocode.utils.docker_sandbox import DockerSandboxConfig
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox import _final_commit
from robocode.utils.sandbox_types import SandboxConfig, SandboxResult

SUBMISSION_FILE = ".agent_submission.json"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _git(workspace: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(workspace), *args], text=True
    ).strip()


class AgentSession:
    """One workspace and isolated native conversation store; idle between calls."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workspace = self.directory / "workspace"

    def run_turn(
        self,
        turn_id: str,
        config: SandboxConfig,
        backend: AgentBackend,
        *,
        environment_id: str,
    ) -> dict[str, Any]:
        """Run one allowance, preserving context and a coordinator-owned receipt.

        A failed/interrupted coordinator leaves a pending record and blocks automatic
        continuation: its unreported spend must be reconciled before more work.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]+", turn_id):
            raise ValueError("Turn ID must contain only letters, digits, '_' or '-'")
        if config.sandbox_dir.resolve() != self.workspace:
            raise ValueError("Turn config must use the session workspace")
        if not math.isfinite(config.max_budget_usd) or config.max_budget_usd <= 0:
            raise ValueError("Each turn needs a finite positive dollar allowance")
        if backend.name not in {"claude", "codex"}:
            raise ValueError("Persistent turns currently support Claude and Codex")
        output = Path(config.output_filename)
        if output.is_absolute() or ".." in output.parts:
            raise ValueError("Required output must be relative to the workspace")
        if not environment_id:
            raise ValueError("A turn must identify its environment revision")
        with (self.directory / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._run_turn(turn_id, config, backend, environment_id)

    def _run_turn(
        self,
        turn_id: str,
        config: SandboxConfig,
        backend: AgentBackend,
        environment_id: str,
    ) -> dict[str, Any]:
        state_path = self.directory / "session.json"
        identity = {
            "backend": backend.name,
            "model": config.model,
            "effort": config.effort,
            "runtime": type(config).__name__,
            "blackbox": config.blackbox,
            "strict": getattr(config, "blackbox_strict", False),
            "output_filename": config.output_filename,
        }
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "identity": identity,
                "turns": [],
                "cost_usd": 0.0,
                "commit": None,
            }
        )
        if state.get("pending_turn"):
            raise ValueError(
                "Previous turn is pending; inspect its ledger before resuming"
            )
        if state["identity"] != identity:
            raise ValueError(
                "Cannot change a session's provider, model or access contract"
            )
        resumed = bool(state["turns"])
        store = self.workspace / (
            ".agent_home/projects"
            if not isinstance(config, (DockerSandboxConfig, ApptainerSandboxConfig))
            and backend.name == "claude"
            else ".agent_sessions"
        )
        if resumed and not any(store.rglob("*.jsonl")):
            raise ValueError("Native conversation store is missing")
        if not resumed and any(store.rglob("*.jsonl")):
            raise ValueError("Cannot adopt an unrecorded native conversation store")
        turn_dir = self.directory / "turns" / turn_id
        turn_dir.mkdir(parents=True)
        state["pending_turn"] = turn_id
        _write_json(state_path, state)
        submission = self.workspace / SUBMISSION_FILE
        submission.unlink(missing_ok=True)
        contract = (
            f"Coordinator turn: {turn_id}. Environment revision: {environment_id}. "
            f"This turn has a fresh ${config.max_budget_usd:.2f} allowance, shared by "
            "all work and retries within this turn. "
            "Keep your existing context and files. "
            "When ready, commit your work and write " + SUBMISSION_FILE + " containing "
            f'{{"turn_id": "{turn_id}", "commit": "<full HEAD SHA>", '
            '"retained": false}. Use retained=true only when explicitly submitting '
            "the previous handoff commit unchanged. Do not modify tracked files after "
            "submission. A normal CLI exit alone does not submit work. Keep your "
            "best working files on disk if the budget runs out."
        )
        active = replace(
            config,
            turn_id=turn_id,
            resume_previous_session=resumed,
            session_id=state.get("session_id"),
            init_files={} if resumed else config.init_files,
            prompt=config.prompt + "\n\n" + contract,
        )
        stream_path = self.directory / "stream.jsonl"
        stream_start = stream_path.stat().st_size if stream_path.exists() else 0
        record: dict[str, Any] = {
            "agent_id": self.directory.name,
            "stream_start_bytes": stream_start,
            "turn_id": turn_id,
            "environment_id": environment_id,
            "allowance_usd": config.max_budget_usd,
            "resumed": resumed,
            "previous_commit": state["commit"],
            "attempts": [],
        }
        record_path = turn_dir / "turn.json"
        _write_json(record_path, record)

        def record_attempt(result: SandboxResult) -> None:
            record["attempts"].append(
                {
                    "cost_usd": result.total_cost_usd,
                    "session_id": result.session_id,
                    "error": result.error,
                    "metrics": (
                        asdict(result.generation_metrics)
                        if result.generation_metrics
                        else None
                    ),
                }
            )
            _write_json(record_path, record)
            cost = result.total_cost_usd
            if cost is None or not math.isfinite(cost) or cost < 0:
                raise ValueError(
                    "Turn attempt has no valid cost; refusing further spend"
                )

            if not result.session_id:
                raise ValueError("Turn did not report its native session ID")

        result = run_with_rate_limit_retry(
            None,
            active,
            backend,
            on_attempt=record_attempt,
        )
        if stream_path.exists():
            with (
                stream_path.open("rb") as source,
                (turn_dir / "stream.jsonl").open("wb") as target,
            ):
                source.seek(stream_start)
                shutil.copyfileobj(source, target)
            record["stream_end_bytes"] = stream_path.stat().st_size
        record["cost_usd"] = sum(a["cost_usd"] for a in record["attempts"])
        record["stop_reason"] = (
            result.generation_metrics.stop_reason if result.generation_metrics else None
        )
        record["error"] = result.error
        record["outcome"] = "unsubmitted"
        if submission.is_symlink():
            raise ValueError("Submission cannot be a symlink")
        if submission.exists():
            handoff = json.loads(submission.read_text())
            commit = _git(self.workspace, "rev-parse", "HEAD")
            if handoff.get("turn_id") != turn_id or handoff.get("commit") != commit:
                raise ValueError("Submission must name this turn and current HEAD")
            retained = commit == state["commit"]
            if (
                not isinstance(handoff.get("retained"), bool)
                or handoff["retained"] != retained
            ):
                raise ValueError(
                    "Submission must explicitly distinguish retained/new work"
                )
            if _git(
                self.workspace, "status", "--porcelain", "--untracked-files=normal"
            ):
                raise ValueError("Submission has uncommitted changes")
            record["outcome"] = "submitted"
            record["retained"] = retained
        elif record["stop_reason"] in {"error_max_budget_usd", "error_max_turns"} or (
            result.generation_metrics and result.generation_metrics.turn_limit_hit
        ):
            # Preserve the existing best-effort behavior, without calling it submitted.
            _final_commit(self.workspace)
            commit = _git(self.workspace, "rev-parse", "HEAD")
            record["outcome"] = "best_effort"
        else:
            commit = None
        if commit is not None:
            snapshot = turn_dir / "artifact"
            export_git_commit(self.workspace, commit, snapshot)
            output = snapshot / config.output_filename
            if config.output_filename and (
                not output.is_file() or output.stat().st_size == 0
            ):
                raise ValueError(
                    "Selected commit does not contain the required output file"
                )
            record["commit"] = commit
            record["artifact"] = file_manifest(snapshot)
            state["commit"] = commit
        state["session_id"] = record["attempts"][-1]["session_id"]
        state["turns"].append(turn_id)
        state["cost_usd"] += record["cost_usd"]
        state.pop("pending_turn")
        record["cumulative_cost_usd"] = state["cost_usd"]
        _write_json(record_path, record)
        _write_json(state_path, state)
        return record
