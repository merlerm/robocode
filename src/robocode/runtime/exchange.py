"""Alternating policy synthesis and environment editing with explicit acceptance."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from robocode import prompts
from robocode.runtime.agent_session import AgentSession
from robocode.runtime.artifacts import EnvironmentArtifact
from robocode.runtime.candidates import (
    CandidateSourceError,
    export_candidate,
    validate_environment,
)
from robocode.runtime.container_evaluation import evaluate_in_container
from robocode.runtime.editor import EditorBinding, prepare_editor_workspace
from robocode.runtime.environment import environment_service
from robocode.runtime.policy_service import policy_service
from robocode.runtime.whitebox import create_whitebox_source
from robocode.utils.apptainer_sandbox import ApptainerSandboxConfig
from robocode.utils.backends import AgentBackend
from robocode.utils.sandbox_types import SandboxConfig


@dataclass(frozen=True)
class ExchangeConfig:
    """Fixed experiment inputs; agent budgets live in their sandbox configs."""

    backend: str
    environment_image: str
    evaluation_image: str
    runtime_source: Path
    eval_seeds: list[int]
    eval_counts: list[int] | None = None
    cycles: int = 1
    editor_attempts: int = 1
    policy_access: str = "blackbox"
    max_steps: int = 200
    timeout: float = 30
    validation_timeout: float = 60
    render: bool = True


class Exchange:
    """Own accepted revisions, two persistent sessions and held-out evaluations."""

    def __init__(
        self,
        directory: Path,
        initial: EnvironmentArtifact,
        config: ExchangeConfig,
        policy_config: SandboxConfig,
        editor_config: SandboxConfig,
        policy_backend: AgentBackend,
        editor_backend: AgentBackend,
    ) -> None:
        if not policy_config.blackbox or not getattr(
            policy_config, "blackbox_strict", False
        ):
            raise ValueError("The exchange currently requires strict-authored policies")
        if config.policy_access not in {"blackbox", "whitebox"}:
            raise ValueError("Unknown editor policy access")
        if config.cycles < 1 or config.editor_attempts < 1:
            raise ValueError("Exchange and repair limits must be positive")
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True)
        self.initial = self.active = initial
        self.config = config
        self.policy = AgentSession(self.directory / "policy")
        self.editor = AgentSession(self.directory / "editor")
        self.policy_config = replace(policy_config, sandbox_dir=self.policy.workspace)
        self.editor_config = replace(editor_config, sandbox_dir=self.editor.workspace)
        self.policy_backend, self.editor_backend = policy_backend, editor_backend
        self.ledger: dict[str, Any] = {
            "status": "running",
            "initial_environment_id": initial.artifact_id,
            "active_environment_id": initial.artifact_id,
            "turns": [],
            "candidates": [],
            "evaluations": [],
            "cycles": config.cycles,
            "editor_attempts": config.editor_attempts,
            "policy_access": config.policy_access,
        }
        self._save()

    def _save(self) -> None:
        temporary = self.directory / "exchange.tmp"
        temporary.write_text(json.dumps(self.ledger, indent=2) + "\n")
        temporary.replace(self.directory / "exchange.json")

    def _validate(self, artifact: EnvironmentArtifact, name: str, expected=None):
        return validate_environment(
            artifact,
            self.directory / "validation" / name,
            backend=self.config.backend,
            image=self.config.environment_image,
            runtime_source=self.config.runtime_source,
            expected=expected,
            timeout=self.config.validation_timeout,
        )

    def _evaluate(self, policy: Path, artifact: EnvironmentArtifact, name: str) -> None:
        result = evaluate_in_container(
            policy,
            artifact,
            self.directory / "evaluations" / name,
            backend=self.config.backend,
            environment_image=self.config.environment_image,
            evaluation_image=self.config.evaluation_image,
            runtime_source=self.config.runtime_source,
            eval_seeds=self.config.eval_seeds,
            eval_counts=self.config.eval_counts,
            replicate_seed=0,
            max_steps=self.config.max_steps,
            timeout=self.config.timeout,
            render=self.config.render,
            determinism_episodes=0,
        )
        self.ledger["evaluations"].append({"name": name, "result": result})
        self._save()

    def _record_turn(self, record: dict[str, Any]) -> bool:
        self.ledger["turns"].append(record)
        submitted = record["outcome"] == "submitted"
        if not submitted:
            self.ledger["status"] = "stopped_without_submission"
        self._save()
        return submitted

    def _policy_turn(self, name: str) -> Path | None:
        with environment_service(
            json.dumps(self.active.manifest["config"]),
            self.directory / "services" / name,
            telemetry_sink=(
                self.policy.directory / "telemetry/events.jsonl"
                if self.policy_config.telemetry
                else None
            ),
            telemetry_run_id=name,
            artifact=self.active,
            container_backend=self.config.backend,
            image=self.config.environment_image,
            runtime_source=self.config.runtime_source,
            strict=True,
        ) as binding:
            workspace = self.policy.workspace
            workspace.mkdir(exist_ok=True)
            metadata = {
                "action_space": binding.metadata["action_space"],
                "observation_space": binding.metadata["observation_space"],
                "host": binding.client_host(self.config.backend),
                "port": binding.port,
                "token": binding.token,
                "strict": True,
                "primitives": [],
                "max_steps": self.config.max_steps,
                "environment_artifact_id": self.active.artifact_id,
                "request_timeout_s": self.config.timeout,
            }
            if "counts" in binding.metadata:
                metadata["counts"] = binding.metadata["counts"]
            # Refresh transport outside init_files: resumed turns keep their workspace.
            (workspace / "env_spaces.json").write_text(json.dumps(metadata))
            shutil.copy2(
                self.config.runtime_source / "robocode/utils/env_client.py",
                workspace / "env_client.py",
            )
            object_centric = metadata["observation_space"]["type"] == "ObjectCentric"
            interface = prompts.build_interface_spec(
                class_interface=prompts.AGENTIC_CLASS_INTERFACE,
                run_commands=prompts.AGENTIC_RUN_COMMANDS,
                python_executable="/opt/robocode-strict/bin/python",
                primitives_description="No primitives are supplied.",
                blackbox=True,
                object_centric=object_centric,
            )
            prompt = prompts.build_agentic_prompt(
                blackbox=True,
                interface_spec=interface,
                geometry=False,
                modular_code=False,
                env_description=binding.metadata.get("description_blackbox"),
                object_centric=object_centric,
                blackbox_strict=True,
            )
            prompt += (
                "\nThe environment service now serves revision "
                + self.active.artifact_id
                + ". Recreate clients from env_spaces.json; previous clients are "
                "closed. "
                "Improve approach.py for this revision while preserving useful "
                "previous behavior. "
                "This is a small pilot: test briefly, then explicitly submit "
                "before the budget ends."
            )
            active = replace(
                self.policy_config,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=prompts.build_system_prompt(
                    intro=prompts.AGENTIC_INTRO_BLACKBOX,
                    blackbox=True,
                    backend_name=self.policy_backend.name,
                    timeout=self.config.timeout,
                    object_centric=object_centric,
                    blackbox_strict=True,
                ),
            )
            if isinstance(active, ApptainerSandboxConfig):
                active = replace(active, env_server_port=binding.port)
            record = self.policy.run_turn(
                name,
                active,
                self.policy_backend,
                environment_id=self.active.artifact_id,
            )
        if not self._record_turn(record):
            return None
        return self.policy.directory / "turns" / name / "artifact/approach.py"

    def _editor_turns(self, policy: Path, cycle: int, contract: dict[str, Any]) -> bool:
        view = create_whitebox_source(
            self.active,
            self.config.runtime_source,
            self.directory / "views" / f"e{cycle}",
        )
        if not (self.editor.workspace / "source").exists():
            prepare_editor_workspace(view, self.editor.workspace)
        (self.editor.workspace / "environment-config.json").write_text(
            json.dumps(self.active.manifest["config"])
        )
        service = self.directory / "services" / f"frozen-p{cycle-1}"
        with policy_service(
            policy,
            contract,
            service,
            backend=self.config.backend,
            image=self.config.evaluation_image,
            blackbox_strict=True,
            seed=0,
            request_timeout=self.config.timeout,
        ) as binding:
            editor = EditorBinding(
                view,
                binding,
                service / "policy" if self.config.policy_access == "whitebox" else None,
            )
            feedback = ""
            for attempt in range(self.config.editor_attempts):
                name = f"e{cycle}-a{attempt}"
                prompt = (
                    "Edit the simulator under source/ to reduce the frozen "
                    "policy's success. "
                    "Keep the task solvable. You may change simulator code, "
                    "including state-dependent "
                    "disturbances, but preserve action space, observation layout "
                    "and object counts. Test every configured object count, including "
                    "evaluation counts. Report the frozen policy's results and your "
                    "solvability checks separately for each count. "
                    f"Count schedule and step limits: {json.dumps(contract.get('counts'))}. "
                    "Only committed source/ is exported. Runtime and "
                    "environment-config.json are fixed. "
                    "Use /robocode/.venv/bin/python. Instantiate locally with "
                    "hydra.utils.instantiate(OmegaConf.create(json.load(open('envir"
                    "onment-config.json')))). "
                    "Query the frozen policy with "
                    "robocode.utils.policy_client.PolicyClient using "
                    "json.load(open('policy_service.json')): reset(state, info), "
                    "act(state), "
                    "update(next_state, reward, done, info). Call reset for every "
                    "episode. "
                    + (
                        "Read-only policy code is at /frozen-policy. "
                        if self.config.policy_access == "whitebox"
                        else "Policy code is hidden; use the API. "
                    )
                    + "Make a small meaningful edit, test it, then commit and "
                    "explicitly submit before the budget ends. " + feedback
                )
                record = self.editor.run_turn(
                    name,
                    replace(
                        self.editor_config,
                        environment_editor=editor,
                        output_filename="",
                        prompt=prompt,
                    ),
                    self.editor_backend,
                    environment_id=self.active.artifact_id,
                )
                if not self._record_turn(record):
                    return False
                entry = {
                    "turn_id": name,
                    "commit": record["commit"],
                    "parent_environment_id": self.active.artifact_id,
                    "status": "validating",
                }
                self.ledger["candidates"].append(entry)
                self._save()
                candidate = None
                validation: dict[str, Any]
                try:
                    candidate = export_candidate(
                        self.editor.workspace,
                        record["commit"],
                        self.active,
                        self.directory / "environments" / name,
                    )
                except CandidateSourceError as error:
                    validation = {"valid": False, "error": str(error)}
                else:
                    entry["environment_artifact_id"] = candidate.artifact_id
                    validation = self._validate(candidate, name, contract)
                entry["validation"] = validation
                entry["status"] = "accepted" if validation["valid"] else "rejected"
                if validation["valid"]:
                    assert candidate is not None
                    self.active = candidate
                    self.ledger["active_environment_id"] = candidate.artifact_id
                    self._save()
                    return True
                feedback = (
                    "Your previous commit was rejected; its edits remain for repair. "
                    + validation["error"]
                )
                self._save()
        self.ledger["status"] = "stopped_after_rejected_candidates"
        self._save()
        return False

    def run(self) -> dict[str, Any]:
        """Run the finite exchange once; all failures retain receipts and workspaces."""
        try:
            baseline = self._validate(self.initial, "e0")
            if not baseline["valid"]:
                raise ValueError(
                    "Initial environment failed validation: " + baseline["error"]
                )
            policy = self._policy_turn("p0")
            if policy is None:
                return self.ledger
            self._evaluate(policy, self.active, "p0-e0")
            for cycle in range(1, self.config.cycles + 1):
                if not self._editor_turns(policy, cycle, baseline["contract"]):
                    return self.ledger
                self._evaluate(policy, self.active, f"p{cycle-1}-e{cycle}")
                policy = self._policy_turn(f"p{cycle}")
                if policy is None:
                    return self.ledger
                self._evaluate(policy, self.active, f"p{cycle}-e{cycle}")
                self._evaluate(policy, self.initial, f"p{cycle}-e0")
            self.ledger["status"] = "completed"
            self._save()
            return self.ledger
        except Exception as error:
            self.ledger.update(status="failed", error=repr(error))
            for candidate in self.ledger["candidates"]:
                if candidate["status"] == "validating":
                    candidate["status"] = "infrastructure_failed"
            self._save()
            raise
