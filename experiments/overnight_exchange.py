"""Shorter research exchanges, optionally training on a uniform revision archive."""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from experiments.mixture_service import uniform_mixture_service
from robocode.runtime.agent_session import AgentSession
from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.exchange import Exchange, ExchangeConfig
from robocode.runtime.snapshots import snapshot_runtime
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import DockerSandboxConfig

PACE = (
    "\nThis is a short exchange turn. Aim to submit within about 15 minutes. "
    "Use compact diagnostics, at most 10 episodes per configured object count "
    "per candidate, and one final verification sweep. Once a working candidate "
    "is found, commit and explicitly submit it. Do not run hundreds of episodes "
    "or optimize further just to use the remaining budget."
)
CREATIVE = (
    "\nExplore qualitatively different code changes across turns. Consider "
    "spatial layout, obstacle geometry and placement, interaction rules, and "
    "state-dependent dynamics. Do not default only to progressively tighter "
    "numerical tolerances or disturbances; choose a meaningful, solvable challenge. "
    "Keep reward, termination and task-success definitions unchanged; introduce "
    "difficulty through layout or transition dynamics."
)


class ExperimentSession(AgentSession):
    def __init__(self, directory, owner, role):
        super().__init__(directory)
        self.owner, self.role = owner, role

    def run_turn(self, turn_id, config, backend, *, environment_id):
        prompt = config.prompt + PACE
        if self.role == "policy" and self.owner.policy_instruction:
            prompt += "\nTask instruction: " + self.owner.policy_instruction
        if self.role == "editor" and self.owner.creative:
            prompt += CREATIVE
        members = []
        if self.role == "policy" and self.owner.mixture:
            members = [a.artifact_id for a in self.owner.archive]
            prompt += (
                "\nEach reset samples uniformly from all accepted environment "
                "revisions so far, including the original, and keeps that revision "
                "fixed for the episode. Fixed seeds reproduce the selected revision. "
                "No revision label is provided. Develop one policy that works across "
                "this mixture; infer relevant behavior from observations and interaction."
            )
            path = self.workspace / "env_spaces.json"
            meta = json.loads(path.read_text())
            meta["training_distribution"] = {"kind": "uniform", "members": members}
            meta["environment_artifact_id"] = self.owner.mixture_id
            path.write_text(json.dumps(meta))
        started = time.time()
        record = super().run_turn(
            turn_id,
            replace(config, prompt=prompt),
            backend,
            environment_id=environment_id,
        )
        record.update(
            wall_time_s=time.time() - started,
            training_environment_ids=members,
            prompt_condition="creative" if self.owner.creative else "neutral",
        )
        (self.directory / "turns" / turn_id / "turn.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return record


class ExperimentExchange(Exchange):
    policy_instruction = ""

    def _mixture_service(self, _config_json, service_dir, *, artifact, **kwargs):
        return uniform_mixture_service(self.archive, service_dir, **kwargs)

    def _policy_turn(self, name):
        if name == "p0" and self.seed_policy is not None:
            return self.seed_policy
        if not self.mixture:
            return super()._policy_turn(name)
        # Only synthesis gets the mixture. Evaluations remain pinned to one revision.
        original = self._mixture_service
        from contextlib import contextmanager

        @contextmanager
        def service(*args, **kwargs):
            with original(*args, **kwargs) as binding:
                self.mixture_id = binding.artifact_id
                yield binding

        with patch("robocode.runtime.exchange.environment_service", service):
            return super()._policy_turn(name)

    def _editor_turns(self, policy, cycle, contract):
        accepted = super()._editor_turns(policy, cycle, contract)
        if accepted:
            self.archive.append(self.active)
        return accepted

    def _evaluate(self, policy, artifact, name):
        super()._evaluate(policy, artifact, name)
        # Add all intermediate revisions after each adapted policy, without
        # supplying held-out scores to either development agent.
        if name.endswith("-e0") and name != "p0-e0":
            policy_name = name.split("-")[0]
            for index, earlier in enumerate(self.archive[1:-1], 1):
                super()._evaluate(policy, earlier, f"{policy_name}-e{index}-retention")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-policy", type=Path)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--policy-budget", type=float, default=5)
    parser.add_argument("--editor-budget", type=float, default=2)
    parser.add_argument("--mixture", action="store_true")
    parser.add_argument("--creative", action="store_true")
    parser.add_argument("--policy-instruction", default="")
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed-start", type=int, default=6000)
    args = parser.parse_args()
    # Use the existing serialized, refreshable, credentials-only mechanism.
    os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    os.environ.pop("ROBOCODE_CLAUDE_AUTH_ALLOW_PARALLEL", None)
    args.output.mkdir(parents=True)
    runtime = args.output.resolve() / "runtime"
    snapshot_runtime(runtime)
    initial = load_environment_artifact(args.artifact)
    seed = args.seed_policy.resolve() if args.seed_policy else None
    common = dict(
        sandbox_dir=args.output, model="claude-opus-5-5", effort="high", telemetry=True
    )
    policy = DockerSandboxConfig(
        **common,
        blackbox=True,
        blackbox_strict=True,
        max_budget_usd=args.policy_budget,
        init_files={"approach.py": seed} if seed else {},
    )
    editor = DockerSandboxConfig(
        **common, docker_image="robocode-sandbox:multi-agent-runtime", max_budget_usd=2
    )
    backend = create_backend(OmegaConf.create({"backend": "claude"}))
    config = ExchangeConfig(
        backend="docker",
        environment_image="robocode-sandbox:multi-agent-runtime",
        evaluation_image="robocode-evaluation:multi-agent",
        runtime_source=runtime,
        eval_seeds=list(range(args.seed_start, args.seed_start + 2 * len(args.counts))),
        eval_counts=args.counts * 2,
        cycles=args.cycles,
        editor_attempts=1,
        policy_access="blackbox",
        render=False,
    )
    exchange = ExperimentExchange(
        args.output / "run", initial, config, policy, editor, backend, backend
    )
    exchange.mixture, exchange.creative = args.mixture, args.creative
    exchange.policy_instruction = args.policy_instruction
    exchange.seed_policy, exchange.archive = seed, [initial]
    exchange.policy = ExperimentSession(exchange.policy.directory, exchange, "policy")
    exchange.editor = ExperimentSession(exchange.editor.directory, exchange, "editor")
    exchange.ledger["experiment"] = {
        "training": "uniform_archive" if args.mixture else "latest_revision",
        "editor_prompt": "creative" if args.creative else "neutral",
        "pacing_prompt": PACE,
        "creative_prompt": CREATIVE if args.creative else None,
        "policy_instruction": args.policy_instruction,
        "seed_policy": str(seed) if seed else None,
        "policy_context": "fresh_at_first_adaptation_then_native_resume",
        "held_out_repeats_per_count": 2,
        "policy_allowance": args.policy_budget,
        "editor_allowance": args.editor_budget,
        "planned_allowance": (args.policy_budget + args.editor_budget) * args.cycles
        + (0 if seed else args.policy_budget),
    }
    exchange._save()
    print(json.dumps(exchange.ledger["experiment"]), flush=True)
    result = exchange.run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "cost": sum(t["cost_usd"] for t in result["turns"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
