"""Matched seeded exchanges with explicit search conditions and a shared allowance."""

import argparse
import json
import os
from pathlib import Path

from experiments.overnight_exchange import ExperimentExchange, ExperimentSession
from omegaconf import OmegaConf
from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.exchange import ExchangeConfig
from robocode.runtime.snapshots import file_manifest, snapshot_policy, snapshot_runtime
from robocode.runtime.whitebox import load_whitebox_source
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import DockerSandboxConfig


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


class Allowance:
    """Sequential model calls share a ledger; reserve reviews and request overshoot."""

    def __init__(self, path):
        self.path = path

    def before_turn(self, amount):
        ledger = json.loads(self.path.read_text())
        spent = sum(t["cost_usd"] for t in ledger["turns"])
        if spent + amount + ledger["reserve_usd"] > ledger["ceiling_usd"]:
            raise RuntimeError(
                "Night allowance exhausted; no further model call launched"
            )

    def record(self, directory, record):
        ledger = json.loads(self.path.read_text())
        key = str(directory / "turns" / record["turn_id"])
        assert all(t["receipt"] != key for t in ledger["turns"])
        ledger["turns"].append(
            dict(
                receipt=key,
                cost_usd=record["cost_usd"],
                allowance_usd=record["allowance_usd"],
            )
        )
        write_json(self.path, ledger)


class ExplorationSession(ExperimentSession):
    def run_turn(self, turn_id, config, backend, *, environment_id):
        self.owner.allowance.before_turn(config.max_budget_usd)
        record = super().run_turn(
            turn_id, config, backend, environment_id=environment_id
        )
        self.owner.allowance.record(self.directory, record)
        return record


class ExplorationExchange(ExperimentExchange):
    def _validate(self, artifact, name, expected=None):
        if name == "e0":
            receipt = json.loads(self.initial_validation.read_text())
            assert (
                receipt["valid"]
                and receipt["environment_artifact_id"] == artifact.artifact_id
            )
            return receipt
        return super()._validate(artifact, name, expected)


def make_exchange(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    runtime = args.output / "runtime"
    snapshot_runtime(runtime)
    initial = load_environment_artifact(args.artifact)
    manifest = json.loads((args.policy_source / "view.json").read_text())
    view = load_whitebox_source(args.policy_source.resolve(), manifest["source_id"])
    assert view.manifest["artifact_id"] == initial.artifact_id
    assert view.manifest["runtime_source_id"] == file_manifest(runtime)["id"]
    seed_dir = args.output / "seed-policy"
    snapshot_policy(args.seed_policy.resolve(), seed_dir)
    provenance = json.loads((seed_dir / "source-provenance.json").read_text())
    assert provenance["artifact_id"] == initial.artifact_id
    assert provenance["whitebox_source_id"] == view.source_id
    seed = seed_dir / "approach.py"
    seed_files = {
        str(p.relative_to(seed_dir)): p
        for p in seed_dir.rglob("*")
        if p.is_file() and p.name not in {".gitignore", "source-provenance.json"}
    }
    backend = create_backend(OmegaConf.create({"backend": "claude"}))
    common = dict(
        sandbox_dir=args.output,
        model="claude-opus-5-5",
        effort="high",
        telemetry=True,
        docker_image="robocode-sandbox:multi-agent-runtime",
    )
    policy = DockerSandboxConfig(
        **common,
        blackbox=False,
        blackbox_strict=False,
        max_budget_usd=args.policy_budget,
        init_files=seed_files
    )
    editor = DockerSandboxConfig(**common, max_budget_usd=args.editor_budget)
    config = ExchangeConfig(
        backend="docker",
        environment_image=common["docker_image"],
        evaluation_image="robocode-evaluation:multi-agent",
        runtime_source=runtime,
        eval_seeds=list(range(6000, 6010)),
        eval_counts=list(range(5)) * 2,
        cycles=args.cycles,
        editor_attempts=args.editor_attempts,
        policy_access="whitebox",
        render=False,
        timeout=60,
        max_steps=initial.manifest["config"]["base_steps"],
    )
    exchange = ExplorationExchange(
        args.output / "run", initial, config, policy, editor, backend, backend
    )
    exchange.initial_validation = args.initial_validation.resolve()
    exchange.allowance = Allowance(args.budget_ledger.resolve())
    exchange.mixture, exchange.creative = True, False
    exchange.require_action = True
    exchange.seed_policy, exchange.archive = seed, [initial]
    exchange.policy_sources[seed.resolve()] = view
    exchange.editor_pacing = not args.no_editor_pacing
    exchange.editor_instruction = (
        args.editor_instruction_file.read_text() if args.editor_instruction_file else ""
    )
    exchange.policy = ExplorationSession(exchange.policy.directory, exchange, "policy")
    exchange.editor = ExplorationSession(exchange.editor.directory, exchange, "editor")
    exchange.ledger["experiment"] = dict(
        condition=args.condition,
        seed_policy_id=file_manifest(seed_dir)["id"],
        seed_policy_source=str(view.path),
        seed_policy_source_id=view.source_id,
        seed_policy_environment_id=view.manifest["artifact_id"],
        initial_validation=str(exchange.initial_validation),
        policy_context="frozen_P0_code_with_fresh_first_adaptation_context_then_native_resume",
        training="uniform_archive",
        policy_allowance=args.policy_budget,
        editor_allowance=args.editor_budget,
        editor_pacing=exchange.editor_pacing,
        editor_instruction=exchange.editor_instruction,
        editor_only=args.editor_only,
    )
    exchange._save()
    return exchange


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-policy", type=Path, required=True)
    parser.add_argument("--policy-source", type=Path, required=True)
    parser.add_argument("--initial-validation", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--editor-budget", type=float, default=8)
    parser.add_argument("--policy-budget", type=float, default=5)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--editor-attempts", type=int, default=1)
    parser.add_argument("--no-editor-pacing", action="store_true")
    parser.add_argument("--editor-instruction-file", type=Path)
    parser.add_argument("--editor-only", action="store_true")
    parser.add_argument("--token-auth", action="store_true")
    args = parser.parse_args()
    if args.token_auth:
        assert os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"), "Token auth requires a token"
    else:
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    os.environ.pop("ROBOCODE_CLAUDE_AUTH_ALLOW_PARALLEL", None)
    exchange = make_exchange(args)
    if args.editor_only:
        try:
            baseline = exchange._validate(exchange.initial, "e0")
            accepted = exchange._editor_turns(
                exchange.seed_policy, 1, baseline["contract"]
            )
            if accepted:
                exchange._evaluate(exchange.seed_policy, exchange.active, "p0-e1")
                exchange.ledger["status"] = "completed_editor_probe"
                exchange._save()
        except Exception as error:
            exchange.ledger.update(status="failed", error=repr(error))
            exchange._save()
            raise
    else:
        exchange.run()
    print(
        json.dumps(
            dict(
                status=exchange.ledger["status"],
                cost=sum(t["cost_usd"] for t in exchange.ledger["turns"]),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
