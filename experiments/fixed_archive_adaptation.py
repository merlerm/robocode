"""One policy adaptation from identical seed code against a fixed archive."""

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from experiments.overnight_exchange import ExperimentExchange, ExperimentSession, PACE
from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.exchange import Exchange, ExchangeConfig
from robocode.runtime.snapshots import snapshot_runtime
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import DockerSandboxConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("seed_policy", type=Path)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--mixture", action="store_true")
    parser.add_argument("--budget", type=float, default=5)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    os.environ.pop("ROBOCODE_CLAUDE_AUTH_ALLOW_PARALLEL", None)
    output = args.output.resolve()
    output.mkdir(parents=True)
    runtime = output / "runtime"
    snapshot_runtime(runtime)
    archive = [load_environment_artifact(p) for p in args.artifacts]
    seed = args.seed_policy.resolve()
    policy = DockerSandboxConfig(
        sandbox_dir=output,
        model="claude-opus-5-5",
        effort="high",
        telemetry=True,
        blackbox=True,
        blackbox_strict=True,
        max_budget_usd=args.budget,
        init_files={"approach.py": seed},
    )
    config = ExchangeConfig(
        backend="docker",
        environment_image="robocode-sandbox:multi-agent-runtime",
        evaluation_image="robocode-evaluation:multi-agent",
        runtime_source=runtime,
        eval_seeds=list(range(args.seed_start, args.seed_start + 5 * len(args.counts))),
        eval_counts=args.counts * 5,
        cycles=1,
        policy_access="blackbox",
        render=False,
    )
    backend = create_backend(OmegaConf.create({"backend": "claude"}))
    exchange = ExperimentExchange(
        output / "run", archive[0], config, policy, policy, backend, backend
    )
    exchange.archive, exchange.active = archive, archive[-1]
    exchange.mixture, exchange.creative = args.mixture, False
    exchange.seed_policy = seed
    exchange.policy = ExperimentSession(exchange.policy.directory, exchange, "policy")
    exchange.ledger.update(
        status="running",
        cycles=0,
        active_environment_id=archive[-1].artifact_id,
        experiment={
            "kind": "fixed_archive_adaptation",
            "training": "uniform_archive" if args.mixture else "latest_revision",
            "seed_policy": str(seed),
            "policy_context": "fresh_with_identical_seed_code",
            "archive": [{"path": str(a.path), "id": a.artifact_id} for a in archive],
            "pacing_prompt": PACE,
            "held_out_repeats_per_count": 5,
            "planned_allowance": args.budget,
        },
    )
    exchange._save()
    submitted = exchange._policy_turn("p1")
    if submitted is None:
        exchange.ledger["status"] = "stopped_without_submission"
    else:
        # Use the base evaluator to score each revision exactly once.
        for index, artifact in enumerate(archive):
            Exchange._evaluate(exchange, submitted, artifact, f"p1-e{index}")
        exchange.ledger["status"] = "completed"
    exchange._save()
    print(exchange.ledger["status"], flush=True)


if __name__ == "__main__":
    main()
