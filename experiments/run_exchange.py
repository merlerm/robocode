"""Run a bounded strict-blackbox policy/environment exchange."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.exchange import Exchange, ExchangeConfig
from robocode.runtime.snapshots import snapshot_runtime
from robocode.utils.apptainer_sandbox import ApptainerSandboxConfig
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import DockerSandboxConfig
from robocode.utils.sandbox_types import SandboxConfig


def main() -> None:
    """Take an exported E0 and retain the exchange's revisions, logs and evaluations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--provider", choices=["claude", "codex"], default="claude")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--budget-per-turn", type=float, default=2)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--editor-attempts", type=int, default=1)
    parser.add_argument(
        "--policy-access", choices=["blackbox", "whitebox"], default="blackbox"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004]
    )
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True)
    runtime = args.output.resolve() / "runtime"
    snapshot_runtime(runtime)
    common = {
        "sandbox_dir": args.output,
        "model": args.model,
        "effort": "high",
        "max_budget_usd": args.budget_per_turn,
        "telemetry": True,
    }
    policy: SandboxConfig
    editor: SandboxConfig
    if args.backend == "docker":
        environment_image = "robocode-sandbox:multi-agent-runtime"
        evaluation_image = "robocode-evaluation:multi-agent"
        policy = DockerSandboxConfig(**common, blackbox=True, blackbox_strict=True)
        editor = DockerSandboxConfig(**common, docker_image=environment_image)
    else:
        environment_image = str(root / "robocode-sandbox.sif")
        evaluation_image = str(root / "robocode-evaluation.sif")
        policy = ApptainerSandboxConfig(
            **common,
            blackbox=True,
            blackbox_strict=True,
            strict_sif_path=root / "robocode-strict-blackbox.sif",
        )
        editor = ApptainerSandboxConfig(**common, sif_path=Path(environment_image))
    backend = create_backend(OmegaConf.create({"backend": args.provider}))
    config = ExchangeConfig(
        backend=args.backend,
        environment_image=environment_image,
        evaluation_image=evaluation_image,
        runtime_source=runtime,
        eval_seeds=args.seeds,
        eval_counts=args.counts,
        cycles=args.cycles,
        editor_attempts=args.editor_attempts,
        policy_access=args.policy_access,
        render=not args.no_render,
    )
    ledger = Exchange(
        args.output / "run",
        load_environment_artifact(args.artifact),
        config,
        policy,
        editor,
        backend,
        backend,
    ).run()
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "turn_cost_usd": sum(t["cost_usd"] for t in ledger["turns"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
