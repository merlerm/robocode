"""Run an experiment with a given approach and environment.

See README.md for the paper's strict-black-box commands, five-replicate sweeps, planner
settings, and reevaluation. The runner rejects the legacy local transport for generated-
code experiments. Docker supports agentic and LLMGenPlan runs; isolated Apptainer
supports Claude/Codex agentic runs.
"""

import _thread
import json
import logging
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from robocode.environments.variable_count import VariableCountEnv
from robocode.experiment_protocol import EXPERIMENT_ID_PATTERN, validate_eval_seed
from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.container_evaluation import evaluate_in_container
from robocode.runtime.containers import image_identity
from robocode.runtime.environment import environment_service
from robocode.runtime.evaluation import evaluate_policy, run_determinism_check
from robocode.runtime.remote_environment import RemoteEnvironment
from robocode.runtime.snapshots import snapshot_runtime
from robocode.runtime.whitebox import create_whitebox_source
from robocode.utils.approach_history import get_snapshots, record_episodes
from robocode.utils.episode import (
    run_per_instance_eval,
    summarize_count_regimes,
)
from robocode.utils.telemetry import SINK_FILENAME, require_registered

logger = logging.getLogger(__name__)


def resolve_experiment_id(cfg: DictConfig) -> str | None:
    """Return a tracker-generated condition ID and reject unsafe identifiers."""
    value = cfg.get("experiment_id")
    if value is None:
        return None
    if not isinstance(value, str) or EXPERIMENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "experiment_id must be null or a tracker-generated identifier, "
            f"got {value!r}"
        )
    return value


def resolve_eval_seed(cfg: DictConfig) -> int:
    """Return the required, experimenter-only evaluation-suite seed."""
    return validate_eval_seed(cfg.get("eval_seed"))


def generate_eval_seeds(eval_seed: int, num_eval_tasks: int) -> list[int]:
    """Derive the fixed episode suite from the protocol seed."""
    if num_eval_tasks <= 0:
        raise ValueError("num_eval_tasks must be positive")
    task_rng = np.random.default_rng(eval_seed)
    return [int(task_rng.integers(0, 2**63)) for _ in range(num_eval_tasks)]


def validate_eval_seed_isolation(cfg: DictConfig) -> None:
    """Reject a synthesis backend that can read experimenter-side files.

    Hydra writes ``eval_seed`` into the run directory before ``_main`` starts.
    The local coding-agent sandbox permits host filesystem reads, whereas Docker
    and Apptainer mount only the child synthesis directory and filtered source.
    """
    if cfg.approach.get("container_backend") == "local":
        raise ValueError(
            "container_backend=local cannot isolate eval_seed from generated-code "
            "methods; use docker or apptainer"
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _main(cfg: DictConfig) -> float:
    """Run a single experiment."""
    with ExitStack() as resources:
        return _run_experiment(cfg, resources)


def _run_experiment(cfg: DictConfig, resources: ExitStack) -> float:
    """Own services independently of synthesis and evaluation lifetimes."""
    validate_eval_seed_isolation(cfg)
    replicate_seed = cfg.replicate_seed
    eval_seed = resolve_eval_seed(cfg)
    condition_id = resolve_experiment_id(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if condition_id is not None:
        logger.info("Experiment ID: %s", condition_id)

    if cfg.eval_timeout is None:
        raise ValueError(
            "eval_timeout must be set to a number of seconds; it bounds every "
            "evaluation rollout. Use a large value for an effectively unlimited budget."
        )

    environment_binding = None
    whitebox_source = None
    runtime_source = None
    runtime = cfg.get("environment_runtime")
    if runtime is not None:
        if runtime.backend != cfg.approach.get("container_backend"):
            raise ValueError(
                "Artifact service and policy must use the same container backend"
            )
        if (
            cfg.approach.get("_target_")
            != "robocode.approaches.agentic_approach.AgenticApproach"
        ):
            raise ValueError(
                "Artifact runner currently supports the generalized agentic approach"
            )
        blackbox = bool(cfg.approach.get("blackbox", False))
        strict = bool(cfg.approach.get("blackbox_strict", False))
        if blackbox != strict or cfg.primitives:
            raise ValueError(
                "Artifact runner requires whitebox or strict blackbox, "
                "with primitives=[]"
            )
        if cfg.record_approach_history:
            raise ValueError(
                "Artifact runner does not yet support approach-history replay"
            )
        evaluation_image = runtime.evaluation_image
        evaluation_identity = image_identity(runtime.backend, evaluation_image)
        artifact = load_environment_artifact(
            Path(runtime.artifact), expected_id=runtime.expected_id
        )
        if not blackbox and evaluation_identity != artifact.manifest["runtime_image"]:
            raise ValueError(
                "Whitebox evaluation must use the environment artifact image"
            )
        runtime_source = output_dir / "service-runtime"
        runtime_manifest = snapshot_runtime(runtime_source)
        (output_dir / "service-runtime.json").write_text(
            json.dumps(runtime_manifest, indent=2) + "\n"
        )
        if not blackbox:
            whitebox_source = create_whitebox_source(
                artifact, runtime_source, output_dir / "whitebox-source"
            )
        environment_binding = resources.enter_context(
            environment_service(
                json.dumps(OmegaConf.to_container(cfg.environment, resolve=True)),
                output_dir / "services" / "development",
                artifact=artifact,
                runtime_source=runtime_source,
                container_backend=runtime.backend,
                image=runtime.image,
                strict=True,
                telemetry_sink=(
                    output_dir / "telemetry" / SINK_FILENAME
                    if cfg.approach.get("telemetry", False)
                    else None
                ),
                telemetry_run_id=output_dir.name,
                on_failure=_thread.interrupt_main,
            )
        )
        env = RemoteEnvironment(environment_binding, cfg.max_steps)
        (output_dir / "environment-artifact.json").write_text(
            json.dumps(artifact.manifest, indent=2) + "\n"
        )
    else:
        env = hydra.utils.instantiate(cfg.environment)
    resources.callback(env.close)

    # Whitebox telemetry instruments specific registered env classes, so an
    # unregistered env would silently log nothing -- fail loud instead. Blackbox
    # instruments generically at the env server, so it needs no registration.
    if cfg.approach.get("telemetry", False) and not cfg.approach.get("blackbox", False):
        require_registered(cfg.environment["_target_"])

    # If the environment provides a description (e.g. kinder envs), write it
    # to a file so the agentic approach can read it in its sandbox. In blackbox
    # mode, use the variant that omits source-code pointers and direct-import
    # examples, since the agent has access to neither.
    blackbox = bool(cfg.approach.get("blackbox", False))
    description = env.env_description_blackbox if blackbox else env.env_description
    env_description_path: str | None = None
    if description is not None:
        desc_path = output_dir / "env_description.md"
        desc_path.write_text(description)
        env_description_path = str(desc_path)

    primitives = {}
    if runtime is None:
        from robocode.primitives import (  # pylint: disable=import-outside-toplevel
            build_primitives,
        )

        primitives = build_primitives(env, cfg.primitives, blackbox=blackbox)

    if blackbox and environment_binding is None:
        environment_binding = resources.enter_context(
            environment_service(
                json.dumps(OmegaConf.to_container(cfg.environment, resolve=True)),
                output_dir / "services" / "development",
                telemetry_sink=(
                    output_dir / "telemetry" / SINK_FILENAME
                    if cfg.approach.get("telemetry", False)
                    else None
                ),
                telemetry_run_id=output_dir.name,
                strict=bool(cfg.approach.get("blackbox_strict", False)),
                on_failure=_thread.interrupt_main,
            )
        )

    # Write env config for MCP server (if mcp_tools are configured).
    mcp_tools = tuple(cfg.get("mcp_tools", []))
    if mcp_tools:
        env_config_path = output_dir / "env_config.json"
        env_config_path.write_text(
            json.dumps(OmegaConf.to_container(cfg.environment, resolve=True))
        )

    # The Hydra environment choice name (e.g. "obstruction2d_medium") is used
    # to locate pre-written helper files in src/robocode/primitives/<env_name>/.
    env_name = HydraConfig.get().runtime.choices.get("environment")

    # primitives can hold live runtime objects (e.g. the bilevel_models
    # SesameModels dataclass) that OmegaConf tries to structure, and rejects, when
    # merging kwargs into the approach config. Build the approach as a partial from
    # the config-safe arguments and pass primitives straight to the constructor.
    approach_ctor = hydra.utils.instantiate(
        cfg.approach,
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=replicate_seed,
        env_description_path=env_description_path,
        mcp_tools=mcp_tools,
        env_name=env_name,
        env=env,
        # JSON (not the DictConfig) so hydra.utils.instantiate doesn't recursively
        # instantiate it; the llm_genplan docker driver rebuilds the env from it.
        env_cfg=json.dumps(OmegaConf.to_container(cfg.environment, resolve=True)),
        max_steps=cfg.max_steps,
        eval_timeout=cfg.eval_timeout,
        _partial_=True,
    )
    approach = approach_ctor(
        primitives=primitives,
        environment_binding=environment_binding,
        whitebox_source=whitebox_source,
        environment_image=runtime.image if runtime is not None else None,
    )

    num_eval = cfg.num_eval_tasks

    # A variable-count env sweeps its configured object counts evenly across the eval
    # set, so program and planner face the same (seed, count) instances and a
    # solve-rate-vs-count curve can be reported. Unpinned resets sample
    # design_counts; explicit reset options can request other counts during
    # synthesis too, so the configured split is not evidence of non-exposure.
    eval_counts: list[int] | None = None
    if isinstance(env, VariableCountEnv):
        count_pool = list(env.eval_counts)
        eval_counts = [count_pool[i % len(count_pool)] for i in range(num_eval)]

    # Two eval lifecycles, chosen explicitly. Per-instance approaches spend a
    # global agent budget seed by seed (no single train() step); generalized
    # approaches train once and are then rolled out for free over every seed.
    if approach.per_instance:
        eval_seeds = generate_eval_seeds(eval_seed, num_eval)
        if cfg.record_approach_history:
            raise ValueError(
                "record_approach_history is not supported for per-instance "
                "approaches (it replays a single sandbox's git history; "
                "per-instance approaches use a separate sandbox per seed)"
            )
        results: dict[str, Any] = run_per_instance_eval(
            env,
            approach,
            eval_seeds,
            max_budget_usd=cfg.approach.max_budget_usd,
            output_dir=output_dir,
            max_budget_per_instance_usd=cfg.approach.get(
                "max_budget_per_instance_usd", None
            ),
            render=cfg.render_videos,
            eval_counts=eval_counts,
        )
        # Surface the accumulated cross-seed cost via the shared reporting hook.
        approach.total_cost_usd = results.pop("total_cost_usd")
        mean_reward = results["mean_eval_reward"]
        mean_steps = results["mean_eval_steps"]
        solve_rate = results["solve_rate"]
    elif runtime is not None:
        policy_path = approach.synthesize()
        eval_seeds = generate_eval_seeds(eval_seed, num_eval)
        results = evaluate_in_container(
            policy_path,
            artifact,
            output_dir / "evaluation",
            backend=runtime.backend,
            environment_image=runtime.image,
            evaluation_image=(
                evaluation_identity if runtime.backend == "docker" else evaluation_image
            ),
            eval_seeds=eval_seeds,
            replicate_seed=replicate_seed,
            max_steps=cfg.max_steps,
            timeout=cfg.eval_timeout,
            render=cfg.render_videos,
            determinism_episodes=cfg.get("determinism_check_num_episodes", 10),
            runtime_source=runtime_source,
            whitebox_source=whitebox_source,
        )
        videos = output_dir / "evaluation" / "output" / "videos"
        if videos.exists():
            shutil.copytree(videos, output_dir / "videos")
        mean_reward = results["mean_eval_reward"]
        mean_steps = results["mean_eval_steps"]
        solve_rate = results["solve_rate"]
    else:
        approach.train()

        # Derive the held-out episode suite only after synthesis has finished.
        # The fixed master seed remains in the experimenter process and is never
        # placed in the agent's prompt, environment, command line, or sandbox.
        eval_seeds = generate_eval_seeds(eval_seed, num_eval)

        # Record approach history: replay every sandbox snapshot.
        if cfg.record_approach_history:
            load_dir = cfg.approach.get("load_dir", None)
            sandbox_dir = (
                Path(load_dir) / "sandbox" if load_dir else output_dir / "sandbox"
            )
            snapshots = get_snapshots(sandbox_dir)
            record_episodes(
                snapshots,
                sandbox_dir,
                env,
                primitives,
                replicate_seed,
                max_steps=cfg.max_steps,
                output_dir=output_dir,
            )

        # Evaluate on held-out episodes.
        logger.info(
            "Training complete; starting %d held-out evaluation episodes", num_eval
        )
        render = cfg.render_videos
        results = evaluate_policy(
            env,
            approach,
            eval_seeds,
            max_steps=cfg.max_steps,
            timeout=cfg.eval_timeout,
            render=render,
            output_dir=output_dir,
            eval_counts=eval_counts,
        )
        mean_reward = results["mean_eval_reward"]
        mean_steps = results["mean_eval_steps"]
        solve_rate = results["solve_rate"]

    # A determinism check replays a few eval episodes twice and records the
    # agreement rate. Defaulted on because it is cheap (no LLM cost; only the
    # frozen policy is rolled out) and it is the only way a wall-clock-budgeted
    # policy becomes visible in the run rather than during analysis. Set
    # determinism_check_num_episodes=0 to disable. Per-instance approaches are
    # skipped: re-running them spends eval-time agent budget per seed, which is
    # a different (paid) measurement rather than a free replay.
    determinism_num = cfg.get("determinism_check_num_episodes", 10)
    if runtime is None and not approach.per_instance and determinism_num > 0:
        logger.info(
            "Running determinism check: replaying %d eval episodes twice",
            min(determinism_num, num_eval),
        )
        results.update(
            run_determinism_check(
                env,
                approach,
                eval_seeds,
                num_episodes=determinism_num,
                max_steps=cfg.max_steps,
                timeout=cfg.eval_timeout,
                eval_counts=eval_counts,
            )
        )
        logger.info(
            "Determinism check: %.3f agreement over %d replayed episodes",
            results["determinism_check_agreement_rate"],
            results["determinism_check_num_episodes"],
        )
    if isinstance(env, VariableCountEnv):
        count_regimes = summarize_count_regimes(results["by_count"], env.design_counts)
        results["count_regimes"] = count_regimes
        results["design_count_solve_rate"] = count_regimes["design"]["solve_rate"]
        results["held_out_count_solve_rate"] = count_regimes["held_out"]["solve_rate"]
    agent_cost = getattr(approach, "total_cost_usd", None)
    if agent_cost is not None:
        results["agent_cost_usd"] = agent_cost
    num_generations = getattr(approach, "num_generations", None)
    if num_generations is not None:
        results["num_generations"] = num_generations
    gen_metrics = getattr(approach, "generation_metrics", None)
    if gen_metrics is not None:
        results.update(gen_metrics.to_dict())
    if environment_binding is not None and environment_binding.artifact_id is not None:
        results["environment_artifact_id"] = environment_binding.artifact_id
        results["environment_runtime_image"] = artifact.manifest["runtime_image"]
        results["environment_runtime_source_id"] = environment_binding.metadata[
            "runtime_source_id"
        ]
        if "determinism_check" in results:
            results["determinism_check"][
                "environment_artifact_id"
            ] = environment_binding.artifact_id
        for episode in results.get("per_episode", []):
            episode["environment_artifact_id"] = environment_binding.artifact_id
            if "policy_artifact_id" in results:
                episode["policy_artifact_id"] = results["policy_artifact_id"]
        if "policy_artifact_id" in results and "determinism_check" in results:
            results["determinism_check"]["policy_artifact_id"] = results[
                "policy_artifact_id"
            ]
    results["eval_seed"] = eval_seed
    results["replicate_seed"] = replicate_seed
    if condition_id is not None:
        results["experiment_id"] = condition_id
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    logger.info(
        "Mean reward: %.1f, Mean steps: %.1f, Solve rate: %.2f",
        mean_reward,
        mean_steps,
        solve_rate,
    )
    return mean_reward


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
