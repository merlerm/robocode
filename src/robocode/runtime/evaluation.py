"""Shared scoring and determinism checks for frozen generalized policies."""

import logging
from pathlib import Path
from typing import Any

from robocode.environments.variable_count import VariableCountEnv
from robocode.utils.episode import (
    run_episode_with_timeout,
    save_video,
    summarize_by_count,
    summarize_determinism_replays,
    summarize_eval_episodes,
)

logger = logging.getLogger(__name__)


def run_determinism_check(
    env: Any,
    approach: Any,
    eval_seeds: list[int],
    *,
    num_episodes: int,
    max_steps: int,
    timeout: float,
    eval_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Replay the first evaluation seeds twice using the ordinary timeout path."""
    replayed = min(num_episodes, len(eval_seeds))
    if replayed <= 0:
        return {
            "determinism_check_num_episodes": 0,
            "determinism_check_agreement_rate": float("nan"),
            "determinism_check": {
                "num_episodes": 0,
                "agreement_rate": float("nan"),
                "episodes": [],
            },
        }
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(replayed):
        seed = eval_seeds[i]
        count = eval_counts[i] if eval_counts is not None else None
        episode_max_steps = (
            env.max_steps_for_count(count)
            if count is not None and isinstance(env, VariableCountEnv)
            else max_steps
        )
        first: dict[str, Any]
        second: dict[str, Any]
        for attempt in range(2):
            try:
                metrics, _, _ = run_episode_with_timeout(
                    env, approach, seed, episode_max_steps, timeout=timeout, count=count
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # A loaded policy can raise on a replayed seed. Record the crash as
                # that replay's outcome so the check never aborts the completed
                # evaluation before results.json is written.
                logger.exception(
                    "Determinism replay %d for seed %d crashed; recorded as failed",
                    attempt + 1,
                    seed,
                )
                metrics = {
                    "total_reward": None,
                    "num_steps": None,
                    "solved": False,
                    "crashed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if attempt == 0:
                first = metrics
            else:
                second = metrics
        pairs.append((first, second))
    summary = summarize_determinism_replays(pairs)
    return {
        "determinism_check_num_episodes": summary["num_episodes"],
        "determinism_check_agreement_rate": summary["agreement_rate"],
        "determinism_check": summary,
    }


def evaluate_policy(
    env: Any,
    approach: Any,
    eval_seeds: list[int],
    *,
    max_steps: int,
    timeout: float,
    render: bool,
    output_dir: Path,
    eval_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Score held-out episodes, preserving crash and policy-time semantics."""
    per_episode: list[dict[str, Any]] = []
    for i, s in enumerate(eval_seeds):
        logger.info("Evaluating episode %d/%d", i + 1, len(eval_seeds))
        count = eval_counts[i] if eval_counts is not None else None
        episode_max_steps = (
            env.max_steps_for_count(count)
            if count is not None and isinstance(env, VariableCountEnv)
            else max_steps
        )
        try:
            episode_result, frames, _ = run_episode_with_timeout(
                env,
                approach,
                s,
                episode_max_steps,
                timeout=timeout,
                render=render,
                count=count,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # A loaded policy can raise on an unseen eval seed. It counts as
            # unsolved, but there is no per-env worst-case return to charge it
            # with, so it carries no reward or step count and is left out of
            # those means rather than given an invented score.
            logger.exception(
                "Eval episode (seed %d) crashed; scored unsolved, and left out "
                "of the reward and step means",
                s,
            )
            per_episode.append(
                {
                    "total_reward": None,
                    "num_steps": None,
                    "solved": False,
                    "crashed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    **({"object_count": count} if count is not None else {}),
                }
            )
            continue
        per_episode.append(episode_result)
        logger.info(
            "Episode %d/%d: solved=%s, steps=%s",
            i + 1,
            len(eval_seeds),
            episode_result["solved"],
            episode_result["num_steps"],
        )
        if frames:
            video_dir = output_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            save_video(frames, video_dir / f"episode_{i}.gif")

    # Crashed episodes count as unsolved but carry no reward or step count, so
    # the summary reports the two on different denominators and flags the run
    # incomplete. See summarize_eval_episodes.
    results = {**summarize_eval_episodes(per_episode), "per_episode": per_episode}
    num_crashed = results["num_crashed_episodes"]
    if num_crashed:
        logger.warning(
            "%d/%d eval episodes crashed; they count as unsolved in solve_rate "
            "but are excluded from the reward and step means, and the run is "
            "marked eval_complete=false. See per_episode errors in results.json.",
            num_crashed,
            len(per_episode),
        )
    if eval_counts is not None:
        by_count, largest_all, largest_any = summarize_by_count(
            eval_counts, per_episode
        )
        results["by_count"] = by_count
        results["largest_count_all_solved"] = largest_all
        results["largest_count_any_solved"] = largest_any
    return results
