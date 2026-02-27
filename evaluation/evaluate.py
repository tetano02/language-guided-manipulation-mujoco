"""Evaluation pipeline: compare baseline replay vs task-graph abstraction."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from execution.execute_from_graph import NodeConfig, run_episode_from_graph
from mujoco_env.env import EnvConfig, PandaPickPlaceEnv
from utils.logging import StructuredLogger


# ---------------------------------------------------------------------------
# Baseline: replay recorded actions
# ---------------------------------------------------------------------------

def _run_baseline_episode(
    env: PandaPickPlaceEnv,
    demo_episode: list[dict[str, Any]],
    logger: StructuredLogger,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Replay actions from a recorded demo episode."""
    obs = env.reset(seed=env.config.seed)
    total_steps = 0
    grasp_achieved = False
    last_info: dict[str, Any] = {}

    for i, demo_obs in enumerate(demo_episode):
        if total_steps >= max_steps:
            break
        action = np.asarray(demo_obs["action"], dtype=float)
        obs, reward, done, info = env.step(action)
        total_steps += 1
        last_info = info
        if obs["gripper_state"]["attached"]:
            grasp_achieved = True
        if done:
            break

    target = np.asarray(obs["target_pos"], dtype=float)
    goal = np.asarray(obs["goal_pos"], dtype=float)
    final_dist = float(np.linalg.norm(target[:2] - goal[:2]))
    success = bool(
        final_dist < 0.06
        and target[2] < 0.07
        and not obs["gripper_state"]["attached"]
        and obs["gripper_state"]["width"] > 0.008
    )

    return {
        "success": success,
        "steps_used": total_steps,
        "grasp_achieved": grasp_achieved,
        "final_target_goal_dist": final_dist,
        "final_target_z": float(target[2]),
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    *,
    graph: dict[str, Any],
    dataset: list[list[dict[str, Any]]],
    seeds: list[int],
    max_steps: int = 500,
    render: bool = False,
    output_dir: Path = Path("artifacts/evaluation"),
    logger: StructuredLogger,
) -> dict[str, Any]:
    """Run baseline and abstraction on the same seeds, return report."""

    baseline_results: list[dict[str, Any]] = []
    abstraction_results: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{len(seeds)}) ---")

        # --- Baseline ---
        if i < len(dataset):
            demo_episode = dataset[i]
        else:
            # If we have more seeds than demos, use last demo episode
            demo_episode = dataset[-1] if dataset else []

        config_baseline = EnvConfig(
            seed=seed, render=render, max_steps=max_steps,
            output_dir=output_dir, sanity_asserts=False, log_jsonl=False,
        )
        with PandaPickPlaceEnv(config_baseline) as env:
            b_result = _run_baseline_episode(env, demo_episode, logger, max_steps=max_steps)
        b_result["seed"] = seed
        baseline_results.append(b_result)
        b_status = "OK" if b_result["success"] else "FAIL"
        print(f"  Baseline:    {b_status:4s} | steps={b_result['steps_used']:3d} | tgt-goal={b_result['final_target_goal_dist']:.3f}")

        # --- Abstraction ---
        config_abs = EnvConfig(
            seed=seed, render=render, max_steps=max_steps,
            output_dir=output_dir, sanity_asserts=False, log_jsonl=False,
        )
        with PandaPickPlaceEnv(config_abs) as env:
            a_result = run_episode_from_graph(env, graph, logger, max_steps=max_steps)
        a_result["seed"] = seed
        abstraction_results.append(a_result)
        a_status = "OK" if a_result["success"] else "FAIL"
        print(f"  Abstraction: {a_status:4s} | steps={a_result['steps_used']:3d} | tgt-goal={a_result['final_target_goal_dist']:.3f}")

    # --- Aggregation ---
    def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(results)
        successes = [r for r in results if r["success"]]
        n_success = len(successes)
        n_grasp = sum(1 for r in results if r["grasp_achieved"])
        mean_steps_success = float(np.mean([r["steps_used"] for r in successes])) if successes else 0.0
        mean_dist = float(np.mean([r["final_target_goal_dist"] for r in results]))
        return {
            "total": n,
            "success_count": n_success,
            "success_rate": n_success / n if n > 0 else 0.0,
            "grasp_count": n_grasp,
            "grasp_rate": n_grasp / n if n > 0 else 0.0,
            "mean_steps_success": mean_steps_success,
            "mean_target_goal_dist": mean_dist,
        }

    baseline_agg = _aggregate(baseline_results)
    abstraction_agg = _aggregate(abstraction_results)

    report = {
        "seeds": seeds,
        "num_episodes": len(seeds),
        "baseline": {"aggregate": baseline_agg, "episodes": baseline_results},
        "abstraction": {"aggregate": abstraction_agg, "episodes": abstraction_results},
    }

    # --- Print table ---
    print("\n" + "=" * 70)
    print(f"{'Metric':<30s} {'Baseline':>15s} {'Abstraction':>15s}")
    print("-" * 70)
    print(f"{'Success rate':<30s} {baseline_agg['success_rate']:>14.1%} {abstraction_agg['success_rate']:>14.1%}")
    print(f"{'Grasp rate':<30s} {baseline_agg['grasp_rate']:>14.1%} {abstraction_agg['grasp_rate']:>14.1%}")
    print(f"{'Mean steps (success only)':<30s} {baseline_agg['mean_steps_success']:>14.1f} {abstraction_agg['mean_steps_success']:>14.1f}")
    print(f"{'Mean target-goal dist':<30s} {baseline_agg['mean_target_goal_dist']:>14.3f} {abstraction_agg['mean_target_goal_dist']:>14.3f}")
    print("=" * 70)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate baseline replay vs task-graph execution.")
    parser.add_argument("--task-graph", type=Path, required=True, help="Path to task_graph.json.")
    parser.add_argument("--dataset", type=Path, default=Path("demos/dataset.pkl"), help="Path to demos dataset.pkl.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes (seeds) to evaluate.")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (episode i uses seed + i).")
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true", help="Enable MuJoCo viewer.")
    render_group.add_argument("--headless", action="store_true", help="Force headless mode.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"), help="Artifact directory.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # Load task graph
    with args.task_graph.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    # Load dataset
    with args.dataset.open("rb") as f:
        dataset = pickle.load(f)

    seeds = [args.seed + i for i in range(args.episodes)]
    render_mode = bool(args.render and not args.headless)
    logger = StructuredLogger(output_dir=args.output_dir, run_name="evaluation", enabled=True)

    report = run_evaluation(
        graph=graph,
        dataset=dataset,
        seeds=seeds,
        max_steps=args.steps,
        render=render_mode,
        output_dir=args.output_dir,
        logger=logger,
    )

    logger.save_json("report.json", report)
    logger.close()
    print(f"\nReport saved to: {logger.run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
