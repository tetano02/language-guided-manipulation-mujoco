"""Rule-based segmentation for pick-and-place trajectories."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class SegmentationConfig:
    approach_dist: float = 0.10
    lift_height: float = 0.065
    place_dist: float = 0.07
    goal_progress_delta: float = 0.004


def _as_bool(mapping: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = mapping.get(key, default)
    return bool(value)


def _primitive_for_step(
    obs: Mapping[str, Any],
    *,
    prev_goal_dist: float | None,
    prev_gripper_closed: bool,
    config: SegmentationConfig,
) -> tuple[str, str, float, bool]:
    ee_pos = np.asarray(obs["ee_pos"], dtype=float)
    target_pos = np.asarray(obs["target_pos"], dtype=float)
    goal_pos = np.asarray(obs["goal_pos"], dtype=float)
    gripper = obs.get("gripper_state", {})
    if isinstance(gripper, Mapping):
        gripper_closed = _as_bool(gripper, "closed")
        attached = _as_bool(gripper, "attached")
    else:
        gripper_closed = bool(gripper)
        attached = False
    contacts = obs.get("contacts_summary", {})
    target_contact = bool(contacts.get("target_hand_contact", False))

    ee_target_dist = float(np.linalg.norm(ee_pos - target_pos))
    target_goal_dist = float(np.linalg.norm(target_pos - goal_pos))
    target_height = float(target_pos[2])
    goal_progress = 0.0 if prev_goal_dist is None else prev_goal_dist - target_goal_dist

    if (not gripper_closed) and prev_gripper_closed and target_goal_dist <= config.place_dist:
        primitive = "open_gripper"
        trigger = f"target_goal_dist={target_goal_dist:.3f}<={config.place_dist:.3f}"
    elif not gripper_closed and ee_target_dist > config.approach_dist and not target_contact:
        primitive = "approach"
        trigger = f"ee_target_dist={ee_target_dist:.3f}>{config.approach_dist:.3f}"
    elif not gripper_closed and ee_target_dist <= config.approach_dist and target_height < config.lift_height:
        primitive = "lower"
        trigger = f"ee_target_dist={ee_target_dist:.3f}<={config.approach_dist:.3f}"
    elif gripper_closed and (target_contact or attached) and target_height < config.lift_height:
        primitive = "close_gripper"
        trigger = f"contact={target_contact or attached}"
    elif gripper_closed and target_height >= config.lift_height and goal_progress > config.goal_progress_delta:
        primitive = "move_to_goal"
        trigger = f"goal_progress={goal_progress:.3f}>{config.goal_progress_delta:.3f}"
    elif gripper_closed and target_height >= config.lift_height:
        primitive = "lift"
        trigger = f"target_height={target_height:.3f}>={config.lift_height:.3f}"
    else:
        primitive = "stabilize"
        trigger = "fallback"

    return primitive, trigger, target_goal_dist, gripper_closed


def segment_episode(
    episode: Sequence[Mapping[str, Any]],
    config: SegmentationConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or SegmentationConfig()
    if not episode:
        return []

    labels: list[str] = []
    triggers: list[str] = []
    prev_goal_dist: float | None = None
    prev_gripper_closed = False

    for obs in episode:
        primitive, trigger, prev_goal_dist, prev_gripper_closed = _primitive_for_step(
            obs,
            prev_goal_dist=prev_goal_dist,
            prev_gripper_closed=prev_gripper_closed,
            config=config,
        )
        labels.append(primitive)
        triggers.append(trigger)

    segments: list[dict[str, Any]] = []
    start = 0
    current_label = labels[0]
    current_trigger = triggers[0]
    for index in range(1, len(labels)):
        if labels[index] == current_label:
            continue
        segments.append(
            {
                "primitive": current_label,
                "start": start,
                "end": index - 1,
                "trigger": current_trigger,
            }
        )
        start = index
        current_label = labels[index]
        current_trigger = triggers[index]
    segments.append(
        {
            "primitive": current_label,
            "start": start,
            "end": len(labels) - 1,
            "trigger": current_trigger,
        }
    )
    return segments


def segment_dataset(
    dataset: Sequence[Sequence[Mapping[str, Any]]],
    config: SegmentationConfig | None = None,
) -> list[list[dict[str, Any]]]:
    return [segment_episode(episode, config=config) for episode in dataset]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based segmentation for MuJoCo pick-and-place trajectories.")
    parser.add_argument("--input", type=Path, required=True, help="Path to demos dataset.pkl.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSON file.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    with args.input.open("rb") as handle:
        dataset = pickle.load(handle)
    segmented = segment_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(segmented, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
