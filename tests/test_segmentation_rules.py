from __future__ import annotations

import numpy as np

from segmentation.segment import segment_episode


def _obs(
    *,
    ee_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
    goal_pos: tuple[float, float, float],
    gripper_closed: bool,
    attached: bool = False,
    target_hand_contact: bool = False,
) -> dict:
    return {
        "ee_pos": np.array(ee_pos, dtype=float),
        "target_pos": np.array(target_pos, dtype=float),
        "goal_pos": np.array(goal_pos, dtype=float),
        "gripper_state": {"closed": gripper_closed, "attached": attached, "width": 0.01 if gripper_closed else 0.08},
        "contacts_summary": {"target_hand_contact": target_hand_contact},
    }


def test_segmentation_rule_snapshot() -> None:
    episode = [
        _obs(
            ee_pos=(0.2, 0.0, 0.2),
            target_pos=(0.5, 0.0, 0.03),
            goal_pos=(0.7, 0.0, 0.03),
            gripper_closed=False,
        ),
        _obs(
            ee_pos=(0.47, 0.0, 0.05),
            target_pos=(0.5, 0.0, 0.03),
            goal_pos=(0.7, 0.0, 0.03),
            gripper_closed=False,
        ),
        _obs(
            ee_pos=(0.5, 0.0, 0.05),
            target_pos=(0.5, 0.0, 0.03),
            goal_pos=(0.7, 0.0, 0.03),
            gripper_closed=True,
            attached=True,
            target_hand_contact=True,
        ),
        _obs(
            ee_pos=(0.5, 0.0, 0.15),
            target_pos=(0.5, 0.0, 0.08),
            goal_pos=(0.7, 0.0, 0.03),
            gripper_closed=True,
            attached=True,
        ),
        _obs(
            ee_pos=(0.58, 0.0, 0.15),
            target_pos=(0.58, 0.0, 0.08),
            goal_pos=(0.7, 0.0, 0.03),
            gripper_closed=True,
            attached=True,
        ),
        _obs(
            ee_pos=(0.67, 0.0, 0.14),
            target_pos=(0.67, 0.0, 0.06),
            goal_pos=(0.7, 0.0, 0.03),
            gripper_closed=False,
        ),
    ]

    segments = segment_episode(episode)
    summary = [(s["primitive"], s["start"], s["end"]) for s in segments]
    assert summary == [
        ("approach", 0, 0),
        ("lower", 1, 1),
        ("close_gripper", 2, 2),
        ("lift", 3, 3),
        ("move_to_goal", 4, 4),
        ("open_gripper", 5, 5),
    ]

