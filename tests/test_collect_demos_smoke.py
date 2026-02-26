from __future__ import annotations

import pickle
import subprocess
import sys

import pytest


pytest.importorskip("mujoco")


def test_collect_demos_smoke(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.pkl"
    output_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable,
        "-m",
        "demos.collect_demos",
        "--headless",
        "--episodes",
        "1",
        "--steps",
        "40",
        "--seed",
        "0",
        "--output-dir",
        str(output_dir),
        "--dataset-path",
        str(dataset_path),
        "--no-sanity-asserts",
    ]
    subprocess.run(cmd, check=True)

    assert dataset_path.exists()
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)
    assert isinstance(dataset, list)
    assert len(dataset) == 1
    assert len(dataset[0]) > 0
    assert "target_pos" in dataset[0][0]
