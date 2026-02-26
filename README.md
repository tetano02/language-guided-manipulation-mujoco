# MujocoLLM (Step 1)

Repository for "Language-Guided Task Abstraction for Robotic Manipulation in MuJoCo" with Franka Panda.

Current status:
- Part 1 complete: environment core, EE-first controller, debug/logging hooks, rule-based segmentation, LLM task-graph generation with JSON validation + retry + offline fallback, and initial reproducible tests.
- Part 2 in progress: scripted demo collection is now available; execution engine from task graph and full evaluation pipeline are still pending.

## Requirements

- Python 3.10+
- `uv` already configured
- Franka Menagerie assets already present in `mujoco_env/assets/franka_emika_panda/`

## Install dependencies

```bash
uv sync
```

## Run smoke environment

Headless:

```bash
uv run mujoco-env-smoke --headless --steps 50 --seed 0 --output-dir artifacts/smoke
```

Render (interactive inspection):

```bash
uv run mujoco-env-smoke --render --steps 200 --seed 0 --output-dir artifacts/render
```

## Collect scripted demos

```bash
uv run collect-demos --headless --episodes 5 --steps 220 --seed 0 --output-dir artifacts/demos --dataset-path demos/dataset.pkl
```

Optional video output:

```bash
uv run collect-demos --render --episodes 1 --steps 220 --seed 0 --save-mp4 --output-dir artifacts/demos --dataset-path demos/dataset.pkl
```

## Controller details

The project uses an `EE-first` controller:
- Command format is Cartesian EE target + gripper flag: `[ee_x, ee_y, ee_z, gripper_closed]`.
- The controller computes arm joint targets (`joint1..joint7`) with IK and writes them to MuJoCo position actuators.

IK is solved with damped least squares (DLS):

```text
dq = J^T (J J^T + lambda^2 I)^(-1) e
```

Where:
- `J` is the site Jacobian of the gripper (`mj_jacSite`).
- `e` is Cartesian position error (orientation term optional and currently disabled by default).
- `lambda` is damping for numerical stability near singularities.

Stability safeguards:
- Joint delta clipping per step (`max_delta_q`).
- Joint-limit clipping.
- EE target rate limiting in `env.step` to avoid impulsive commands.

## Grasp model (`weld`)

Contacts alone can be unstable in early development, so grasp is simplified with a runtime MuJoCo `equality/weld`:
- `close` + finger contact with target + near-distance condition -> weld attaches target to hand.
- `open` -> weld detaches.
- Failsafe detach if weld is active but EE-target distance diverges.

Important implementation detail:
- Before enabling weld, relative pose is written to weld `eq_data` (rel position + rel quaternion) to prevent teleport/impulsive jumps.

## Debug observations and fixes

Observed issue during visual debug:
- In some runs, the robot appeared to go "under the floor" and simulation became unstable.

Root cause:
- The problem was not only target placement/workspace.
- Main cause was collision filtering in the MJX Panda model: several collision geoms had no effective interaction with the floor.

Applied fix:
- Updated `mujoco_env/assets/franka_emika_panda/mjx_panda.xml` collision defaults to use `conaffinity=1` (while keeping `contype=0` to avoid aggressive self-collisions), so links correctly collide with floor/objects.
- Tightened workspace sampling in `mujoco_env/env.py` to keep targets/goals in a more reachable region:
  - `workspace_low = [0.42, -0.24, 0.05]`
  - `workspace_high = [0.70, 0.24, 0.60]`

## Run segmentation (rule-based)

```bash
uv run segment-dataset --input demos/dataset.pkl --output artifacts/segments.json
```

## Generate task graph

Offline deterministic fallback:

```bash
uv run generate-task-graph --primitives-json artifacts/primitives.json --output artifacts/task_graph.json
```

Gemini (if `GEMINI_API_KEY` is set):

```bash
uv run generate-task-graph --primitives-json artifacts/primitives.json --output artifacts/task_graph.json --api-key-env GEMINI_API_KEY
```

Notes:
- API key is read from environment variable (default: `GEMINI_API_KEY`).
- If key is missing or generation fails, code falls back to a deterministic offline stub task graph (for testability without network/API).

## Run tests

```bash
uv run pytest
```
