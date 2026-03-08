# nmpc_demo

Standalone pure-Python xArm6 NMPC dynamic obstacle avoidance demo.

## Overview

This repository packages a self-contained MuJoCo demo for dynamic obstacle avoidance with an xArm6 model. The demo keeps the arm near a fixed joint configuration when the obstacle is far away and pushes the arm away when a movable obstacle enters the protected workspace around the robot links.

The project is intended as a compact example and testbed. It does not require `xarm6_mpc` or `improved_rrt_robot` at runtime.

## Features

- Fixed `start_q` acts as both the nominal pose and return target.
- Dynamic obstacle avoidance is driven by a draggable MuJoCo mocap capsule.
- Distance checks are computed between the obstacle capsule and robot capsules on `link2` through `link6`.
- Rendered and headless execution paths are both supported.
- The project includes focused unit and smoke tests.

## Repository Layout

- `run_demo.py`: command-line entrypoint for render and headless execution.
- `nmpc_demo/`: controller, geometry, constants, rendering, and kinematics modules.
- `config/xarm6_capsules.yaml`: capsule geometry used for robot-obstacle distance checks.
- `assets/`: MuJoCo mesh assets for the xArm6 and gripper.
- `scene.xml`: MuJoCo scene definition with the mocap obstacle.
- `xarm6.xml`: standalone robot model included by the scene.
- `tests/`: unit tests and smoke tests for the demo.

## Requirements

- Python 3.10 or newer is recommended.
- MuJoCo and the Python dependencies listed in `requirements.txt` must be installed.
- A graphical desktop environment is required for `--backend render`.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Demo

Start the interactive renderer with:

```bash
python run_demo.py --backend render
```

When the viewer opens, drag the orange obstacle capsule close to the arm. The controller should move the robot away from the obstacle and then bring it back toward `start_q` when the obstacle is moved away.

### Moving the obstacle in MuJoCo

1. Left-double-click the orange capsule to select it.
2. Hold `Ctrl` while dragging the mouse to move the selected capsule.
3. Dragging without `Ctrl` moves the camera instead of the obstacle.
4. Press `F1` in the viewer to see MuJoCo's built-in mouse help.

### Running without a time limit

```bash
python run_demo.py --backend render --sim-time 0
```

`--sim-time 0` keeps the demo running until the viewer is closed.

## Headless Mode

For a short non-visual smoke run:

```bash
python run_demo.py --backend headless --sim-time 0.2
```

This is useful for quick checks in environments where a viewer is unavailable.

## Tests

Run the full test suite with:

```bash
pytest tests -v
```

The suite includes controller, geometry, scene, kinematics, and headless smoke coverage.

## Known Limitations

- The demo uses a fixed nominal joint configuration instead of goal-tracking between arbitrary poses.
- The obstacle model is limited to a single draggable capsule in the provided MuJoCo scene.
- The project is designed as a standalone demo and validation target, not a full production control stack.
