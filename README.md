# nmpc_demo

Standalone pure-Python xArm6 NMPC dynamic obstacle avoidance demo.

## Features

- Fixed `start_q` is both the start and the target.
- The robot stays near `start_q` when the obstacle is far away.
- A draggable MuJoCo mocap capsule acts as the dynamic obstacle.
- Distances are computed between the obstacle capsule and robot capsules on `link2` to `link6`.
- No runtime dependency on `xarm6_mpc` or `improved_rrt_robot`.

## Install

```bash
pip install -r requirements.txt
```

## Run Render Demo

```bash
python run_demo.py --backend render
```

Open the viewer and drag the orange obstacle capsule. The arm should move away when the capsule gets close and return toward `start_q` after the obstacle is moved away.

### How To Drag The Capsule In MuJoCo

1. Left-double-click the orange capsule to select it.
2. Hold `Ctrl` and drag with the mouse to move the selected capsule.
3. Plain mouse drag without `Ctrl` moves the camera, not the obstacle.
4. Press `F1` in the viewer if you want to see MuJoCo's built-in mouse help.

### Run Without Time Limit

```bash
python run_demo.py --backend render --sim-time 0
```

`--sim-time 0` means run indefinitely until you close the viewer.

## Headless Smoke

```bash
python run_demo.py --backend headless --sim-time 0.2
```

## Tests

```bash
pytest tests -v
```
