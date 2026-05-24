# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sim-soccer** is a bipedal/multi-legged robot soccer simulation and decision control framework. It wraps physics simulation (MotrixSim), process management, and ZMQ networking so that users write only high-level soccer strategy in Python state machines.

The project splits into two independent modules that communicate via ZMQ:

- **Decider** (`decider/`) — the robot "brain". Runs state machines, connects to simulation as a client.
- **Simulation** (`simulation/`) — physics simulation engines and the Sim Manager web dashboard.

## Environment Setup (Conda)

Two separate Conda environments are required:

| Environment | Python | Purpose | Dependencies |
|---|---|---|---|
| `motrixsim0508` | 3.12 | Simulation (MotrixSim) + Sim Manager | `simulation/motrixsim/requirements.txt` |
| `k2` | 3.8 | Decider decision logic | `decider/requirements.txt` |

```bash
# Simulation env
conda create -n motrixsim0508 python=3.12 -y
conda activate motrixsim0508
pip install -r simulation/motrixsim/requirements.txt

# Decider env
conda create -n k2 python=3.8 -y
conda activate k2
pip install -r decider/requirements.txt
```

## Key Commands

### Start Sim Manager (web dashboard)
```bash
conda run -n motrixsim0508 python simulation/motrixsim/sim_manager.py --host 0.0.0.0 --port 8000
```
Then open `http://127.0.0.1:8000/` to launch simulations.

### Start Simulation via CLI (no web manager)
```bash
conda run -n motrixsim0508 python simulation/motrixsim/sim2sim_runner.py --team-size 3 --real-time
```

### Start a Single Decider
```bash
conda activate k2
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0
```

### Start a Full Team
```bash
./decider/scripts/start_team.sh           # auto-detect counts
./decider/scripts/start_team.sh --red 3 --blue 2
./decider/scripts/start_team.sh --kill    # stop all
```

### Cleanup Residual Processes
```bash
sudo pkill -f decider.py
```

### RL Training (MotrixLab sub-project)
```bash
cd MotrixLab
uv sync --all-packages --all-extras
uv run scripts/train.py --env cartpole
uv run scripts/view.py --env cartpole          # visualize
uv run pytest                                  # run tests
```

## Architecture

### Decider (`decider/`)
- **Entry point**: `decider/decider.py` — defines `Agent` (ROS2 mode) and `SimAgent` (ZMQ/simulation mode). Run with `--simulation` flag.
- **User strategy**: `decider/user_entry.py` — implement `init(agent)` and `loop(agent)`. Decider calls `user_entry.loop(self)` at each tick.
- **Config**: `decider/config.yaml` — velocities, thresholds, state machine parameters, league size (S/M/L), networking.
- **Interfaces**: `interfaces/action.py` (cmd_vel, kick), `interfaces/vision.py` (ball/self position, objects), `interfaces/gamecontroller.py`, `interfaces/sim_client.py` (ZMQ client).
- **State Machines (three-layer hierarchy)**:
  - `logic/sub_statemachines/` — Basic actions: `find_ball`, `chase_ball`, `dribble`, `kick`, `go_back_to_field`
  - `logic/strategy_statemachines/` — Tactical: `attack`, `defend_ball`, `dribble_ball`, `shoot_ball`
  - `logic/policy_statemachines/` — Role: `goalkeeper`
- **Team launch**: `scripts/start_team.sh` — uses `screen` sessions per robot.

### Simulation (`simulation/`)
- **motrixsim/** — Primary engine. Key files:
  - `sim2sim_runner.py` — CLI binary that starts a K1/Pi+ soccer sim instance
  - `soccer_env.py` — Gymnasium-style env wrapper (`MotrixSoccerSim`)
- **labbridge/** — Sim Manager (FastAPI web app) and WebView server:
  - `sim_manager.py` / `sim_manager2.py` — web dashboard for managing sim processes
  - `webview_server.py` — WebView streaming server

### RL Training (`MotrixLab/`)
- Standalone RL training framework (UV workspace with `motrix_envs` and `motrix_rl` packages).
- K1 locomotion environment lives under `MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/`.

### Communication
- ZMQ REQ/REQ pattern over TCP. Each robot uses one port.
- **Request**: `{"cmd": [vx, vy, w], "id": N, "timestamp": T}`
- **Response**: `{"state": {"robots": [...], "ball": {...}, "gamecontroller": {...}}, "sim_timestamp": ..., "step_latency": ...}`

### Coordinate System
- **Robot frame**: X = forward, Y = left, Theta = CCW (radians), 0 = straight ahead
- **Map frame**: Y = toward opponent goal, X = right side
- Blue team coordinates are mirrored so both teams reuse the same strategy code.
- Ball angle: `atan2(ball_y, ball_x)` — positive = left of robot.

### Robot ID Mapping (fixed regardless of team size)
- `0..6` → `robot_rp0..robot_rp6` (red team)
- `7..13` → `robot_bp0..robot_bp6` (blue team)

## Key API Patterns

### Perception (agent -> robot)
```python
agent.get_ball_pos()          # [x, y] in robot frame or [None, None]
agent.get_ball_distance()     # float
agent.get_ball_angle()        # radians, atan2(y, x)
agent.get_if_ball()           # bool
agent.get_self_pos()          # [x, y] in map frame
agent.get_self_yaw()          # float in map frame
agent.get_ball_pos_in_map()   # [x, y] in map frame
```

### Action (agent -> robot commands)
```python
agent.cmd_vel(vx, vy, vtheta)  # robot-relative velocity (scaled by config internally)
agent.stop()
agent.kick(foot=0, death=0)
agent.save_ball(direction=1)
agent.move_head(pitch, yaw)
```

### State Machine Invocation
```python
agent.state_machine_runners['find_ball']()
agent.state_machine_runners['chase_ball']()
agent.state_machine_runners['goalkeeper']()
```

## Important Notes
- **No test framework** for the decider module. The `transitions` library (v0.9.2) is used for implementing hierarchical state machines.
- ROS2 is optional; all simulation work uses `--simulation` flag (no ROS dependencies).
- Decider uses Python 3.8 (old). Don't introduce syntax/features incompatible with 3.8.
- The `legged_gym/` directory is a detached copy of another project — treat as reference only.
- `config.yaml` is the single source of truth for tunable parameters; avoid hardcoding magic numbers in strategy code.
- Field size presets: S = 9x6m, M = 14x9m, L = 22x14m (configured via `config.yaml` league field).
