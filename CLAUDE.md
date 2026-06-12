# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sim-soccer** is a bipedal/multi-legged robot soccer simulation and decision control framework. It wraps physics simulation (MotrixSim), process management, and ZMQ networking so that users write only high-level soccer strategy in Python state machines.

The project splits into two independent modules that communicate via ZMQ:

- **Decider** (`decider/`) — the robot "brain". Runs state machines, connects to simulation as a client.
- **Simulation** (`simulation/`) — physics simulation engines and the Sim Manager web dashboard.

## Environment Setup (Conda)

Three separate Conda environments are required:

| Environment | Python | Purpose | Dependencies |
|---|---|---|---|
| `motrixsim0508` | 3.12 | Simulation (MotrixSim) + Sim Manager | `simulation/motrixsim/requirements.txt` |
| `k2` | 3.8 | Decider decision logic | `decider/requirements.txt` |
| `sim_soccer_rl` | 3.10 | K1 RL training (MotrixLab) | UV-managed (`MotrixLab/`) |

> **Note:** The MotrixLab sub-project uses UV for its own dependency management (`uv sync`/`uv run`). The `train_k1.sh` wrapper invokes it via `conda run -n sim_soccer_rl`. See [MotrixLab/CLAUDE.md](MotrixLab/CLAUDE.md) for full RL training details.

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
# Using the convenience script:
./scripts/start_sim.sh                          # 1v1, real-time
./scripts/start_sim.sh --team-size 3             # 3v3
./scripts/start_sim.sh --policy path/to/model.pt # custom policy
./scripts/start_sim.sh --record-video /tmp/frames # save top-down frames for video

# Or directly:
PYTHONPATH=simulation/motrixsim:$PYTHONPATH uv run --directory MotrixLab python -u -m app.runner --team-size 1 --real-time
```
**Note:** The real entry point is `simulation/motrixsim/app/runner.py` (imports `parse_runtime_args` + `run_sim`). `sim2sim_runner.py` is a standalone alternative that uses the sim2simlib config path (legacy).

Key CLI flags for `runner.py` / `sim2sim_runner.py`:
- `--robot-type k1|pi_plus` (default: k1)
- `--team-size N` — robots per team (0-7)
- `--real-time` / `--no-real-time` — real-time pace vs as-fast-as-possible
- `--policy PATH` — override policy file (.pt/.onnx)
- `--k1-policy-flavor motrixlab|legged_gym` — policy compatibility mode (default: `motrixlab`)
- `--k1-legged-gym` / `--no-k1-legged-gym` — toggle 47→12 legged locomotion policy
- `--policy-device cpu|gpu` — inference device (default: gpu)
- `--webview` / `--no-webview` — enable/disable WebView streaming
- `--record-video DIR` — save top-down frames (PNG) every 1/30s to DIR (auto-disables WebView)

### Start a Single Decider
```bash
# Using the convenience script:
./scripts/start_decider.sh                              # red, id=0
./scripts/start_decider.sh --color blue --id 0 --port 5556

# Or directly:
PYTHONPATH=decider:$PYTHONPATH uv run --directory MotrixLab python -u decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0
```

### Start a Full Team
```bash
./decider/scripts/start_team.sh           # auto-detect counts
./decider/scripts/start_team.sh --red 3 --blue 2
./decider/scripts/start_team.sh --kill    # stop all
```

### One-Click Video Recording
```bash
./scripts/record_match.sh                        # 60s, 1v1, auto-named MP4
./scripts/record_match.sh --d 30 --output ~/Videos/test.mp4
./scripts/record_match.sh --demo-3v3 --d 120     # 3v3 demo with 3 red deciders
./scripts/record_match.sh --trajectory            # also record trajectory CSV
```
Launches simulation + decider(s), records top-down frames at 30fps, encodes to MP4 with ffmpeg on exit. `--demo-3v3` spawns 3 red deciders (ids 0/1/2) running attacker/support/defender roles. See `scripts/match_config.sh` for all options (policy override, fixed-command debug, etc.).

### Trajectory Recording (Decider-side diagnostics)
```bash
# Enable trajectory recording when running decider directly:
PYTHONPATH=decider:$PYTHONPATH uv run --directory MotrixLab python -u decider/decider.py \
    --simulation --color red --id 0 --record-trajectory

# Custom output directory:
... --trajectory-dir /path/to/output

# Analyze after recording:
python decider/scripts/analyze_trajectory.py /path/to/output/trajectory.csv

# Full diagnosis (why push failed):
python decider/scripts/diagnose_trajectory.py /path/to/output/trajectory.csv
```
Trajectory CSVs record per-frame robot pose, ball position, commands, FSM state, and game state. `diagnose_trajectory.py` produces a structured `diagnosis.json` and Chinese-text `diagnosis.txt` with a three-layer analysis: state transitions → condition checks → push execution.

### Run Tests
```bash
# Trajectory diagnosis tests (Python 3.8+):
python -m pytest tests/test_diagnose_trajectory.py -v
# or:
python -m unittest tests.test_diagnose_trajectory -v

# Specific test:
python -m unittest tests.test_diagnose_trajectory.StateLayerTests.test_chase_only_never_05m
```
Tests live under `tests/`; fixtures are in `tests/fixtures/`. There is no test framework for the broader decider module.

### Cleanup Residual Processes
```bash
sudo pkill -f decider.py
fuser -k 5555/tcp 2>/dev/null || true
```

### RL Training (MotrixLab sub-project)
```bash
cd MotrixLab
uv sync --all-packages --all-extras
uv run scripts/train.py --env cartpole
uv run scripts/view.py --env cartpole          # visualize
uv run pytest                                  # run tests
```

Available K1 environment names: `k1-flat-terrain-walk`, `k1-point-navigation`, `k1-ball-navigation`, `k1-amp-walk`, `k1-amp-walk-small`, `k1-amp-walk-lift`.

**Important `train.py` flags:**
- `--rllib skrl|rslrl` — RL framework (default: `skrl`). RSLRL is required for K1 walk training.
- `--num-envs N` — vectorized environments (default: 2048; K1 walk uses 4096)
- `--resume-policy PATH` — resume training from a checkpoint (RSLRL only)
- `--resume-noise-std FLOAT` — reset exploration noise when resuming (RSLRL only)
- `--max-iterations N` — override the config's iteration count
- `--train-backend jax|torch` — backend override (auto-detected by default)

### K1 Walk Training (via shell wrapper)
```bash
cd MotrixLab
bash scripts/train_k1.sh                          # RSLRL, 4096 envs, seed 1
bash scripts/train_k1.sh --resume-policy PATH     # resume from checkpoint
```
The wrapper sets `CUDA_VISIBLE_DEVICES=0`, syncs RSLRL extras, and runs `train.py --env k1-flat-terrain-walk --rllib rslrl --num-envs 4096 --seed 1`.

### Playing / Evaluating Policies
```bash
uv run scripts/play.py --env cartpole
uv run scripts/play.py --env cartpole --policy path/to/model_3600.pt --rllib rslrl
```
With no `--policy` or `--rllib`, `play.py` **auto-discovers** the most recent training run under `runs/{env}/` and picks its best checkpoint.

### K1 Locomotion Smoke Test
```bash
cd MotrixLab
conda run -n sim_soccer_rl env PYTHONPATH=./motrix_envs/src:./motrix_rl/src python scripts/smoke_k1_env.py --num-envs 4 --steps 16 --zero-action
```

## Architecture

### Decider (`decider/`)
- **Entry point**: `decider/decider.py` — defines `Agent` (ROS2 mode) and `SimAgent` (ZMQ/simulation mode). Run with `--simulation` flag.
- **User strategy**: `decider/user_entry.py` — implement `init(agent)` and `loop(agent)`. Decider calls `user_entry.loop(self)` at each tick.
- **Config**: `decider/config.yaml` — velocities, thresholds, state machine parameters, league size (S/M/L), networking.
- **Configuration loading** (`decider/configuration.py`): loads `config.yaml` then deep-merges any `config_override.json/yaml/yml` on top. Override files support JSON-with-comments (`//` and `/* */`). All three formats (JSON, YAML, YML) are tried in order; the first existing file wins.
- **Interfaces**: `interfaces/action.py` (cmd_vel, kick), `interfaces/vision.py` (ball/self position, objects), `interfaces/gamecontroller.py`, `interfaces/sim_client.py` (ZMQ client).
- **Trajectory recording** (`decider/trajectory.py`): `TrajectoryRecorder` streams per-frame diagnostics to CSV. Fields include robot/ball pose, commands, FSM state, alignment mode, and kick readiness. Instantiated by `SimAgent` when `--record-trajectory` or `--trajectory-dir` is passed.
- **State Machines (three-layer hierarchy)**:
  - `logic/sub_statemachines/` — Basic actions: `find_ball`, `chase_ball`, `dribble`, `kick`, `go_back_to_field`
  - `logic/strategy_statemachines/` — Tactical: `attack`, `defend_ball`, `dribble_ball`, `shoot_ball`
  - `logic/policy_statemachines/` — Role: `goalkeeper`
- **Team launch**: `scripts/start_team.sh` — uses `screen` sessions per robot.

### Decider Strategies (in `user_entry.py`)

The active `game()` function routes based on `agent.is_simulation`:

- **Simulation mode**: role-based dispatch by `agent.id`:
  - `id=0` → **Attacker**: `find_ball` → `ContinuousPushController` (continuous chase/behind-ball/push-to-goal error control)
  - `id=1` → **Support**: position behind the ball on the ball-to-own-goal line (~1.2m)
  - `id=2` → **Defender**: anchor at X=-2m in own half, track ball Y laterally
  - Other ids fall back to attacker role
- **ROS2 mode**: calls `_gc_test_go_back_to_field` (GameController state machine test)

Other strategy controllers available in the file:
- `PushToGoalController` — simple P-control: target position 12cm behind ball, face goal, kick when near
- `ContinuousPushController` — primary simulation attacker: continuous error-space control for chase, ball-line alignment, sideline repulsion, and push-to-goal
- `AdvancedDribbler` — vector-field dribbling with goal attraction, boundary repulsion, turn-to-ball fallback, and alignment-adaptive speed
- `_navigate_to_pose()` — reusable P-control navigation to world-coordinate targets

### Simulation (`simulation/`)
- **motrixsim/** — Primary engine:
  - `app/runner.py` — **real CLI entry point**; imports `parse_runtime_args` + `run_sim`
  - `app/runtime_config.py` — all K1/Pi+ robot constants (joint orders, PD gains, action scales, obs configs), `RobotRuntimeConfig` dataclass, `build_robot_runtime_config()`, and `parse_runtime_args()` — the single source of truth for robot simulation configuration
  - `app/multi_robot_sim.py` — main simulation loop, policy inference, ZMQ communication, K1 policy mixing/AMP logic, top-down frame rendering for `--record-video`
  - `sim2sim_runner.py` — standalone alternative CLI using sim2simlib config path (legacy)
  - `soccer_env.py` — Gymnasium-style env wrapper (`MotrixSoccerSim`)
- **labbridge/** — Sim Manager (FastAPI web app) and WebView server:
  - `sim_manager.py` / `sim_manager2.py` — web dashboard for managing sim processes
  - `webview_server.py` — WebView streaming server

### RL Training (`MotrixLab/`)
- Standalone RL training framework (UV workspace with `motrix_envs` and `motrix_rl` packages).
- K1 locomotion environment lives under `MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/`.
- Has its own [MotrixLab/CLAUDE.md](MotrixLab/CLAUDE.md) with full RL training documentation (RSLRL, SKRL, RSLRL config constraints, etc.).

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
- **No test framework** for the decider module (except `tests/test_diagnose_trajectory.py`). The `transitions` library (v0.9.2) is used for implementing hierarchical state machines.
- ROS2 is optional; all simulation work uses `--simulation` flag (no ROS dependencies).
- Decider uses Python 3.8 (old). Don't introduce syntax/features incompatible with 3.8.
- The `legged_gym/` directory is a detached copy of another project — treat as reference only. **Do not move or delete it** — training configs hardcode paths like `legged_gym/resources/robots/K1/k1_train_scene.xml` and `legged_gym/policy/booster_k1/model_4700.onnx`.
- Root-level model files (`model_20000_new.onnx`, `model_4700.pt`) are default policy paths referenced by runtime configs — do not remove.
- `config.yaml` is the single source of truth for tunable parameters; avoid hardcoding magic numbers in strategy code. Optional `config_override.json/yaml/yml` files (in `decider/`) are deep-merged on top at load time via `decider/configuration.py`. Override files support JSON-with-comments (`//` and `/* */`).
- Field size presets: S = 9×6m, M = 14×9m, L = 22×14m (configured via `config.yaml` league field).

### Branch State (inspect/simple-push-20260607)
This branch contains uncommitted changes:
- `decider/decider.py` — trajectory recording integrated into `SimAgent`
- `decider/user_entry.py` — `ContinuousPushController`, `PushToGoalController`, `_support_role`, `_defender_role`, role-based `game()`
- `decider/trajectory.py` — new module (untracked)
- `decider/scripts/analyze_trajectory.py` — trajectory visualization (untracked)
- `decider/scripts/diagnose_trajectory.py` — automated failure diagnosis (untracked)
- `scripts/match_config.sh` + `scripts/record_match.sh` — one-click recording (untracked)
- `tests/` — unit tests for trajectory diagnosis (untracked)
- `video/` — recorded match MP4s and trajectory outputs (untracked)

### K1 Policy Flavor System
`simulation/motrixsim/app/runtime_config.py` defines two K1 policy compatibility modes for 47→12 legged locomotion policies:
- **`motrixlab`** (default) — policies trained by MotrixLab/motrix_envs `k1-flat-terrain-walk`. Uses MotrixLab-specific observation scales, action scales, PD gains, and torque limits (see `K1_MOTRIXLAB_*` constants).
- **`legged_gym`** — legacy policies trained by legged_gym (T1/T1_config.py). Uses different scaling/PD constants (see `K1_LEGGED_GYM_*` constants).

Select with `--k1-policy-flavor motrixlab|legged_gym`. The default walk policy is `simulation/motrixsim/assets/policies/k1_walk_model_3600_motrixlab.pt`.

### K1 Policy Mixing (runtime_config.py)
When the leg-control policy (47→12) is active and a full-body stand policy (`k1_model_46000.pt`, 78→22) exists, the runner auto-mixes: small velocity commands use the 78→22 full-body policy (standing/upper-body posture), while larger commands switch to the 47→12 leg-control policy (with hysteresis to avoid oscillation). Disable with `--no-k1-legged-gym`.

### MotrixLab Source Path Pattern
All MotrixLab scripts use `from _source_path import ensure_source_path; ensure_source_path()` to add `motrix_envs/src` and `motrix_rl/src` to `sys.path` before any other imports. This pattern is required whenever running scripts from within the MotrixLab directory.

### Multi-Agent Strategy (WIP)
`decider/strategy/team_manager.py` implements a multi-agent coordination system (role assignment, world model fusion) — intended architecture for team play, currently work-in-progress. The simpler role-based dispatch in `user_entry.py:game()` is the currently active approach for 3v3 simulations.

### Docker (Isaac Sim, legacy)
`Dockerfile` and `compose.yaml` target the NVIDIA Isaac Sim image — this is a legacy/alternate deployment path, not the primary MotrixSim workflow.
