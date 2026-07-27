# MotrixArena S2 3v3 Robot Soccer

[中文](README.zh-CN.md)

Multi-robot decision, control, replay, and diagnostics project for the MotrixArena S2 robot soccer simulation competition. The system runs K1 / Pi Plus soccer agents through MotrixSim, a Decider state-machine stack, and ZMQ-based simulator communication.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Decision](https://img.shields.io/badge/Decider-State%20Machine-FF6F00)
![Communication](https://img.shields.io/badge/Communication-ZMQ-555555)
![Robot](https://img.shields.io/badge/Robot-K1%20%7C%20Pi%20Plus-4B8BBE)

## Demo / Replay

The repository includes local demo and diagnostic assets that show the simulation, trajectory logging, and locomotion analysis pipeline.

| Asset | Link | Purpose |
| --- | --- | --- |
| 3v3 demo replay | [demo-match-20260614-105108.mp4](docs/assets/demo/demo-match-20260614-105108.mp4) | Short local recording of the 3v3 simulation workflow |
| Trajectory time series | [demo-trajectory-timeseries.png](docs/assets/demo/demo-trajectory-timeseries.png) | Robot, ball, command, and diagnostic values over time |
| Locomotion velocity tracking | [loco-v030-velocity-tracking.png](docs/assets/demo/loco-v030-velocity-tracking.png) | Velocity-tracking diagnostic for `T1_forward_velocity/v030` |

The locomotion baseline run `20260614_135745_default` recorded 3 cases with `0 pass / 3 fail` and an acceptance rating of `Not recommended`. These failing cases are kept as engineering diagnostics for gait tracking and stability work, not as official competition results.

## Highlights

- Competition result: MotrixArena S2 excellent award, ranked 10th, with a goal rate above 70%.
- Implemented a 3v3 simulation decision policy with attacker, support, and defender roles assigned by robot ID.
- Used `decider/user_entry.py` as the main strategy entry point, with role behavior backed by `decider/logic/` state-machine components.
- Built `ContinuousPushController` to convert simple ball chasing into continuous behind-ball alignment and goal-directed pushing.
- Connected simulation and decision processes through ZMQ, with independent ports and robot IDs for multi-agent runs.
- Added a repeatable diagnostics path for match video recording, trajectory CSV capture, trajectory plots, and failure analysis.

## Results

| Item | Value |
| --- | --- |
| Competition | MotrixArena S2 robot soccer simulation |
| Result | Excellent award |
| Rank | 10th |
| Goal rate | Above 70% |
| Primary robot | K1 / Pi Plus |
| Decision framework | Decider state machines |
| Simulation platform | MotrixSim, with historical Isaac Sim compatibility |
| Communication | ZMQ with independent multi-robot ports |
| Primary attacking strategy | `ContinuousPushController` |

## Task Overview

MotrixArena S2 focuses on robot soccer in a simulated field. Multiple robots need to find the ball, approach it, align behind it, push or shoot toward the opponent goal, and hold defensive positions while avoiding teammate interference.

Key engineering challenges:

| Challenge | Approach |
| --- | --- |
| Multiple robots tend to crowd the same ball target | Assign attacker, support, and defender roles in simulation by robot ID |
| Ball chasing alone pushes the ball sideways or out of bounds | Track ball-to-goal geometry and command the attacker to stay behind the ball |
| Hard switching between chase, align, and push behavior causes jitter | Use continuous error-space control, speed limits, soft clipping, and near-ball angular damping |
| Match failures are difficult to diagnose visually | Record video, trajectory CSVs, controller state, alignment metrics, and generated plots |

## Architecture

```text
motrixarena-competition-S2/
├── decider/                   # Decision module
│   ├── user_entry.py          # Custom strategy entry point
│   ├── decider.py             # Decider runtime
│   ├── config.yaml            # Decision parameters
│   ├── interfaces/            # Action, vision, game controller, and sim clients
│   ├── logic/                 # State machines and strategy logic
│   └── scripts/               # Team launch and diagnostics scripts
├── simulation/                # Simulation module
│   ├── motrixsim/             # MotrixSim runtime
│   ├── isaac_sim/             # Historical Isaac Sim implementation
│   └── labbridge/             # WebView, bridge, and sim manager
├── MotrixLab/                 # K1 locomotion / RL subproject
├── models/k1/                 # Default K1 policy models
├── docs/                      # Documentation and demo assets
├── tools/                     # Maintenance tools
└── scripts/                   # Common launch and recording scripts
```

The Decider module owns strategy logic. The simulation module owns physics, visualization, and simulator runtime. The two sides communicate through ZMQ.

## Strategy Design

### State-Machine Layers

| Layer | Directory | Responsibility | Examples |
| --- | --- | --- | --- |
| Primitive actions | `decider/logic/sub_statemachines/` | Single robot behaviors | `find_ball`, `chase_ball`, `dribble`, `kick`, `go_back_to_field` |
| Tactical behaviors | `decider/logic/strategy_statemachines/` | Multi-action strategy composition | `attack`, `defend_ball`, `dribble_ball`, `shoot_ball` |
| Role policies | `decider/logic/policy_statemachines/` | Competition-level roles | `goalkeeper` |

### 3v3 Roles

In simulation mode, `game(agent)` in `decider/user_entry.py` assigns behavior by robot ID:

| Robot ID | Role | Behavior |
| --- | --- | --- |
| `0` | Attacker | Finds the ball, then runs `ContinuousPushController` to push toward goal |
| `1` | Support | Positions about 1.2 m behind the ball on the ball-to-own-goal line and avoids blocking the attacker |
| `2` | Defender | Holds an own-half anchor and tracks the ball laterally |
| Other IDs | Fallback attacker | Uses the attacker path |

### ContinuousPushController

`ContinuousPushController` is the primary attacker strategy. Instead of only driving toward the ball, it continuously estimates:

| Signal | Meaning | Use |
| --- | --- | --- |
| `behind_depth` | Whether the robot is behind the ball relative to the opponent goal | Controls forward/backward positioning for a valid push |
| `lateral_err` | Offset from the ball-to-goal line | Reduces diagonal or sideways pushes |
| `yaw_err` | Heading error relative to the goal direction | Aligns the robot toward goal |
| `ball_dist` | Robot-to-ball distance | Blends approach and push behavior |
| `sideline_risk` | Near-boundary risk | Adds correction toward the field center |

The controller records its active alignment mode and diagnostic values so trajectory analysis can identify whether a failure came from perception, behind-ball positioning, heading alignment, sideline correction, or locomotion.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Logic-TARS/motrixarena-competition-S2.git
cd motrixarena-competition-S2
```

### 2. Install Dependencies

The helper scripts use `uv` for Python execution. The full simulator stack also depends on the project-specific MotrixSim / MotrixLab environment and assets.

```bash
python3 -m pip install --user uv
pip install -r requirements.txt
```

For container-based setup, see [compose.yaml](compose.yaml).

### 3. Start Simulation

```bash
# 1v1 real-time simulation
./scripts/start_sim.sh

# 3v3 real-time simulation
./scripts/start_sim.sh --team-size 3

# Use a specific policy
./scripts/start_sim.sh --policy models/k1/model_4700.pt
```

### 4. Start Decider

```bash
./scripts/start_decider.sh
./scripts/start_decider.sh --color blue --id 0 --port 5556
```

### 5. Start a Team

```bash
./decider/scripts/start_team.sh
./decider/scripts/start_team.sh --red 3 --blue 2
./decider/scripts/start_team.sh --kill
```

### 6. Record a Match

```bash
# Record a 1v1 match, default duration 60 seconds
./scripts/record_match.sh

# Record a 3v3 demo for 120 seconds
./scripts/record_match.sh --demo-3v3 --d 120

# Record video and trajectory CSV
./scripts/record_match.sh --trajectory
```

## Diagnostics

The project keeps diagnostics close to the simulation loop:

| Tool | Purpose |
| --- | --- |
| `scripts/record_match.sh` | Starts simulation and decider processes, records frames, and encodes a match video |
| `--record-trajectory` / `--trajectory` | Enables trajectory CSV logging for a run |
| `decider/scripts/analyze_trajectory.py` | Generates trajectory summaries and plots |
| `decider/scripts/diagnose_trajectory.py` | Diagnoses likely push and alignment failure modes |
| `decider/scripts/analyze_loco_baseline.py` | Summarizes locomotion baseline trajectory metrics |

Trajectory rows include robot pose, ball position, command values, FSM state, alignment mode, kick conditions, and controller diagnostic values. This makes failures reproducible at frame level instead of relying only on video inspection.

## Repository Structure

```text
.
├── decider/                   # Decision runtime and strategy logic
│   ├── user_entry.py          # Main custom strategy entry
│   ├── interfaces/            # Perception, action, game controller, and sim interfaces
│   ├── logic/                 # State machines and role policies
│   └── scripts/               # Team startup and diagnostics
├── simulation/                # MotrixSim / historical Isaac Sim integration
├── MotrixLab/                 # K1 locomotion and policy execution subproject
├── models/k1/                 # Default K1 policy models
├── docs/assets/demo/          # Demo replay and diagnostic plots
├── scripts/                   # Simulation, decider, and recording helpers
├── requirements.txt
├── compose.yaml
├── LICENSE
└── COPYRIGHT
```

## License

This repository is organized as a portfolio and reproducibility record for the MotrixArena S2 competition. Upstream simulation assets, robot models, and related framework components remain the property of their respective maintainers.

See [LICENSE](LICENSE) and [COPYRIGHT](COPYRIGHT) for license and attribution details.
