# MotrixArena S2 3v3 Robot Soccer _(motrixarena-competition-S2)_

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Decision](https://img.shields.io/badge/Decider-State%20Machine-FF6F00)
![Communication](https://img.shields.io/badge/Communication-ZMQ-555555)
![Robot](https://img.shields.io/badge/Robot-K1%20%7C%20Pi%20Plus-4B8BBE)

3v3 robot soccer simulation strategy, replay, and diagnostics for MotrixArena S2.

This repository demonstrates a K1 / Pi Plus soccer simulation workflow built around MotrixSim, a Decider state-machine strategy stack, and reproducible match analysis tools. It includes a local 3v3 replay, trajectory time-series diagnostics, and locomotion tracking assets for reviewing the competition run.

[中文](README.zh-CN.md)

## Effect Showcase

![3v3 robot soccer replay 1](docs/assets/demo/football-1.gif)

![3v3 robot soccer replay 2](docs/assets/demo/football-2.gif)

![Locomotion velocity tracking](docs/assets/demo/loco-v030-velocity-tracking.png)

## Background

MotrixArena S2 focuses on simulated robot soccer. Multiple robots need to find the ball, approach from a useful angle, push toward the opponent goal, and hold support or defensive positions without interfering with teammates.

This project keeps the high-level soccer behavior in the Decider layer, with the main strategy entry at [`decider/user_entry.py`](decider/user_entry.py) and state-machine logic in [`decider/logic/`](decider/logic/). Recording and analysis scripts make each run reproducible and inspectable at frame level.

## Install

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/Logic-TARS/motrixarena-competition-S2.git
cd motrixarena-competition-S2
python3 -m pip install --user uv
pip install -r requirements.txt
```

For container-based setup, see [`compose.yaml`](compose.yaml). The default K1 policy model used by the examples is [`models/k1/model_4700.pt`](models/k1/model_4700.pt).

## Usage

Start a simulation:

```bash
# 1v1 real-time simulation
./scripts/start_sim.sh

# 3v3 real-time simulation
./scripts/start_sim.sh --team-size 3

# Use a specific policy
./scripts/start_sim.sh --policy models/k1/model_4700.pt
```

Start one Decider process:

```bash
./scripts/start_decider.sh
./scripts/start_decider.sh --color blue --id 0 --port 5556
```

Start or stop a team:

```bash
./decider/scripts/start_team.sh
./decider/scripts/start_team.sh --red 3 --blue 2
./decider/scripts/start_team.sh --kill
```

Record a match:

```bash
# Record a 1v1 match, default duration 60 seconds
./scripts/record_match.sh

# Record a 3v3 demo for 120 seconds
./scripts/record_match.sh --demo-3v3 --d 120

# Record video and trajectory CSV
./scripts/record_match.sh --trajectory
```

## Results

| Item | Value |
| --- | --- |
| Competition | MotrixArena S2 robot soccer simulation |
| Award | Excellent award |
| Rank | 10th |
| Goal rate | Above 70% |
| Robot platform | K1 / Pi Plus |
| Decision framework | Decider state machines |
| Simulation platform | MotrixSim, with historical Isaac Sim compatibility |
| Main attacking strategy | `ContinuousPushController` |

<a id="demo-replay"></a>

## Demo / Replay

| Asset | Link | Description |
| --- | --- | --- |
| 3v3 match replay | [`football-1.mp4`](docs/assets/demo/football-1.mp4) | Local 3v3 simulation replay |
| Trajectory time series | [`demo-trajectory-timeseries.png`](docs/assets/demo/demo-trajectory-timeseries.png) | Robot, ball, and command values over time |
| Locomotion tracking | [`loco-v030-velocity-tracking.png`](docs/assets/demo/loco-v030-velocity-tracking.png) | Velocity-tracking diagnostic for `T1_forward_velocity/v030` |

## Architecture

```text
motrixarena-competition-S2/
├── decider/                   # Decision runtime and strategy logic
│   ├── user_entry.py          # Custom strategy entry point
│   ├── decider.py             # Decider runtime
│   ├── interfaces/            # Action, vision, game controller, and sim clients
│   ├── logic/                 # State machines and role policies
│   └── scripts/               # Team launch and diagnostics scripts
├── simulation/                # MotrixSim and historical Isaac Sim integration
├── MotrixLab/                 # K1 locomotion and policy execution subproject
├── models/k1/                 # Default K1 policy models
├── docs/assets/demo/          # Demo replay and diagnostic plots
└── scripts/                   # Simulation, decider, and recording helpers
```

## Strategy

In simulation mode, `game(agent)` assigns each robot a role by ID:

| Robot ID | Role | Behavior |
| --- | --- | --- |
| `0` | Attacker | Finds the ball and pushes it toward the opponent goal |
| `1` | Support | Positions behind the ball on the ball-to-own-goal line and avoids blocking the attacker |
| `2` | Defender | Holds an own-half defensive anchor and tracks the ball laterally |
| Other IDs | Fallback attacker | Uses the attacker behavior path |

`ContinuousPushController` turns ball chasing into a controlled push sequence. It estimates behind-ball depth, lateral offset from the ball-to-goal line, heading error, ball distance, and sideline risk, then generates smooth velocity commands for approach, alignment, and push phases.

## Diagnostics

| Tool | Purpose |
| --- | --- |
| [`scripts/record_match.sh`](scripts/record_match.sh) | Starts simulation and decider processes, records frames, and encodes a match video |
| `--record-trajectory` / `--trajectory` | Enables trajectory CSV logging for a run |
| [`decider/scripts/analyze_trajectory.py`](decider/scripts/analyze_trajectory.py) | Generates trajectory summaries and plots |
| [`decider/scripts/diagnose_trajectory.py`](decider/scripts/diagnose_trajectory.py) | Diagnoses push and alignment failure modes |

Trajectory records include robot pose, ball position, velocity commands, FSM state, alignment mode, kick conditions, and controller diagnostics.

## Repository Structure

```text
.
├── decider/                   # Decision runtime and strategy logic
├── simulation/                # Simulation integration
├── MotrixLab/                 # K1 locomotion and policy execution
├── models/k1/                 # Default K1 policy models
├── docs/assets/demo/          # Demo replay and diagnostic plots
├── scripts/                   # Launch and recording scripts
├── requirements.txt
├── compose.yaml
├── LICENSE
└── COPYRIGHT
```

## Contributing

Issues and pull requests are welcome. Documentation, reproducibility notes, and diagnostic script improvements are especially useful. Avoid committing large generated assets unless they are necessary for review or reproduction.

## License

GPL-3.0-or-later © MOS-Brain Contributors.

See [`LICENSE`](LICENSE) and [`COPYRIGHT`](COPYRIGHT) for license and attribution details.
