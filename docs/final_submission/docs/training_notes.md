# Training Notes — K1 MotrixLab Legged Locomotion

This document describes the locomotion policy submitted with this package:

`gait/k1_walk_model_3600_motrixlab.pt`

The submitted model is a low-level gait policy. It is responsible only for converting robot proprioception and velocity commands into leg joint targets. Soccer decisions such as chasing, defending, pushing, shooting, role assignment, and field positioning are implemented in the Decider module.

## 1. Selected Gait Scheme

This submission uses the MotrixLab K1 legged locomotion scheme:

| Item | Value |
|---|---|
| Robot | Booster K1 |
| Policy flavor | `motrixlab` |
| Observation shape | `N x 47` |
| Action shape | `N x 12` |
| Control frequency | 50 Hz |
| Simulation timestep | 0.002 s |
| Control decimation | 10 |
| Export format | PyTorch TorchScript `.pt` |

The competition documents include a separate Scheme A / AMP-style interface with `N x 375 -> N x 22`. That is not the scheme used by this package. This package follows the documented Scheme B style boundary: the exported actor keeps the official `47 -> 12` MotrixLab legged interface and does not add extra observation dimensions.

## 2. Observation and Action Design

The 47-dimensional observation contains:

| Range | Component |
|---|---|
| 0:3 | Base linear velocity |
| 3:6 | Base angular velocity |
| 6:9 | Projected gravity |
| 9:12 | Body-frame command velocity `[vx, vy, yaw_rate]` |
| 12:24 | 12 leg joint positions |
| 24:36 | 12 leg joint velocities |
| 36:47 | Previous action history |

The 12-dimensional action controls only the leg joints as position offsets relative to the default pose. The action is scaled by joint-specific action scales and then applied through PD control. Upper-body joints are not driven by this policy.

## 3. Reward Design

The locomotion policy was selected for stable commanded walking rather than task-specific soccer behavior. The reward family follows a standard humanoid/legged locomotion design:

| Reward Term | Purpose |
|---|---|
| Linear velocity tracking | Match commanded forward and lateral velocity. |
| Angular velocity tracking | Match commanded yaw-rate for turning. |
| Upright orientation | Penalize roll/pitch deviation and encourage a stable torso. |
| Base height stability | Keep the body near the expected walking height. |
| Joint regularization | Penalize excessive joint position and velocity deviations. |
| Action smoothness | Penalize rapid changes in consecutive actions. |
| Energy / torque regularization | Discourage high-effort motions that destabilize contact. |
| Foot contact rhythm | Encourage alternating support and swing phases. |
| Foot slip penalty | Reduce lateral sliding during stance. |
| Collision / termination penalty | Penalize falls, unsafe contacts, or early episode termination. |

The reward intentionally does not include ball position, goal position, score, opponent behavior, passing, shooting, or any game-state features. Those belong to the high-level Decider and are not embedded in the gait model.

## 4. Training Hyperparameters

The exact cloud training run metadata is not packaged with the submission, so the table below records the effective assumptions and runtime interface used by the exported policy.

| Category | Value |
|---|---|
| Algorithm family | PPO-style Actor-Critic locomotion training |
| Actor export | Actor-only inference graph |
| Critic at submission | Removed / not used by runtime inference |
| Control rate | 50 Hz |
| Physics timestep | 0.002 s |
| Control decimation | 10 |
| Command input | Body-frame normalized velocity command |
| Command scale in observation | `[2.0, 2.0, 0.25]` |
| DOF position scale | `1.0` |
| DOF velocity scale | `0.05` |
| Gyro scale | `0.25` |
| Gait frequency | Approximately 1.5 Hz |
| Inference device | CPU-compatible, GPU optional |

Runtime command limits are configured in `decider/config.yaml`:

| Parameter | Value |
|---|---|
| `walk_vel_x` | 0.5 |
| `walk_vel_y` | 0.5 |
| `walk_vel_theta` | 1.0 |
| `max_walk_vel_x` | 1.5 |
| `max_walk_vel_y` | 2.0 |
| `max_walk_vel_theta` | 3.0 |
| `continuous_push.vx_max_approach` | 1.0 |
| `continuous_push.vx_max_push` | 0.9 |
| `continuous_push.vy_max` | 0.75 |
| `continuous_push.w_max` | 1.5 |

The submitted Decider clips outgoing velocity commands to the expected normalized command range before sending them to the simulator.

## 5. Sim-to-Game Integration

The runtime stack is intentionally separated:

1. Decider reads the game state and computes a body-frame velocity command.
2. The simulator converts the command into the 47-dimensional policy observation.
3. The gait policy outputs 12 leg action offsets.
4. The simulator applies PD gains, torque limits, and joint limits.

This separation preserves the competition boundary: the gait model controls locomotion only, while soccer intelligence lives in readable Python code under `decider/`.

## 6. Robustness Notes

The final Decider configuration favors stable pushing and recovery over maximum speed:

- Continuous push uses smooth proportional control rather than abrupt state transitions.
- Sideline repulsion keeps the robot away from out-of-bounds regions.
- Velocity saturation is soft around the configured threshold.
- Field size is set to league `M` by default.
- `server_ip` is sanitized to `127.0.0.1` for submission safety.

For strict pre-submission verification, run the submitted model in the official environment with `--k1-policy-flavor motrixlab` and confirm at least one uninterrupted long run without fall-induced process failure.
