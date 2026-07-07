# Technical Solution — MotrixArena S2 3v3 Challenge

## 1. Architecture Overview

Our Decider follows a **three-layer hierarchical state machine** architecture, separating concerns from primitive behaviors up to team-level role assignment.

```
Layer 3: Policy (role)     ───  goalkeeper, ContinuousPushController
Layer 2: Strategy (tactic) ───  attack, defend_ball, dribble_ball, shoot_ball
Layer 1: Sub (primitive)   ───  find_ball, chase_ball, dribble, kick, go_back_to_field
```

This hierarchy ensures:
- **Reusability**: Primitive behaviors are shared across tactics and roles.
- **Composability**: Higher layers orchestrate lower layers without duplicating logic.
- **Testability**: Each layer can be tested and tuned independently.

### Communication Flow

```
Decider (Python state machines)
    │  ZMQ REQ/REP (JSON)
    ▼
Simulation (MotrixSim)
    │  Observation tensor (47-dim)
    ▼
Policy Model (PyTorch .pt)
    │  Action tensor (12-dim)
    ▼
PD Controller → Joint Torques → MuJoCo Physics
```

## 2. Three-Layer State Machine Design

### Layer 1: Behavior Primitives (`logic/sub_statemachines/`)

Five primitive state machines provide atomic robot skills:

| Primitive | Function | Key Parameters |
|---|---|---|
| `find_ball` | Rotate in place to locate the ball when lost | Rotation timeout: 10s, vel_theta: 1.0 |
| `chase_ball` | Walk toward the ball position | Close angle threshold: 1.2 rad, walk_vel_x: 1.0 |
| `dribble` | P-control ball manipulation with goal alignment | P-gains: kp_x=1.5, kp_y=1.5, kp_theta=1.2 |
| `kick` | Approach and kick the ball toward the opponent goal | Good angle: ±10°, kick distance: 0.12-0.17m |
| `go_back_to_field` | Navigate to a designated field position | Coarse/fine yaw alignment with arc turning |

Each primitive exposes a standard interface: `enter()`, `execute()`, `exit()`. Transitions are triggered by condition checks (e.g., ball visible → chase, ball close → kick).

### Layer 2: Tactical Composites (`logic/strategy_statemachines/`)

Tactical state machines combine primitives into goal-directed behavior:

| Composite | Orchestrated Flow |
|---|---|
| `attack` | chase_ball → dribble → kick (with find_ball recovery on loss) |
| `defend_ball` | Position between ball and own goal, intercept when ball approaches |
| `dribble_ball` | Dribble toward opponent goal with obstacle avoidance and alignment |
| `shoot_ball` | Align behind ball, face goal, execute kick sequence |

### Layer 3: Role Policies (`logic/policy_statemachines/` + `user_entry.py`)

The top layer assigns roles to specific robot IDs and implements continuous control:

- **Goalkeeper** (`goalkeeper.py`): Position-based saving with ball velocity prediction, safe area enforcement, and early interception timing.
- **ContinuousPushController** (`user_entry.py`): A continuous error-space controller (no discrete FSM transitions) that unifies chase, behind-ball alignment, sideline repulsion, and push-to-goal into a single smooth control law.

## 3. 3v3 Team Coordination

### Role-Based Dispatch

Robots are assigned fixed roles by ID, dispatched from `user_entry.py:game()`:

```
Robot ID 0 → Attacker    (ContinuousPushController)
Robot ID 1 → Support     (ball-to-goal line positioning)
Robot ID 2 → Defender    (own-half anchor with lateral tracking)
```

### Attacker (ID 0) — Continuous Push Controller

The primary scoring robot uses a unified continuous control law:

1. **Chase phase** (far from ball): Proportional approach toward the ball position.
2. **Behind-ball alignment** (intermediate): Position behind the ball on the ball-to-goal vector. Lateral error is corrected with a P-controller.
3. **Push-to-goal** (aligned): Apply forward velocity to push the ball toward the opponent goal.
4. **Sideline repulsion**: A repulsive force keeps the robot away from field boundaries.

Key gains (configurable in `config.yaml → continuous_push`):
```
k_approach: 0.65    # Approach aggressiveness
k_depth: 2.3         # Behind-ball positioning gain
k_lat: 1.6           # Lateral alignment gain
k_yaw: 1.8           # Yaw alignment gain
k_sideline: 1.2      # Sideline repulsion strength
target_behind: 0.15  # Target distance behind ball (meters)
sideline_margin: 0.8 # Distance from sideline to activate repulsion
```

### Support (ID 1) — Ball-Line Positioning

The support robot positions itself ~1.2m behind the ball on the line from the ball to the opponent goal. This provides:
- Backup for loose balls
- Passing option
- Defensive coverage if possession is lost

### Defender (ID 2) — Half-Field Anchor

The defender anchors at X=-2m in the own half and tracks the ball's Y position laterally. This ensures:
- Last line of defense before the goalkeeper
- Coverage of the entire defensive width
- Readiness to intercept long balls

### Team Manager (`strategy/team_manager.py`)

A coordination layer (work-in-progress) handles role assignment, world model fusion across robots, and dynamic role switching. Currently, static role dispatch by ID is the active approach.

## 4. Velocity Parameters

All velocity commands are issued in the robot body frame and scaled by configurable maximums:

| Parameter | Config Value | Description |
|---|---|---|
| `walk_vel_x` | 0.5 | Forward velocity (m/s) before scaling |
| `walk_vel_y` | 0.5 | Lateral velocity (m/s) before scaling |
| `walk_vel_theta` | 1.0 | Rotational velocity (rad/s) before scaling |
| `max_walk_vel_x` | 1.5 | Maximum scaled forward velocity |
| `max_walk_vel_y` | 2.0 | Maximum scaled lateral velocity |
| `max_walk_vel_theta` | 3.0 | Maximum scaled rotational velocity |

The ZMQ `cmd` field `[vx, vy, w]` is clipped to `[-1, 1]` per dimension before transmission. The simulation then rescales by `max_walk_vel_*`.

## 5. Gait Model and Training Method

The submitted gait model is:

`gait/k1_walk_model_3600_motrixlab.pt`

This is a low-level MotrixLab K1 legged locomotion policy with a **47-dimensional observation** and **12-dimensional leg action**. It is not the Scheme A / AMP `375 -> 22` interface. The gait policy only produces leg joint position offsets; all ball, goal, opponent, and team strategy logic remains in the Decider layer.

### Reward Family

The gait policy follows a standard commanded humanoid locomotion reward design:

| Term | Purpose |
|---|---|
| Linear velocity tracking | Follow commanded forward/lateral velocity. |
| Angular velocity tracking | Follow commanded yaw rate. |
| Upright orientation | Keep the torso stable and reduce roll/pitch drift. |
| Base height stability | Maintain a usable walking height. |
| Joint/action regularization | Avoid extreme joint motion and abrupt action changes. |
| Energy regularization | Reduce high-effort unstable movements. |
| Foot contact and slip terms | Encourage usable stepping contacts and reduce sliding. |
| Termination/fall penalty | Discourage unstable falls or unsafe contacts. |

No soccer-specific reward is embedded into the gait policy.

### Training and Runtime Parameters

| Parameter | Value |
|---|---|
| Algorithm family | PPO-style Actor-Critic locomotion training |
| Submitted network | Actor-only inference graph |
| Physics timestep | 0.002 s |
| Control decimation | 10 |
| Control frequency | 50 Hz |
| Command scale | `[2.0, 2.0, 0.25]` |
| DOF velocity scale | `0.05` |
| Gait frequency | Approximately 1.5 Hz |

Detailed model interface and training notes are provided in `README_Gait.md` and `docs/training_notes.md`.

## 6. Robustness Strategies

### Ball Loss Recovery
When the ball moves out of the robot's field of view, `find_ball` rotates the robot through a predefined scan pattern (6 head angles) with a 10-second timeout. This ensures the ball is reacquired quickly in most situations.

### Sideline Safety
The `ContinuousPushController` includes a sideline repulsion term that activates within `sideline_margin=0.8m` of the field boundaries. The repulsive force scales inversely with distance, preventing out-of-bounds situations.

### Soft Velocity Clipping
Rather than hard-clipping velocity commands at max limits, a soft clipping function (`soft_clip_threshold: 0.7`) applies a smooth saturation curve, avoiding abrupt velocity discontinuities that could destabilize locomotion.

### Continuous Control (No Discrete States)
The attacker uses a continuous error-space controller instead of a discrete FSM. This eliminates state-transition deadlocks (e.g., oscillating between "chase" and "align" modes) and produces smoother robot trajectories.

### Config File Deep-Merge
The system supports optional `config_override.json/yaml/yml` files that deep-merge with `config.yaml` at load time. This allows per-match tuning without modifying the base configuration.

## 7. Configuration and Tuning

All tunable parameters live in `decider/config.yaml`. The configuration is loaded by `configuration.py` which exposes typed accessors. Key tuning areas:

- **Field size**: S (9×6m), M (14×9m), L (22×14m) — set via `league` field
- **Kick thresholds**: Attempt-to-dribble transitions based on ball Y-position in map frame
- **State machine gains**: Per-primitive P-control gains, velocity limits, angle thresholds
- **Vision**: Camera resolution, head control PID, ball prediction parameters
- **Obstacle avoidance**: Detection range, stop distance, safe corridor width

## 8. Dependencies

```
numpy >= 1.19.0
PyYAML >= 5.0
transitions >= 0.8.0
pyzmq >= 22.0.0
```

Python 3.8+ compatible. No ROS2, no CUDA, no hardware-specific libraries.
