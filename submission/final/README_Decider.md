# Decider README — MotrixArena S2

## 1. Purpose

The Decider reads simulation/perception state and sends normalized body-frame velocity commands:

```json
{"cmd": [vx, vy, w], "id": 0, "timestamp": 0.0}
```

The Decider does not output joint targets directly. Joint control is handled by the gait policy and simulator PD layer.

## 2. Entrypoints

| File | Purpose |
|---|---|
| `decider/user_entry.py` | Main team behavior entry used by the submitted strategy |
| `decider/decider.py` | Simulation client loop and argument parsing |
| `decider/config.yaml` | Final submitted configuration |
| `decider/requirements.txt` | Python dependencies for the Decider |

## 3. Main Strategy

Default role mapping:

| Robot ID | Role | Behavior |
|---|---|---|
| 0 | Attacker | Continuous push controller, chase-align-push loop |
| 1 | Support | Position behind ball line for backup |
| 2 | Defender | Own-half anchor with lateral tracking |

The active attacker uses a continuous controller rather than a hard discrete kick-only routine. It combines:

- Ball chasing
- Behind-ball positioning
- Goal-directed pushing
- Sideline repulsion
- Soft velocity clipping

## 4. Key Configuration Fields

| Field | Meaning |
|---|---|
| `id` | Robot id used by this Decider instance |
| `team_id` | Team id |
| `color` | `red` or `blue` |
| `league` | Field size preset: `S`, `M`, or `L` |
| `walk_vel_x/y/theta` | Base command magnitudes |
| `max_walk_vel_x/y/theta` | Maximum simulator-side velocity scaling |
| `server_ip` | Simulation server IP, sanitized to `127.0.0.1` |
| `continuous_push.*` | Attacker control gains and velocity limits |

## 5. Install Check

From the package root:

```bash
python -m venv /tmp/motrix_decider_check
. /tmp/motrix_decider_check/bin/activate
pip install -r decider/requirements.txt
python -c "import yaml; yaml.safe_load(open('decider/config.yaml')); print('config OK')"
```

## 6. Simulation Integration

Typical local simulation command from the full repository:

```bash
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0
```

The simulator should be launched with the MotrixLab K1 gait flavor and the submitted policy:

```bash
python -m app.runner --robot-type k1 --k1-policy-flavor motrixlab --policy gait/k1_walk_model_3600_motrixlab.pt
```

For official evaluation, use the platform-provided runner and keep the Decider command format unchanged.

## 7. Submission Boundaries

- The Decider may implement soccer logic and team coordination.
- The gait model must not contain soccer logic.
- ZMQ JSON command format is unchanged.
- The package contains no ROS2 or hardware-specific dependency requirement.
- The code does not depend on `/opt` or other private machine paths.
