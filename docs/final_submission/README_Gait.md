# K1 Walk Gait Model — MotrixArena S2 Submission

## Model Overview

| Property | Value |
|---|---|
| **File** | `gait/k1_walk_model_3600_motrixlab.pt` |
| **Size** | ~778 KB |
| **SHA256** | `13aed9e30705f9564812564e68cf252b76061eea891cbc128df29274c11257e6` |
| **Architecture** | 47-dim observation → 12-dim leg action (legged locomotion policy) |
| **Format** | PyTorch checkpoint (can be loaded with `torch.jit.load` or `torch.load`) |
| **Flavor** | `motrixlab` (trained with MotrixLab/motrix_envs `k1-flat-terrain-walk`) |
| **Control Frequency** | 50 Hz |
| **Gait Frequency** | ~1.5 Hz |

## Interface Scheme

This package uses the MotrixLab K1 legged locomotion scheme: **47-dimensional observation → 12-dimensional leg action**.

Some MotrixArena planning documents also describe a Scheme A / AMP-style interface with **375-dimensional observation → 22-dimensional action**. That is a separate gait scheme and is not used by this package. The submitted model keeps the official MotrixLab legged interface and does not add custom observation fields.

Because this model is submitted as TorchScript `.pt`, validate it with `torch.jit.load`. `onnxruntime` validation applies only if an `.onnx` export is submitted.

## Input: Observation Space (47 dimensions)

The policy expects a 47-element float32 observation vector with the following layout (MotrixLab flavor):

| Index Range | Component | Description |
|---|---|---|
| 0:3 | Base linear velocity | Robot body linear velocity (vx, vy, vz) in body frame |
| 3:6 | Base angular velocity | Robot body angular velocity (roll, pitch, yaw rate) |
| 6:9 | Projected gravity | Gravity vector projected into body frame |
| 9:12 | Commands | Target (vx, vy, yaw_rate) in body frame, scaled by `[2.0, 2.0, 0.25]` |
| 12:24 | DOF positions | 12 leg joint positions, scale=1.0 |
| 24:36 | DOF velocities | 12 leg joint velocities, scale=0.05 |
| 36:47 | Last actions | Previous 11 action outputs (or padded to 11) |

**Observation scales (MotrixLab flavor):**
- `cmd_scale`: `[2.0, 2.0, 0.25]`
- `dof_pos_scale`: `1.0`
- `dof_vel_scale`: `0.05`
- `gyro_scale`: `0.25`

## Output: Action Space (12 dimensions)

The policy outputs a 12-element float32 vector mapping to 12 leg joint position **offsets** (relative to default pose). Each dimension is scaled by a joint-specific action scale before being applied as a PD target.

| Index | Joint | Action Scale | Torque Limit (Nm) | KP | KD |
|---|---|---|---|---|---|
| 0 | Left Hip Pitch | 0.1700 | 68.0 | 100.0 | 2.0 |
| 1 | Right Hip Pitch | 0.1700 | 68.0 | 100.0 | 2.0 |
| 2 | Left Hip Roll | 0.1900 | 76.0 | 100.0 | 2.0 |
| 3 | Right Hip Roll | 0.1900 | 76.0 | 100.0 | 2.0 |
| 4 | Left Hip Yaw | 0.09575 | 38.3 | 100.0 | 2.0 |
| 5 | Right Hip Yaw | 0.09575 | 38.3 | 100.0 | 2.0 |
| 6 | Left Knee Pitch | 0.1867 | 112.0 | 150.0 | 4.0 |
| 7 | Right Knee Pitch | 0.1867 | 112.0 | 150.0 | 4.0 |
| 8 | Left Ankle Pitch | 0.2394 | 38.3 | 40.0 | 2.0 |
| 9 | Right Ankle Pitch | 0.2394 | 38.3 | 40.0 | 2.0 |
| 10 | Left Ankle Roll | 0.2394 | 38.3 | 40.0 | 2.0 |
| 11 | Right Ankle Roll | 0.2394 | 38.3 | 40.0 | 2.0 |

## Loading the Model

### In PyTorch
```python
import torch

# Method 1: TorchScript (recommended)
model = torch.jit.load("gait/k1_walk_model_3600_motrixlab.pt")
model.eval()

# Method 2: State dict checkpoint
checkpoint = torch.load("gait/k1_walk_model_3600_motrixlab.pt", map_location="cpu")
# Extract actor state_dict and rebuild network as needed
```

### In Simulation (MotrixSim)
```bash
# The simulation auto-loads this model by default when using motrixlab flavor:
python -m app.runner --robot-type k1 --k1-policy-flavor motrixlab

# To explicitly specify this model file:
python -m app.runner --robot-type k1 --policy gait/k1_walk_model_3600_motrixlab.pt

# To use GPU inference:
python -m app.runner --robot-type k1 --policy-device gpu
```

## Running Parameters

| Parameter | Recommended Value |
|---|---|
| Control frequency | 50 Hz |
| `--k1-policy-flavor` | `motrixlab` |
| `--k1-legged-gym` | Enabled (default) |
| Command velocity range | vx: [-1.0, 1.0], vy: [-1.0, 1.0], w: [-1.0, 1.0] |
| `walk_vel_x` (config) | 0.5 m/s (scaled by max_walk_vel_x=1.5) |
| `walk_vel_y` (config) | 0.5 m/s (scaled by max_walk_vel_y=2.0) |
| `walk_vel_theta` (config) | 1.0 rad/s (scaled by max_walk_vel_theta=3.0) |

## Important Notes

1. **Single input, single output**: The model is Actor-only (no Critic). It takes one observation tensor and produces one action tensor.
2. **No game logic**: This model only handles locomotion ("how to walk"). All tactical decisions ("where to walk", shooting, passing, positioning) are handled by the Decider module.
3. **Velocity commands**: The Decider sends `[vx, vy, w]` velocity commands through ZMQ. These are clipped to `[-1, 1]` and scaled into the observation before inference.
4. **Joint limits**: Torque limits and PD gains are applied post-inference. Do not remove or bypass these safety limits.
5. **Path restrictions**: Model file paths must not contain Chinese characters, spaces, or system-specific directories (e.g., `/opt`).
6. **Compatibility**: This model is compatible with `--k1-policy-flavor motrixlab`. Using `legged_gym` flavor will apply different observation scaling and may degrade performance.
7. **Training documentation**: Reward design, training assumptions, runtime command limits, and integration notes are documented in `docs/training_notes.md`.
