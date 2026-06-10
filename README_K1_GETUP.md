# K1 Get-Up Training Branch

This branch is for training the K1 autonomous get-up policy. It is the machine B side of the dual-machine plan: machine A trains robust walking, while this branch trains recovery from fallen poses.

## Purpose

The goal is to train a full-body policy that can recover from:

- Supine falls
- Prone falls
- Left and right side falls
- Random fallen poses with randomized joints and base velocity

The training environment is `k1-getup`. Its policy interface is:

```text
78 observations -> 22 actions
```

The policy is intended to be used by the runtime recovery state machine after a fall is detected.

## Key Files

```text
MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/getup_np.py
MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/getup_cfg.py
MotrixLab/motrix_rl/src/motrix_rl/tasks/k1.py
MotrixLab/scripts/train_k1_getup.sh
MotrixLab/scripts/train_k1_getup_seeds.sh
MotrixLab/scripts/eval_k1_getup.py
MotrixLab/scripts/monitor_k1_candidates.sh
docs/K1_DUAL_MACHINE_TRAINING.md
```

## Train

Start with 2048 environments on a 24 GB GPU. If memory is stable, raise `NUM_ENVS` to 4096.

Single seed:

```bash
cd MotrixLab
NUM_ENVS=2048 SEED=1 ./scripts/train_k1_getup.sh
```

Multiple seeds with continuous seed rotation:

```bash
cd MotrixLab
NUM_ENVS=2048 SEEDS="1 2 3" ./scripts/train_k1_getup_seeds.sh
```

Run only one seed round:

```bash
cd MotrixLab
CONTINUOUS=0 NUM_ENVS=2048 SEEDS="1 2 3" ./scripts/train_k1_getup_seeds.sh
```

Useful overrides:

```bash
MAX_ITERATIONS=10000
CUDA_VISIBLE_DEVICES=0
```

## Evaluate

Evaluate a checkpoint on random fallen samples:

```bash
cd MotrixLab
uv run --frozen python scripts/eval_k1_getup.py \
  --policy runs/k1-getup/rslrl/<run>/model_<iter>.pt \
  --episodes 200 \
  --output getup_eval.json
```

Monitor checkpoints and copy passing candidates into `runs/candidates/`:

```bash
cd MotrixLab
ENV_NAME=k1-getup SEED=1 GETUP_EPISODES=200 ./scripts/monitor_k1_candidates.sh
```

Each candidate directory should contain:

- `model.pt`
- `evaluation.json`
- A training config snapshot
- `model.pt.manifest.json` with Git commit, seed, iteration, and SHA256

## Promotion Criteria

A get-up policy is a candidate only if it meets all of these:

- Overall success rate is at least 95%
- Each pose class success rate is at least 90%
- P95 get-up time is below 12 seconds
- Every successful recovery finishes within 20 seconds

## Runtime Integration

The runtime branch should load walking and recovery policies separately:

```bash
./scripts/start_sim.sh \
  --policy /path/to/walk_policy.pt \
  --recovery-policy /path/to/getup_policy.pt
```

The recovery state machine is:

```text
LOCOMOTION -> FALLEN -> RECOVERING -> STABILIZING -> LOCOMOTION
```

Recovery must only write joint targets. Do not implement fall recovery by writing base position, base quaternion, or base velocity to teleport the robot upright.

## Git Hygiene

Do not commit training outputs:

- `MotrixLab/runs/`
- TensorBoard logs
- Intermediate checkpoints
- Local candidate model archives

Transfer candidate models, evaluation JSON files, and config snapshots between machines with `rsync` or `scp`.
