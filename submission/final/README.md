# MotrixArena S2 Submission Package

## 1. Package Overview

This package contains the final deliverables for the MotrixArena S2 3v3 Humanoid Robot Simulation Challenge.

| Item | Path | Description |
|---|---|---|
| Gait / walk policy | `gait/k1_walk_model_3600_motrixlab.pt` | TorchScript K1 legged locomotion policy |
| Gait documentation | `README_Gait.md` | Model interface, observation/action layout, loading notes |
| Gait documentation (Chinese) | `README_Gait_CN.md` | Chinese version of the gait model notes |
| Decider source | `decider/` | Decision entrypoint, state machines, interfaces, config, dependencies |
| Decider documentation | `README_Decider.md` | Entrypoint, config fields, install and integration steps |
| Decider documentation (Chinese) | `README_Decider_CN.md` | Chinese version of the Decider notes |
| Technical solution | `docs/technical_solution.md` | Architecture, training strategy, coordination logic |
| Training notes | `docs/training_notes.md` | Reward family, training assumptions, runtime parameters |
| Submission checklist | `docs/submission_checklist.md` | Local verification record and remaining checks |

## 2. Version Information

| Field | Value |
|---|---|
| Submission version | `2026-06-12-final-candidate` |
| Robot | Booster K1 |
| Gait scheme | MotrixLab legged locomotion |
| Gait interface | `N x 47 -> N x 12` |
| Gait model SHA256 | `13aed9e30705f9564812564e68cf252b76061eea891cbc128df29274c11257e6` |
| Decider command | `[vx, vy, w]` velocity command |
| Default team color | `red` |
| Default robot id | `0` |
| Default field league | `M` |

The package uses a TorchScript `.pt` gait model. The competition notes recommend `.onnx` where possible, but also allow `.pt` TorchScript models. Because this submitted model is TorchScript, local model validation should use `torch.jit.load` rather than `onnxruntime`.

Documentation filenames are kept in English/ASCII (`README_CN.md`, `README_Gait_CN.md`, `README_Decider_CN.md`) to avoid Chinese characters in package paths.

## 3. Contact / Team

| Field | Value |
|---|---|
| Team ID in config | `12` |
| Contact | Use the official RoboGo / MotrixArena team account associated with this submission |

No private email address, cloud token, password, SSH key, or machine-specific credential is included in this package.

## 4. Change Summary

- Uses a stable MotrixLab K1 `47 -> 12` legged gait policy.
- Keeps game logic out of the gait model.
- Provides a Decider with role-based 3v3 logic and continuous push control.
- Sanitizes runtime networking defaults to `127.0.0.1`.
- Documents reward family, training assumptions, command limits, and submission checks.

## 5. Recommended Local Checks

Run from the package root:

```bash
python -m venv /tmp/motrix_submission_check
. /tmp/motrix_submission_check/bin/activate
pip install -r decider/requirements.txt
python -c "import yaml; yaml.safe_load(open('decider/config.yaml')); print('config OK')"
python -c "import torch; m=torch.jit.load('gait/k1_walk_model_3600_motrixlab.pt', map_location='cpu'); y=m(torch.zeros(1,47)); print(tuple(y.shape))"
```

Expected model output shape:

```text
(1, 12)
```

## 6. Directory Structure

```text
submission/final/
  README.md
  README_CN.md
  README_Gait.md
  README_Gait_CN.md
  README_Decider.md
  README_Decider_CN.md
  gait/
    k1_walk_model_3600_motrixlab.pt
  decider/
    user_entry.py
    decider.py
    config.yaml
    requirements.txt
    interfaces/
    logic/
    strategy/
  docs/
    technical_solution.md
    training_notes.md
    submission_checklist.md
```
