# Submission Checklist — MotrixArena S2

Use this checklist before submitting your package to the RoboGo platform.

## Model Compliance

- [x] **Input/output dimensions match selected official scheme**
  - Our submission: 47-dim observation → 12-dim action (legged locomotion, MotrixLab flavor)
  - Note: this is the MotrixLab legged scheme, not the separate Scheme A / AMP 375-dim → 22-dim interface.
  - Verify: `runtime_config.py` sets `K1_LEGGED_GYM_NUM_OBS=47`, `K1_LEGGED_GYM_NUM_ACT=12`
- [x] **Single input, single output**
  - The model file contains only the Actor network. No Critic, no auxiliary heads.
  - Verified: `torch.jit.load(...)` accepts a `1 x 47` input and returns a `1 x 12` output.
- [x] **Model is Actor-only (no Critic)**
  - Training may have used asymmetric Actor-Critic, but the exported model must retain only the policy network.
- [x] **Model file path is clean**
  - No Chinese characters, no spaces, no absolute system paths (e.g., `/opt/`, `/home/`).
  - Verified: `find . -name "*[一-龥]*"` returned 0 matches.
- [x] **Model checksum recorded**
  - SHA256: `13aed9e30705f9564812564e68cf252b76061eea891cbc128df29274c11257e6`

## Module Boundaries

- [x] **No game logic in the gait model**
  - The model only outputs joint position targets. All tactical decisions (shooting, passing, positioning, role assignment) live in the Decider module (`decider/`).
  - Verify: Model output is purely 12 floating-point joint offsets.
- [x] **ZMQ message format is unchanged**
  - The Decider communicates with the simulation via standard ZMQ JSON packets:
    ```json
    {"cmd": [vx, vy, w], "id": N, "timestamp": T}
    ```
  - No custom fields added, no standard fields removed or renamed.

## Security & Privacy

- [x] **No plaintext secrets**
  - No API keys, passwords, private tokens, or cloud credentials in the submission.
  - Verified: only the intentional `PLACEHOLDER` token string was found by the quick leakage check.
- [x] **No private network information**
  - `server_ip` is set to a safe default (`127.0.0.1`). No private LAN IPs or hostnames.
  - No SSH keys, `.pem` files, or certificate files included.
- [x] **No personal environment paths**
  - All file references use relative paths or well-known locations. No `/home/user/...` or `/opt/...` paths.

## Configuration Consistency

- [x] **Velocity limits match the gait model**
  - `config.yaml` velocity parameters (`walk_vel_x/y/theta`, `max_walk_vel_*`) are documented and remain within the normalized command path used by the submitted model.
  - Long-run fall/stability validation is tracked separately below.
- [x] **Policy flavor matches simulation flags**
  - Our model uses `motrixlab` flavor. The simulation must be launched with `--k1-policy-flavor motrixlab`.
  - Verify: Run `python -m app.runner --robot-type k1 --k1-policy-flavor motrixlab` and confirm model loads successfully.
- [x] **Field size league is correctly set**
  - `config.yaml` sets `league: "M"` (14×9m field). Adjust if the competition uses a different size.

## File Integrity

- [x] **No excluded file types in the package**
  - No `__pycache__/`, `*.pyc`, `*.log`, `*.mp4`, `.git/`, or temporary files.
  - Verified: excluded-file scan returned 0 matches.
- [x] **No disabled/legacy code in the active submission**
  - Excluded: `old_version.py.disable`, `test.py`, `scripts/`, hardware service files.
  - Included only: Core decider logic, interfaces, state machines, and configuration.
- [x] **Model file is not corrupted**
  - Verified: `torch.jit.load(...)` succeeded and returned output shape `(1, 12)`.
- [x] **config.yaml is valid YAML**
  - Verified: `yaml.safe_load(open('decider/config.yaml'))` succeeded.

## Documentation

- [x] **Technical solution document is complete** (`docs/technical_solution.md`)
  - Covers: architecture, 3v3 coordination logic, velocity parameters, robustness strategies, reward family, and training/runtime parameters.
- [x] **Model specification document is complete** (`README_Gait.md`)
  - Covers: input/output dimensions, action scales, PD gains, loading instructions, running parameters.
- [x] **Training methodology notes are included** (`docs/training_notes.md`)
  - Covers selected gait scheme, reward family, training assumptions, runtime command limits, and sim-to-game integration.
- [x] **Top-level package documentation is included** (`README.md`, `README_Decider.md`)
  - Covers version information, contact/team field, change summary, Decider entrypoint, dependencies, and integration steps.
- [x] **Chinese README files are included with ASCII filenames**
  - Included: `README_CN.md`, `README_Gait_CN.md`, `README_Decider_CN.md`.
  - Filenames remain ASCII to satisfy path restrictions.
- [ ] **Long-run official environment verification**
  - Recommended before upload: run an uninterrupted official-environment match or 20-minute simulation smoke test and record the result here.

## Pre-Submission Quick Verify

Run these commands from `final_submission/`:

```bash
# 1. Check directory structure
find . -type f | sort

# 2. Verify config parses
python -c "import yaml; yaml.safe_load(open('decider/config.yaml')); print('config OK')"

# 3. Verify model size
python -c "import os; s=os.path.getsize('gait/k1_walk_model_3600_motrixlab.pt'); print(f'model: {s} bytes')"

# 3b. Verify model checksum
sha256sum gait/k1_walk_model_3600_motrixlab.pt

# 4. Check for excluded patterns (should return nothing)
find . \( -name "__pycache__" -o -name "*.pyc" -o -name "*.log" -o -name "*.mp4" \)

# 5. Check for Chinese in paths (should return nothing)
find . -name "*[一-龥]*"

# 6. Check for token leakage (should return nothing or only "PLACEHOLDER")
grep -r "NjM5NWVkMSAgLQo" .
```

## Latest Local Verification

Executed from `final_submission/`:

```bash
python -c "import yaml; yaml.safe_load(open('decider/config.yaml')); print('config OK')"
uv run --with torch python -c "import torch; m=torch.jit.load('gait/k1_walk_model_3600_motrixlab.pt', map_location='cpu'); m.eval(); y=m(torch.zeros(1,47)); print('model OK', tuple(y.shape))"
find . \( -name "__pycache__" -o -name "*.pyc" -o -name "*.log" -o -name "*.mp4" -o -path "*/.git/*" \)
find . -name "*[一-龥]*"
grep -R "NjM5NWVkMSAgLQo" .
```

Result:

- Config parse: OK
- Model load/forward: OK, output shape `(1, 12)`
- Model size: 778341 bytes
- Model SHA256: `13aed9e30705f9564812564e68cf252b76061eea891cbc128df29274c11257e6`
- Clean venv dependency install: OK, `pip install -r decider/requirements.txt`
- Decider Python syntax compile: OK
- Excluded files: 0
- Chinese paths: 0
- Old `models/` references: 0
- Token placeholder hits: 1 expected `PLACEHOLDER`
