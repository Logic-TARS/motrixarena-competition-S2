# K1 双机并行训练

## 分支与职责

- 机器 A：`train/k1-robust-walk`，训练 `k1-flat-terrain-walk`。
- 机器 B：`train/k1-getup`，训练 `k1-getup`。
- 集成分支：`feature/k1-recovery-runtime`，维护恢复状态机、参数和评测。

训练产物位于 `MotrixLab/runs/`，已被 Git 忽略。不要提交 TensorBoard 日志或中间检查点。

## 机器 A

```bash
git switch train/k1-robust-walk
cd MotrixLab
BASE_POLICY=runs/k1-flat-terrain-walk/rslrl/<run>/model_1350.pt \
  NUM_ENVS=4096 SEED=1 ./scripts/train_k1_robust_walk.sh
```

另开终端，每 500 轮自动评测：

```bash
cd MotrixLab
ENV_NAME=k1-flat-terrain-walk SEED=1 ./scripts/monitor_k1_candidates.sh
```

## 机器 B

先用 2048 个环境压测显存；稳定后设置 `NUM_ENVS=4096`。

```bash
git switch train/k1-getup
cd MotrixLab
NUM_ENVS=2048 SEEDS="1 2 3" ./scripts/train_k1_getup_seeds.sh
```

该脚本默认持续轮换种子且每次生成独立运行目录。设置 `CONTINUOUS=0` 可只跑一轮种子。

另开终端，每 500 轮评测 200 个随机倒地样本：

```bash
cd MotrixLab
ENV_NAME=k1-getup SEED=1 GETUP_EPISODES=200 \
  ./scripts/monitor_k1_candidates.sh
```

## 候选产物

监控器将候选模型写入 `MotrixLab/runs/candidates/`，每个目录包含：

- `model.pt`
- `evaluation.json`
- 训练环境配置快照
- `model.pt.manifest.json`，记录 Git commit、种子、迭代数和 SHA256

使用 `rsync` 或 `scp` 回传整个候选目录。行走模型应满足压力测试跌倒率不高于 1%，并在原事故指令下连续 60 秒不跌倒。起身模型应满足总体成功率不低于 95%、单类不低于 90%、P95 小于 12 秒且所有成功案例小于 20 秒。

## 集成运行

```bash
./scripts/start_sim.sh \
  --policy /path/to/walk_policy.pt \
  --recovery-policy /path/to/getup_policy.pt
```

恢复状态依次为 `LOCOMOTION -> FALLEN -> RECOVERING -> STABILIZING`。策略恢复只写关节目标，禁止写入基座位置、四元数或速度来瞬移扶正。
