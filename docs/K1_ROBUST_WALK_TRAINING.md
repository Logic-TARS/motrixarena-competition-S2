# K1 鲁棒行走训练（机器 A）

> 分支：`train/k1-robust-walk` | 基座模型：model_1350

## 训练目标

从 model_1350 热启动，让 K1 学会：

- **大角度转弯** — yaw 课程 ±0.3 → ±0.6 → ±1.0 → ±1.5 rad/s
- **对称步态** — 不偏航、不侧漂
- **抗推扰** — 最高 0.4 m/s 推搡不倒
- **不摔** — 摔倒一次罚 -10，逼模型学会自保

## 快速启动

```bash
git clone git@github.com:Logic-TARS/motrixlab-soccer-3v3.git
cd motrixlab-soccer-3v3
git switch train/k1-robust-walk

# 下载 model_1350 基座（如果本地没有）
# scp 或 rsync 从训练机取回 MotrixLab/runs/k1-flat-terrain-walk/rslrl/<run>/model_1350.pt

cd MotrixLab
bash scripts/train_k1_robust_walk.sh
```

默认参数：4096 envs、seed=1、最多 5000 轮迭代、噪声 0.30。

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `lin_vel_x` | [0.0, 0.8] | 只走正向，0~0.8 m/s |
| `ang_vel_yaw` | [-1.5, 1.5] | 4 级课程递进 |
| `termination` | -10.0 | 摔倒重罚 |
| `straight_motion` | -1.5 | 惩罚偏航/侧移 |
| `only_positive_rewards` | False | 允许负奖励 |
| `resampling_time` | 3.0 s | 指令频繁变化 |
| `stand/straight/turn` | 10/25/20% | 多样化命令模式 |
| `max_push_vel_xy` | 0.4 | 强推扰 |

完整配置见 `MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/cfg.py`。

## 自动评测

另开终端，每 500 轮自动跑命令网格：

```bash
cd MotrixLab
ENV_NAME=k1-flat-terrain-walk SEED=1 bash scripts/monitor_k1_candidates.sh
```

候选模型写入 `MotrixLab/runs/candidates/`，附带 `evaluation.json` + SHA256 清单。

## 晋级标准

- 压力测试跌倒率 ≤ 1%
- 原事故指令下连续 60 秒不跌倒

## 回传产物

```bash
rsync -avP MotrixLab/runs/candidates/<candidate_dir>/ user@host:/path/to/models/
```

## 导出用于仿真

```bash
cd MotrixLab
uv run python scripts/export_k1_rslrl_torchscript.py \
    runs/k1-flat-terrain-walk/rslrl/<run>/model_NNN.pt \
    -o exported/model_NNN_torchscript.pt
```

详见 `docs/K1_DUAL_MACHINE_TRAINING.md`。
