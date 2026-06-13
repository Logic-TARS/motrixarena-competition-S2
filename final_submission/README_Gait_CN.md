# K1 步态模型说明 — MotrixArena S2

## 1. 模型概览

| 字段 | 值 |
|---|---|
| 模型文件 | `gait/k1_walk_model_3600_motrixlab.pt` |
| 大小 | 约 778 KB |
| SHA256 | `13aed9e30705f9564812564e68cf252b76061eea891cbc128df29274c11257e6` |
| 架构 | 47 维观测 -> 12 维腿部动作 |
| 格式 | PyTorch TorchScript `.pt` |
| 方案 | `motrixlab` |
| 控制频率 | 50 Hz |
| 步态频率 | 约 1.5 Hz |

## 2. 接口方案说明

本提交采用 MotrixLab K1 腿部步态方案：**47 维观测 -> 12 维腿部动作**。

部分比赛资料中提到的方案 A / AMP 接口是 **375 维观测 -> 22 维动作**。那是另一套步态方案，不是本提交使用的方案。本提交没有额外增加观测维度，也没有改变官方 MotrixLab legged 接口。

因为本模型提交格式是 TorchScript `.pt`，所以本地验证应使用 `torch.jit.load`。只有在提交 `.onnx` 模型时，才需要使用 `onnxruntime` 做模型验证。

## 3. 输入观测 47 维

| 维度范围 | 内容 | 说明 |
|---|---|---|
| 0:3 | 机体线速度 | 机器人本体系 vx, vy, vz |
| 3:6 | 机体角速度 | roll, pitch, yaw rate |
| 6:9 | 重力方向 | 投影到机器人本体系的重力向量 |
| 9:12 | 速度指令 | `[vx, vy, yaw_rate]`，按 `[2.0, 2.0, 0.25]` 缩放 |
| 12:24 | 关节位置 | 12 个腿部关节位置 |
| 24:36 | 关节速度 | 12 个腿部关节速度，缩放系数 0.05 |
| 36:47 | 历史动作 | 上一帧动作历史 |

主要归一化参数：

- `cmd_scale`: `[2.0, 2.0, 0.25]`
- `dof_pos_scale`: `1.0`
- `dof_vel_scale`: `0.05`
- `gyro_scale`: `0.25`

## 4. 输出动作 12 维

模型输出 12 个腿部关节的位置偏移量，相对于默认姿态施加。输出会经过关节尺度缩放，再由仿真端 PD 控制器转换为关节控制目标。

| 序号 | 关节 | Action Scale | Torque Limit | KP | KD |
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

## 5. 加载方式

```python
import torch

model = torch.jit.load("gait/k1_walk_model_3600_motrixlab.pt", map_location="cpu")
model.eval()
output = model(torch.zeros(1, 47))
print(output.shape)  # torch.Size([1, 12])
```

## 6. 重要边界

1. 模型是单输入、单输出。
2. 提交模型只保留 Actor 推理网络，不包含 Critic。
3. 模型不包含比赛逻辑，不读取球、球门、比分、对手等比赛状态。
4. 射门、推球、站位、防守、多机器人协同都在 `decider/` 中实现。
5. 模型路径不包含中文、空格或 `/opt` 等机器私有路径。
