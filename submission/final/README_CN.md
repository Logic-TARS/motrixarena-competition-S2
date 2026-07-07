# MotrixArena S2 提交包说明

## 1. 提交包总览

本目录是 MotrixArena S2 3v3 人形机器人仿真挑战赛的最终提交候选包。

| 交付物 | 路径 | 说明 |
|---|---|---|
| 步态 / 行走模型 | `gait/k1_walk_model_3600_motrixlab.pt` | TorchScript 格式的 K1 腿部步态模型 |
| 步态说明 | `README_Gait.md` / `README_Gait_CN.md` | 模型接口、观测/动作维度、归一化和加载方式 |
| 决策端源码 | `decider/` | 决策入口、状态机、接口、配置和依赖 |
| 决策端说明 | `README_Decider.md` / `README_Decider_CN.md` | 入口脚本、配置字段、安装和联调步骤 |
| 技术方案 | `docs/technical_solution.md` | 算法架构、训练策略、多机协同方案 |
| 训练说明 | `docs/training_notes.md` | 奖励设计、训练假设、运行参数 |
| 提交自检 | `docs/submission_checklist.md` | 本地验证记录和剩余检查项 |

## 2. 版本信息

| 字段 | 值 |
|---|---|
| 提交版本 | `2026-06-12-final-candidate` |
| 机器人 | Booster K1 |
| 步态方案 | MotrixLab legged locomotion |
| 步态接口 | `N x 47 -> N x 12` |
| 步态模型 SHA256 | `13aed9e30705f9564812564e68cf252b76061eea891cbc128df29274c11257e6` |
| 决策端输出 | `[vx, vy, w]` 速度指令 |
| 默认队伍颜色 | `red` |
| 默认机器人 ID | `0` |
| 默认场地 | `M` |

本提交使用 `.pt` TorchScript 步态模型。比赛说明推荐 `.onnx`，但也允许 `.pt` TorchScript；因此本模型的本地验证方式是 `torch.jit.load`，不是 `onnxruntime`。

注意：文件名保持英文是为了避免提交路径中出现中文字符。中文说明内容本身可以存在于 `.md` 文件中。

## 3. 联系与团队信息

| 字段 | 值 |
|---|---|
| 配置中的 Team ID | `12` |
| 联系方式 | 使用官方 RoboGo / MotrixArena 平台绑定的队伍账号 |

本提交包不包含私人邮箱、云盘 Token、密码、SSH Key 或机器专用凭据。

## 4. 变更摘要

- 使用 MotrixLab K1 `47 -> 12` 腿部步态模型。
- 步态模型只负责行走控制，不硬编码比赛逻辑。
- 决策端包含 3v3 角色分工、连续推球控制和防守/支援逻辑。
- 网络默认配置已清理为 `127.0.0.1`。
- 补充了奖励设计、训练假设、命令限制和提交自检记录。

## 5. 推荐本地检查

在提交包根目录运行：

```bash
python -m venv /tmp/motrix_submission_check
. /tmp/motrix_submission_check/bin/activate
pip install -r decider/requirements.txt
python -c "import yaml; yaml.safe_load(open('decider/config.yaml')); print('config OK')"
python -c "import torch; m=torch.jit.load('gait/k1_walk_model_3600_motrixlab.pt', map_location='cpu'); y=m(torch.zeros(1,47)); print(tuple(y.shape))"
```

期望模型输出：

```text
(1, 12)
```

## 6. 目录结构

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
