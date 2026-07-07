# 决策端说明 — MotrixArena S2

## 1. 功能说明

Decider 根据仿真/感知状态，输出机器人本体系速度指令：

```json
{"cmd": [vx, vy, w], "id": 0, "timestamp": 0.0}
```

Decider 不直接输出关节目标。关节控制由步态模型和仿真端 PD 控制层完成。

## 2. 入口文件

| 文件 | 作用 |
|---|---|
| `decider/user_entry.py` | 队伍策略主入口 |
| `decider/decider.py` | 仿真客户端循环和命令行参数解析 |
| `decider/config.yaml` | 最终提交配置 |
| `decider/requirements.txt` | 决策端 Python 依赖 |

## 3. 策略结构

默认机器人角色：

| 机器人 ID | 角色 | 行为 |
|---|---|---|
| 0 | 进攻 / 推球 | 连续推球控制，追球、绕到球后、向球门方向推球 |
| 1 | 支援 | 站在球到球门连线后方，准备补位 |
| 2 | 防守 | 己方半场锚点防守，跟踪球的横向位置 |

主进攻机器人使用连续控制器，而不是只依靠离散状态切换。控制器包含：

- 追球
- 到球后方对齐
- 向球门方向推球
- 边线排斥
- 速度软限幅

## 4. 关键配置字段

| 字段 | 含义 |
|---|---|
| `id` | 当前 Decider 控制的机器人 ID |
| `team_id` | 队伍 ID |
| `color` | 队伍颜色，`red` 或 `blue` |
| `league` | 场地大小，`S` / `M` / `L` |
| `walk_vel_x/y/theta` | 基础速度指令大小 |
| `max_walk_vel_x/y/theta` | 仿真侧最大速度缩放 |
| `server_ip` | 仿真服务器 IP，提交包中设为 `127.0.0.1` |
| `continuous_push.*` | 连续推球控制参数 |

## 5. 依赖安装检查

在提交包根目录执行：

```bash
python -m venv /tmp/motrix_decider_check
. /tmp/motrix_decider_check/bin/activate
pip install -r decider/requirements.txt
python -c "import yaml; yaml.safe_load(open('decider/config.yaml')); print('config OK')"
```

## 6. 仿真联调方式

在完整仓库中，典型 Decider 启动命令为：

```bash
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0
```

仿真端应使用 MotrixLab K1 步态方案和提交的步态模型：

```bash
python -m app.runner --robot-type k1 --k1-policy-flavor motrixlab --policy gait/k1_walk_model_3600_motrixlab.pt
```

正式评测时应以平台提供的启动方式为准，并保持标准 ZMQ JSON 命令格式不变。

## 7. 提交边界

- Decider 可以包含足球策略、多机器人协同和状态机逻辑。
- 步态模型不能包含足球比赛逻辑。
- ZMQ JSON 命令格式没有增加或删除字段。
- 提交包不依赖 ROS2、硬件服务或 `/opt` 等机器私有路径。
