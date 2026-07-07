# ⚽ MotrixArena S2 3v3 机器人足球仿真比赛项目

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Decision](https://img.shields.io/badge/Decider-State%20Machine-FF6F00)
![Communication](https://img.shields.io/badge/Communication-ZMQ-555555)
![Robot](https://img.shields.io/badge/Robot-K1%20%7C%20Pi%20Plus-4B8BBE)

> 本仓库是 **MotrixArena S2 机器人足球仿真比赛** 中的多机器人决策控制与仿真工程实践。  
> 项目面向 **K1 / Pi Plus 双足或多足机器人足球任务**，基于 **MotrixSim + Decider + ZMQ** 构建感知、决策、控制、回放和诊断链路。

---

## ✅ 项目亮点 / 可验证结果

- **3v3 多机器人足球决策**：支持 attacker / support / defender 角色分工，按机器人 ID 自动分配策略。
- **Decider 决策框架**：上层策略通过状态机实现，屏蔽仿真进程、网络通信和底层控制复杂度。
- **连续推球控制器**：实现连续推球控制器，通过球-门方向、机器人位姿、横向误差、朝向误差和边线风险计算速度命令。
- **多层状态机结构**：基础动作层、战术层、角色策略层分离，便于扩展找球、追球、盘带、射门、守门等行为。

---

## 🧩 技术挑战与解决方案

| 问题 | 解决方法 | 产生效果 |
|---|---|---|
| 多机器人足球任务中，单机器人追球策略容易出现扎堆、互相干扰和角色不清晰 | 设计 attacker / support / defender 三角色策略：0 号进攻推球，1 号支援站位，2 号防守站位 | 形成基础 3v3 协同框架，降低多机器人同时抢球导致的干扰 |
| 机器人只追球容易把球推向边线，无法稳定朝球门推进 | 设计 `ContinuousPushController`，计算 behind-depth、lateral error、yaw error、sideline risk 等误差项，并连续生成速度指令 | 将“追球”转化为“站到球后方并沿球门方向推球”，提升进攻动作的方向性 |
| 状态机频繁切换会造成控制不连续，机器人接近球后容易震荡 | 使用连续误差空间控制，结合距离相关权重、速度限幅、soft-clip 和近球角速度阻尼 | 减少模式切换带来的抖动，使接近、对齐、推球过程更连续 |
| 多机器人比赛调试困难，肉眼难以判断失败原因 | 增加视频录制、轨迹记录和诊断脚本，记录机器人位姿、球位置、速度命令、FSM 状态和对齐信息 | 可复盘每一帧决策，定位“没看到球、没站到球后、推球方向偏、靠边线”等问题 |
---

## 📊 实验结果 / 工程结果

| 指标 | 结果 |
|---|---|
| 主要机器人 | K1 / Pi Plus |
| 决策框架 | Decider 状态机 |
| 仿真平台 | MotrixSim，兼容 Isaac Sim 历史实现 |
| 通信方式 | ZMQ，多机器人独立端口通信 |
| 进攻策略 | `ContinuousPushController` 连续推球控制 |
| 比赛排名 | 第10名 优胜奖 |
| 进球率 | 高于70% |

### 本地示例运行结果

以下材料来自本地 demo / baseline 运行，用于展示工程链路和诊断能力，不代表最终比赛排名或官方成绩。

| 材料 | 链接 | 说明 |
|---|---|---|
| 3v3 demo 回放 | [demo-match-20260614-105108.mp4](docs/assets/demo/demo-match-20260614-105108.mp4) | 本地录制的短回放片段 |
| 轨迹时间序列 | [demo-trajectory-timeseries.png](docs/assets/demo/demo-trajectory-timeseries.png) | 机器人、球和命令随时间变化 |
| Locomotion 速度跟踪 | [loco-v030-velocity-tracking.png](docs/assets/demo/loco-v030-velocity-tracking.png) | `T1_forward_velocity/v030` 的速度跟踪诊断图 |

Locomotion baseline `20260614_135745_default` 共记录 3 个 case，结果为 `0 pass / 3 fail`，acceptance rating 为 `Not recommended`。这些失败样例用于暴露底层步态跟踪和稳定性问题，支撑后续调参与模型迭代。

---

## 🎯 比赛任务与难点

机器人足球任务要求多个机器人在仿真场地内完成找球、追球、对齐、推球、射门、防守等行为。相比单机器人导航任务，S2 的难点主要在于：

- **多机器人协同**：多个机器人不能只做同一种追球行为，否则容易扎堆和互相阻挡。
- **球权控制**：机器人需要站到球的后方，而不是简单冲向球。
- **方向控制**：推球方向需要对准对方球门，同时避免将球推向边线。
- **连续控制**：接近球、绕到球后、推球、射门之间需要平滑衔接。
- **可调试性**：比赛失败往往不是单一 bug，而是感知、状态机、控制参数和队形策略共同作用的结果。

---

## 🧠 核心方案

### 1. Decider + Simulation 双模块架构

项目主要分为两部分：

```text
motrixarena-competition-S2/
├── decider/                   # 决策模块：机器人“大脑”
│   ├── decider.py             # Decider 入口
│   ├── user_entry.py          # 自定义策略入口
│   ├── config.yaml            # 决策参数配置
│   ├── interfaces/            # Action / Vision / GameController / SimClient
│   └── logic/                 # 状态机与策略逻辑
├── simulation/                # 仿真模块
│   ├── motrixsim/             # MotrixSim 主仿真环境
│   ├── isaac_sim/             # Isaac Sim 历史实现
│   └── labbridge/             # WebView / Bridge / Sim Manager
├── MotrixLab/                 # K1 locomotion / RL 训练子项目
├── models/k1/                 # 默认 K1 policy 模型
├── docs/                      # 启动、训练、比赛文档
├── tools/                     # 维护脚本
└── scripts/                   # 常用启动和录制脚本
```

Decider 负责策略逻辑，Simulation 负责物理仿真和可视化，两者通过 ZMQ 通信。

### 2. 三层状态机组织

| 层级 | 目录 | 作用 | 示例 |
|---|---|---|---|
| 基础动作层 | `decider/logic/sub_statemachines/` | 单个基础动作 | `find_ball`、`chase_ball`、`dribble`、`kick`、`go_back_to_field` |
| 战术层 | `decider/logic/strategy_statemachines/` | 多动作组合 | `attack`、`defend_ball`、`dribble_ball`、`shoot_ball` |
| 角色策略层 | `decider/logic/policy_statemachines/` | 比赛角色 | `goalkeeper` 等 |

### 3. 3v3 角色分工

在仿真模式下，`game(agent)` 按机器人 ID 分配角色：

| Robot ID | 角色 | 策略 |
|---|---|---|
| `0` | Attacker | 找球后执行 `ContinuousPushController`，负责推球进攻 |
| `1` | Support | 站到球与己方球门连线后方约 1.2m，避免干扰进攻机器人 |
| `2` | Defender | 在己方半场锚定防守位置，并跟随球的横向位置 |
| 其他 ID | Fallback Attacker | 默认按进攻机器人处理 |

### 4. ContinuousPushController 连续推球控制

`ContinuousPushController` 的目标不是简单追球，而是让机器人：

1. 接近球；
2. 移动到球的后方；
3. 面向对方球门；
4. 沿球门方向持续推球；
5. 靠近边线时自动向场地中心修正。

核心误差项：

| 误差项 | 含义 | 用途 |
|---|---|---|
| `behind_depth` | 机器人是否位于球后方 | 控制前后距离，保证能把球向球门方向推 |
| `lateral_err` | 机器人相对球-门连线的横向偏差 | 控制左右修正，减少斜推 |
| `yaw_err` | 机器人朝向与球门方向的偏差 | 控制转向，保证朝向正确 |
| `ball_dist` | 机器人到球距离 | 在接近模式和推球模式之间连续加权 |
| `sideline_risk` | 靠近边线的风险 | 增加向场地中心的修正，减少出界 |

### 5. 轨迹记录与诊断

项目支持比赛过程诊断：

- 录制比赛视频：`scripts/record_match.sh`
- 记录轨迹 CSV：`--record-trajectory`
- 分析轨迹：`decider/scripts/analyze_trajectory.py`
- 诊断失败原因：`decider/scripts/diagnose_trajectory.py`

轨迹记录包含机器人位姿、球位置、速度命令、FSM 状态、对齐模式和踢球条件，可用于判断失败发生在感知、站位、对齐还是推球阶段。

---

## 🚀 运行方式

### 1. 启动 Sim Manager

```bash
conda run -n motrixsim0508 python simulation/motrixsim/sim_manager.py --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

### 2. 启动仿真

```bash
# 1v1 实时仿真
./scripts/start_sim.sh

# 3v3 实时仿真
./scripts/start_sim.sh --team-size 3

# 指定 policy
./scripts/start_sim.sh --policy path/to/model.pt
```

### 3. 启动单个 Decider

```bash
./scripts/start_decider.sh
./scripts/start_decider.sh --color blue --id 0 --port 5556
```

### 4. 启动整队

```bash
./decider/scripts/start_team.sh
./decider/scripts/start_team.sh --red 3 --blue 2
./decider/scripts/start_team.sh --kill
```

### 5. 录制比赛视频和轨迹

```bash
# 录制 1v1，默认 60 秒
./scripts/record_match.sh

# 录制 3v3 demo，持续 120 秒
./scripts/record_match.sh --demo-3v3 --d 120

# 同时记录轨迹 CSV
./scripts/record_match.sh --trajectory
```

---

## 📁 仓库结构

```text
├── decider/                   # 决策模块
│   ├── user_entry.py          # 自定义策略主入口
│   ├── interfaces/            # 感知、动作、比赛控制接口
│   ├── logic/                 # 状态机与角色策略
│   └── scripts/               # 队伍启动、诊断脚本
├── simulation/                # 仿真模块
│   ├── motrixsim/             # MotrixSim 主仿真环境
│   ├── isaac_sim/             # Isaac Sim 历史实现
│   └── labbridge/             # WebView / Sim Manager
├── MotrixLab/                 # RL 训练子项目
├── models/k1/                 # 默认策略模型
├── docs/                      # 文档
├── tools/                     # 工具脚本
└── scripts/                   # 常用启动和录制脚本
```

---

## 👤 作者信息

- **GitHub**: [Logic-TARS](https://github.com/Logic-TARS)
