# MotrixArena S2 3v3 机器人足球项目

[English](README.md)

本项目是 MotrixArena S2 机器人足球仿真比赛的多机器人决策、控制、回放与诊断工程。系统面向 K1 / Pi Plus 足球机器人，基于 MotrixSim、Decider 状态机和 ZMQ 仿真通信，完成多机器人策略运行与比赛过程复盘。

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Decision](https://img.shields.io/badge/Decider-State%20Machine-FF6F00)
![Communication](https://img.shields.io/badge/Communication-ZMQ-555555)
![Robot](https://img.shields.io/badge/Robot-K1%20%7C%20Pi%20Plus-4B8BBE)

## Demo / 回放

仓库包含本地 demo 与诊断资产，用于展示仿真、轨迹记录和 locomotion 分析链路。

| 资产 | 链接 | 用途 |
| --- | --- | --- |
| 3v3 demo 回放 | [demo-match-20260614-105108.mp4](docs/assets/demo/demo-match-20260614-105108.mp4) | 本地录制的 3v3 仿真流程短片段 |
| 轨迹时间序列 | [demo-trajectory-timeseries.png](docs/assets/demo/demo-trajectory-timeseries.png) | 机器人、球、命令和诊断量随时间变化 |
| Locomotion 速度跟踪 | [loco-v030-velocity-tracking.png](docs/assets/demo/loco-v030-velocity-tracking.png) | `T1_forward_velocity/v030` 的速度跟踪诊断图 |

Locomotion baseline `20260614_135745_default` 共记录 3 个 case，结果为 `0 pass / 3 fail`，acceptance rating 为 `Not recommended`。这些失败样例保留为步态跟踪和稳定性诊断材料，不代表官方比赛成绩。

## 项目亮点

- 比赛成绩：MotrixArena S2 优胜奖，排名第 10，进球率高于 70%。
- 实现 3v3 仿真决策策略，按机器人 ID 分配 attacker、support、defender 角色。
- 以 `decider/user_entry.py` 为主策略入口，结合 `decider/logic/` 中的状态机组件组织行为。
- 构建 `ContinuousPushController`，将单纯追球转化为站到球后方、对齐球门并持续推球。
- 通过 ZMQ 连接仿真和决策进程，支持多机器人独立端口与独立 ID 运行。
- 建立可复现的诊断链路，支持比赛视频录制、轨迹 CSV、轨迹图和失败原因分析。

## 结果

| 项目 | 内容 |
| --- | --- |
| 比赛 | MotrixArena S2 机器人足球仿真比赛 |
| 结果 | 优胜奖 |
| 排名 | 第 10 名 |
| 进球率 | 高于 70% |
| 主要机器人 | K1 / Pi Plus |
| 决策框架 | Decider 状态机 |
| 仿真平台 | MotrixSim，兼容 Isaac Sim 历史实现 |
| 通信方式 | ZMQ，多机器人独立端口通信 |
| 主要进攻策略 | `ContinuousPushController` |

## 比赛任务

MotrixArena S2 聚焦仿真机器人足球。多台机器人需要在场地内完成找球、接近、站到球后、向对方球门推球或射门，并保持防守站位，同时避免队友之间互相干扰。

核心工程难点：

| 难点 | 方法 |
| --- | --- |
| 多机器人容易同时冲向球并扎堆 | 在仿真中按机器人 ID 分配 attacker、support、defender 角色 |
| 单纯追球容易把球推偏或推出边线 | 跟踪球-门几何关系，让进攻机器人站到球后方再推球 |
| 追球、对齐和推球之间硬切换会产生抖动 | 使用连续误差空间控制、速度限幅、soft clipping 和近球角速度阻尼 |
| 比赛失败仅凭肉眼难以定位 | 记录视频、轨迹 CSV、控制器状态、对齐指标和分析图 |

## 系统架构

```text
motrixarena-competition-S2/
├── decider/                   # 决策模块
│   ├── user_entry.py          # 自定义策略入口
│   ├── decider.py             # Decider 运行时
│   ├── config.yaml            # 决策参数配置
│   ├── interfaces/            # Action / Vision / GameController / SimClient
│   ├── logic/                 # 状态机与策略逻辑
│   └── scripts/               # 队伍启动和诊断脚本
├── simulation/                # 仿真模块
│   ├── motrixsim/             # MotrixSim 运行环境
│   ├── isaac_sim/             # Isaac Sim 历史实现
│   └── labbridge/             # WebView / Bridge / Sim Manager
├── MotrixLab/                 # K1 locomotion / RL 子项目
├── models/k1/                 # 默认 K1 policy 模型
├── docs/                      # 文档和 demo 资产
├── tools/                     # 维护工具
└── scripts/                   # 常用启动与录制脚本
```

Decider 负责策略逻辑，Simulation 负责物理仿真、可视化和运行时管理，两者通过 ZMQ 通信。

## 策略设计

### 状态机层级

| 层级 | 目录 | 职责 | 示例 |
| --- | --- | --- | --- |
| 基础动作层 | `decider/logic/sub_statemachines/` | 单机器人基础行为 | `find_ball`、`chase_ball`、`dribble`、`kick`、`go_back_to_field` |
| 战术行为层 | `decider/logic/strategy_statemachines/` | 多动作策略组合 | `attack`、`defend_ball`、`dribble_ball`、`shoot_ball` |
| 角色策略层 | `decider/logic/policy_statemachines/` | 比赛级角色 | `goalkeeper` |

### 3v3 角色分工

在仿真模式下，`decider/user_entry.py` 中的 `game(agent)` 按机器人 ID 分配行为：

| Robot ID | 角色 | 行为 |
| --- | --- | --- |
| `0` | Attacker | 找球后运行 `ContinuousPushController`，向球门方向推球 |
| `1` | Support | 站在球与己方球门连线后方约 1.2 m，避免干扰进攻机器人 |
| `2` | Defender | 锚定己方半场防守位置，并跟随球的横向位置 |
| 其他 ID | Fallback attacker | 使用进攻机器人路径 |

### ContinuousPushController

`ContinuousPushController` 是主要进攻策略。它不是只驱动机器人冲向球，而是持续估计：

| 信号 | 含义 | 用途 |
| --- | --- | --- |
| `behind_depth` | 机器人是否位于球相对对方球门的后方 | 控制前后位置，保证可有效推球 |
| `lateral_err` | 机器人相对球-门连线的横向偏差 | 减少斜推或侧向推球 |
| `yaw_err` | 机器人朝向与球门方向的误差 | 让机器人朝向球门 |
| `ball_dist` | 机器人到球的距离 | 在接近和推球之间连续过渡 |
| `sideline_risk` | 靠近边线的风险 | 增加向场地中心的修正，降低出界概率 |

控制器会记录当前对齐模式和诊断量，轨迹分析可以据此判断失败来自感知、站位、朝向对齐、边线修正还是底层 locomotion。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Logic-TARS/motrixarena-competition-S2.git
cd motrixarena-competition-S2
```

### 2. 安装依赖

辅助脚本使用 `uv` 执行 Python。完整仿真栈还依赖项目对应的 MotrixSim / MotrixLab 环境和资产。

```bash
python3 -m pip install --user uv
pip install -r requirements.txt
```

容器化配置见 [compose.yaml](compose.yaml)。

### 3. 启动仿真

```bash
# 1v1 实时仿真
./scripts/start_sim.sh

# 3v3 实时仿真
./scripts/start_sim.sh --team-size 3

# 指定 policy
./scripts/start_sim.sh --policy models/k1/model_4700.pt
```

### 4. 启动 Decider

```bash
./scripts/start_decider.sh
./scripts/start_decider.sh --color blue --id 0 --port 5556
```

### 5. 启动整队

```bash
./decider/scripts/start_team.sh
./decider/scripts/start_team.sh --red 3 --blue 2
./decider/scripts/start_team.sh --kill
```

### 6. 录制比赛

```bash
# 录制 1v1，默认 60 秒
./scripts/record_match.sh

# 录制 3v3 demo，持续 120 秒
./scripts/record_match.sh --demo-3v3 --d 120

# 同时记录视频和轨迹 CSV
./scripts/record_match.sh --trajectory
```

## 诊断链路

项目将诊断能力放在仿真循环附近：

| 工具 | 用途 |
| --- | --- |
| `scripts/record_match.sh` | 启动仿真和 decider 进程，记录帧并编码比赛视频 |
| `--record-trajectory` / `--trajectory` | 为运行过程启用轨迹 CSV 记录 |
| `decider/scripts/analyze_trajectory.py` | 生成轨迹摘要和图表 |
| `decider/scripts/diagnose_trajectory.py` | 诊断推球和对齐相关失败模式 |
| `decider/scripts/analyze_loco_baseline.py` | 汇总 locomotion baseline 轨迹指标 |

轨迹行包含机器人位姿、球位置、速度命令、FSM 状态、对齐模式、踢球条件和控制器诊断量，使失败可以按帧复盘，而不只依赖视频观察。

## 仓库结构

```text
.
├── decider/                   # 决策运行时和策略逻辑
│   ├── user_entry.py          # 主自定义策略入口
│   ├── interfaces/            # 感知、动作、比赛控制和仿真接口
│   ├── logic/                 # 状态机与角色策略
│   └── scripts/               # 队伍启动和诊断脚本
├── simulation/                # MotrixSim / 历史 Isaac Sim 集成
├── MotrixLab/                 # K1 locomotion 与 policy 执行子项目
├── models/k1/                 # 默认 K1 policy 模型
├── docs/assets/demo/          # Demo 回放和诊断图
├── scripts/                   # 仿真、决策和录制辅助脚本
├── requirements.txt
├── compose.yaml
├── LICENSE
└── COPYRIGHT
```

## License

本仓库作为 MotrixArena S2 比赛的作品集展示与复现实验记录。上游仿真资产、机器人模型和相关框架组件归其原维护者所有。

许可证和归属信息见 [LICENSE](LICENSE) 与 [COPYRIGHT](COPYRIGHT)。
