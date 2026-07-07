# Project Layout

这个仓库按“比赛运行代码 + 仿真环境 + 训练环境 + 参考工程”来理解会比较清楚。根目录尽量只保留入口文件、运行配置、默认模型和一级功能目录。

## 主要运行目录

- `decider/`
  - 当前主要足球策略、决策逻辑和接口层。
  - `tests/` 里的轨迹诊断测试会直接导入这里的脚本。

- `simulation/motrixsim/`
  - 当前主要仿真运行目录。
  - 包含仿真入口、运行配置、机器人/场地资产和 policy 加载逻辑。

- `scripts/`
  - 根目录级启动脚本，例如启动仿真、决策器、比赛录制等。

- `tests/`
  - 仓库级测试和 fixtures。

- `tools/`
  - 一次性维护、诊断和校准脚本。
  - 这些不是常规启动入口，但对调参和排查很有用。

## 训练与资源目录

- `MotrixLab/`
  - 强化学习训练代码。
  - K1 训练环境在 `MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/`。
  - `MotrixLab/.venv/`、`MotrixLab/runs/` 是本地产物，不应提交；需要释放空间时可以删除并重建/重跑。

- `legged_gym/`
  - 机器人资源、旧 policy 和兼容脚本。
  - 训练配置中有硬编码路径会引用 `legged_gym/resources/robots/K1/k1_train_scene.xml`，运行配置也会引用 `legged_gym/policy/booster_k1/model_4700.onnx`，所以不要随便移动或删除。

- `simulation/labbridge/`
  - 仿真桥接/管理相关代码。

- `simulation/isaac_sim/`
  - 旧 Isaac Sim 相关代码。
  - 不是当前主路径，但文档中仍作为历史/兼容模块出现，暂时保留。

## 参考工程

- `booster_train/`
  - 外部/参考训练工程副本，目录内带独立 `.git/`。
  - 当前根 `.gitignore` 会忽略这个目录的新变动，避免误提交参考工程杂项。

- `robocup_demo/`
  - 外部/参考 RoboCup demo 工程副本，目录内带独立 `.git/`。
  - 当前根 `.gitignore` 会忽略这个目录的新变动，避免误提交参考工程杂项。

## 文档目录

- `models/k1/`
  - 默认 K1 policy/model 文件。
  - 当前运行配置会优先查找 `models/k1/model_20000_new.onnx` 和 `models/k1/model_4700.pt`。

- `docs/`
  - 启动、训练、比赛、Docker 和整理说明文档。

- `README.md`
  - 仓库主说明，保留最常用入口和快速启动信息。

## 模型文件

- `models/k1/model_20000_new.onnx`
- `models/k1/model_4700.pt`

这两个仍是当前运行配置里的默认 policy/model 路径之一。移动它们需要同步修改 `simulation/motrixsim/app/runtime_config.py`、README 和启动文档。

## 本地产物清理建议

已清理/可安全再生成：

- `__pycache__/`
- `.ruff_cache/`
- `.pytest_cache/`
- `.mypy_cache/`
- `*.pyc`

谨慎清理：

- `MotrixLab/.venv/`
  - 虚拟环境，删除后需要重新安装依赖。

- `MotrixLab/runs/`、`MotrixLab/top_runs/`
  - 训练 checkpoint 和 TensorBoard 输出，删除前先确认不再需要。

- 嵌套 `.git/`
  - `booster_train/.git/`、`robocup_demo/.git/` 属于参考工程自己的版本库元数据；如果只是节省空间，可以压缩或移出参考工程，但不要在不确认用途时直接删除。

## 常用入口

训练 K1：

```bash
cd /opt/sim_soccer2_walk_0527/MotrixLab
conda run -n sim_soccer_rl bash scripts/train_k1.sh
```

K1 环境 smoke test：

```bash
cd /opt/sim_soccer2_walk_0527/MotrixLab
conda run -n sim_soccer_rl env PYTHONPATH=/opt/sim_soccer2_walk_0527/MotrixLab/motrix_envs/src:/opt/sim_soccer2_walk_0527/MotrixLab/motrix_rl/src python scripts/smoke_k1_env.py --num-envs 4 --steps 16 --zero-action
```
