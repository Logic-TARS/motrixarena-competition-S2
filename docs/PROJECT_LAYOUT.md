# Project Layout

这个仓库当前按“运行策略 + 仿真 + 训练 + 机器人资源”来理解会比较清楚。

## 保留的主要目录

- `decider/`
  - 足球策略、决策逻辑和接口层。
  - 体积很小，属于上层控制代码。

- `simulation/motrixsim/`
  - 当前主要仿真运行目录。
  - 包含仿真入口、运行配置、机器人/场地资产和 policy 加载逻辑。

- `MotrixLab/`
  - 强化学习训练代码。
  - K1 训练环境在 `motrix_envs/src/motrix_envs/locomotion/k1/`。

- `simulation/labbridge/`
  - 仿真桥接/管理相关代码。
  - README 中仍有引用，暂时保留。

- `simulation/isaac_sim/`
  - 旧 Isaac Sim 相关代码。
  - 不是当前主路径，但文档中仍作为历史/兼容模块出现，暂时保留。

- `legged_gym/`
  - 机器人资源、旧 policy 和一些兼容脚本。
  - 训练配置中有硬编码路径会引用 `legged_gym/resources/robots/K1/k1_train_scene.xml`，运行配置也会引用 `legged_gym/policy/booster_k1/model_4700.onnx`，所以不要随便移动或删除。

- `docs/`
  - 根目录原有的启动、训练、比赛和整理说明文档。
  - 这些文件不参与运行，归档在这里方便根目录保持清爽。

- `tools/`
  - 一次性维护/补丁脚本。
  - 当前包含 `_patch_gait.py`，不是常规运行入口。

## 根目录模型文件

- `model_20000_new.onnx`
- `model_4700.pt`

这两个是当前运行配置里的默认 policy/model 路径之一，保留。

## 已清理的内容

- `docx/`
  - 微信文章离线清理工具/素材，和仿真训练主流程没有引用关系。

- `envs/`、`examples/`
  - 早期 MotrixArena demo，未被当前 README、训练入口或 motrixsim 主路径引用。

- `simulation/run.py`
  - 旧入口，导入路径已经不匹配，并且文件末尾有异常字符。

- `MotrixLab/exported/`
  - 空目录。

- `decider/requirements (copy).txt`
  - 和 `decider/requirements.txt` 完全重复。

- `simulation/motrixsim/sim2sim_runner (copy).py`
  - 早期模拟 runner，不是当前正式入口，末尾也有异常字符。

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
