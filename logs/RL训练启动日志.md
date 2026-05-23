# RL 训练启动日志

日期：2026-05-23

## 目标

启动强化学习训练链路，优先验证底层 locomotion RL，而不是直接训练比赛策略。

本次重点：

1. 创建独立 conda 环境；
2. 使用国内镜像安装训练依赖；
3. 跑通 MotrixLab 训练链路；
4. 接入 K1 flat terrain locomotion starter 环境；
5. 完成一次 K1 starter policy 训练。

## Conda 环境

新建环境：

```bash
conda create -y -n sim_soccer_rl python=3.10 pip
```

后续命令均使用：

```bash
PYTHONNOUSERSITE=1 /home/1ctnltug/miniconda3/envs/sim_soccer_rl/bin/python
```

使用 `PYTHONNOUSERSITE=1` 是为了避免误用用户目录下的 Python 包，确保依赖安装在 `sim_soccer_rl` 环境中。

## 国内镜像

普通 PyPI 包使用清华源：

```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

PyTorch CUDA 12.8 wheel 使用上海交大镜像：

```text
https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
```

## 已安装核心依赖

验证结果：

```text
torch 2.7.0+cu128
torchvision 0.22.0+cu128
torchaudio 2.7.0+cu128
gymnasium 1.1.1
motrixsim 0.8.0
motrix_envs 0.3.0
motrix_rl 0.3.0
rsl_rl 4.0.1
skrl 1.4.3
```

CUDA 验证：

```text
cuda_available True
cuda_device NVIDIA RTX5880-Ada-24Q
```

说明：

- `rsl-rl-lib 5.3.0` 与当前 MotrixLab 配置不兼容，会报 `MLPModel.__init__() got an unexpected keyword argument 'stochastic'`；
- 已固定为 `rsl-rl-lib 4.0.1`，RSLRL 训练可正常启动。

## 训练链路验证

先用 `cartpole` 验证 RSLRL 训练链路：

```bash
PYTHONNOUSERSITE=1 /home/1ctnltug/miniconda3/envs/sim_soccer_rl/bin/python scripts/train.py \
  --env cartpole \
  --rllib rslrl \
  --train-backend torch
```

结果：

- 训练完整跑完 300 iteration；
- 最终 mean reward 达到 `1000.00`；
- checkpoint 输出目录：

```text
simulation/MotrixLab-main/runs/cartpole/rslrl/26-05-23_15-29-14-_323960_PPO/
```

## K1 训练环境接入

新增 K1 训练 scene：

```text
legged_gym/resources/robots/K1/k1_train_scene.xml
```

原因：

- 原有 `legged_gym/resources/robots/K1/scene.xml` include 路径指向不存在的 `/opt/booster_gym/...`；
- `K1_locomotion.xml` 本体可以被 `motrixsim 0.8.0` 正常加载；
- 新 scene 直接 include `K1_locomotion.xml`，并补充训练需要的 `local_linvel` 和 `gyro` sensor。

新增 MotrixLab 环境：

```text
simulation/MotrixLab-main/motrix_envs/src/motrix_envs/locomotion/k1/
```

新增 RSLRL/SKRL 任务配置：

```text
simulation/MotrixLab-main/motrix_rl/src/motrix_rl/tasks/k1.py
```

注册任务名：

```text
k1-flat-terrain-walk
```

环境规格：

```text
observation_dim = 48
action_dim = 12
num_envs = 256
max_iterations = 200
```

当前 starter reward 包括：

- 速度跟踪；
- yaw rate 跟踪；
- base 姿态约束；
- 竖向速度惩罚；
- 横滚/俯仰角速度惩罚；
- torque 惩罚；
- action rate 惩罚；
- 关节默认位姿正则；
- 低高度或过大倾斜 termination。

## K1 Starter 训练

训练命令：

```bash
PYTHONNOUSERSITE=1 /home/1ctnltug/miniconda3/envs/sim_soccer_rl/bin/python scripts/train.py \
  --env k1-flat-terrain-walk \
  --rllib rslrl \
  --train-backend torch
```

训练结果：

- 完整跑完 200 iteration；
- 训练环境数：`256`；
- 总步数：`1,228,800`；
- 训练时间：约 `3分57秒`；
- mean reward 从约 `0.06` 提升到末尾约 `2.76`；
- mean episode length 从约 `18` 提升到末尾约 `47`；
- 未出现 NaN 或训练崩溃。

checkpoint 输出目录：

```text
simulation/MotrixLab-main/runs/k1-flat-terrain-walk/rslrl/26-05-23_15-38-19-_206905_PPO/
```

最终 checkpoint：

```text
simulation/MotrixLab-main/runs/k1-flat-terrain-walk/rslrl/26-05-23_15-38-19-_206905_PPO/model_199.pt
```

## 当前结论

K1 locomotion RL 的训练闭环已经跑通：

```text
K1 XML -> MotrixLab env -> RSLRL PPO -> checkpoint
```

但 `model_199.pt` 只是 starter policy，不能直接替换比赛中的 `model_4700.pt`。它的价值是证明环境、依赖、训练入口和 checkpoint 输出都已经可用。

## 下一步

1. 增加 play/eval 脚本，加载 `model_199.pt` 可视化 K1 行走表现；
2. 根据可视化结果调整 reward 和 termination；
3. 增加足端接触信息和 gait/air-time 类奖励；
4. 扩大训练规模和训练轮数；
5. 导出可被现有比赛系统加载的 TorchScript/ONNX policy。
