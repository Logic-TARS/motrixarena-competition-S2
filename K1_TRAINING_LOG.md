# K1 训练配置修复与优化日志

日期: 2026-05-23

## 审查发现的问题 (5项)

1. **观测归一化 bug** — `walk_np.py:132` 陀螺仪数据用了 `lin_vel` 归一化而非 `ang_vel`
2. **dof_acc 奖励权重过小** — `cfg.py:114` 权重 `-2.5e-7` 实际贡献为零
3. **stand_still 与 joint_regularization 冲突** — 同一目标用 L1/L2 不同范数，梯度不一致
4. **缺少足部相关奖励** — 无 feet_air_time / collision 奖励，无法引导行走步态
5. **base_height 惩罚过重** — `-5.0` 可能压制探索

## 修改的文件

### 1. `MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/cfg.py`

- Asset 配置扩展：新增 `foot_name="Foot"`, `ground_name="ground"`, `penalize_contacts_on=["Trunk","Shank"]`
- RewardConfig.scales 变更：
  - `base_height`: -5.0 → -2.0
  - `dof_acc`: -2.5e-7 → -2.5e-6
  - 移除 `stand_still`
  - 新增 `feet_air_time`: 1.0
  - 新增 `collision`: -1.0
- RewardConfig 新增字段：`max_foot_height=0.15`, `target_base_height=0.68`, `gait_frequency=1.5`

### 2. `MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/walk_np.py`

- 修复观测归一化：陀螺仪使用 `normalization.ang_vel` (line 184)
- 观测空间 47→52 维，新增：
  - `[3:6]` local_linvel (局部线速度 x3)
  - `[50]` left_contact (左脚触地)
  - `[51]` right_contact (右脚触地)
- `_init_buffer`: 构建足部/碰撞接触对，按左右脚分离
- `_get_obs`: 重组观测布局，加入 local_linvel + 足部接触
- `update_observation`: 新增接触检测，左右脚分离追踪
- `update_reward`: termination penalty 单独处理，clip 下限 None (允许负奖励)
- `reset`: 新增 `gait_phase`, `feet_air_time`, `contacts`, `left_contact`, `right_contact` 字段
- 移除 `_reward_stand_still`
- 新增 `_update_feet_air_time`, `_reward_feet_air_time`, `_reward_collision`

最终观测布局 (52维):
```
0:3    local_gravity       重力方向
3:6    local_linvel        局部线速度 (新增)
6:9    gyro                角速度
9:12   commands            速度指令
12     cos(gait_phase)     步态相位 cos
13     sin(gait_phase)     步态相位 sin
14:26  diff_dof_pos        关节位置偏移
26:38  dof_vel             关节速度
38:50  prev_actions        上一步动作
50     left_contact        左脚触地 (新增)
51     right_contact       右脚触地 (新增)
```

最终奖励函数 (13项):
```
tracking_lin_vel     1.0    追踪线速度 (奖励)
tracking_ang_vel     0.5    追踪角速度 (奖励)
feet_air_time        1.0    足部离地时间 (奖励，新增)
lin_vel_z           -2.0    垂直速度惩罚
ang_vel_xy          -0.05   横滚/俯仰角速度惩罚
orientation         -1.0    倾斜惩罚
base_height         -2.0    高度偏差惩罚 (降低)
torques             -0.0001 力矩惩罚
dof_vel             -0.001  关节速度惩罚
dof_acc             -2.5e-6 关节加速度惩罚 (提高10倍)
action_rate         -0.005  动作变化率惩罚
joint_regularization -0.05  默认姿态偏离惩罚
collision           -1.0    非足部触地惩罚 (新增)
termination         -0.5    终止惩罚
```

### 3. `MotrixLab/motrix_rl/src/motrix_rl/tasks/k1.py`

- RSLRL: `max_iterations`: 200 → 1500 (总步数 1.2M → 9.2M)

### 4. `MotrixLab/scripts/smoke_k1_env.py`

- 期望 obs_dim: 47 → 52

### 5. `MotrixLab/scripts/export_k1_rslrl_torchscript.py`

- 期望 obs_dim: 47 → 52

### 6. 新增 `/opt/sim_soccer2/train_k1.sh` 和 `/opt/sim_soccer2/train_k1.py`

- 训练启动脚本和 Python 启动器
- 绕过 absl 缓冲问题
- 使用 `conda activate sim_soccer_rl` 替代 `conda run`

## 训练命令

```bash
bash /opt/sim_soccer2/train_k1.sh
```

配置: 256 envs × 24 steps × 1500 iters = 9.2M 步, 约45分钟 (GPU)
