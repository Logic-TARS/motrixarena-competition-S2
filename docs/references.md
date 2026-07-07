找到了，而且有几个很值得参考。按和我们当前 K1 AMP 训练的相关性排序：

最值得看

BoosterRobotics/booster_train
https://github.com/BoosterRobotics/booster_train
这是最贴近的：官方 Booster 机器人 RL 任务，基于 Isaac Lab，README 明确说包含适配 Booster K1 的 BeyondMimic motion tracking 框架，并配套 rsl_rl/train.py、play/export 流程。很适合参考 K1 reward、motion tracking、训练任务组织。来源里写到它使用 Isaac Lab，并适配 Booster K1。(github.com)

BoosterRobotics/booster_assets
https://github.com/BoosterRobotics/booster_assets
这个对我们特别关键：里面有 Booster robot models、motion data，而且列出了 K1 的 22 关节顺序。这个顺序和我们现在 AMP 的顺序一致：头、手臂、左腿、右腿。可以用它校验 motion data、joint order、默认姿态。(github.com)

BoosterRobotics/booster_deploy
https://github.com/BoosterRobotics/booster_deploy
官方部署框架，目标是同一套 policy code 在仿真和真机上运行。我们现在改的 sim2sim 部署链路，可以参考它的 policy export、action mapping、deployment wrapper。(github.com)

BoosterRobotics/booster_gym
https://github.com/BoosterRobotics/booster_gym
官方 RL locomotion 框架。页面说现在提供支持 K1 的新 RL pipeline，并覆盖 training、playing、cross-simulation testing、deployment。它偏完整训练到部署框架，不一定直接搬代码，但很适合对照我们的训练/导出/部署闭环。(github.com)

足球方向
5. BoosterRobotics/robocup_demo
https://github.com/BoosterRobotics/robocup_demo
官方 RoboCup demo，支持 T1/K1，自主决策踢球和完整比赛，包含 vision、brain、game_controller。这个更偏上层比赛系统，不是底层 RL locomotion，但对后续 3v3 策略/决策很有用。(github.com)

bit-bots/SoccerDiffusion
https://github.com/bit-bots/SoccerDiffusion
用 RoboCup 真实比赛数据训练 humanoid soccer 控制，包括 walking、kicking、fall recovery。不是我们的 K1/RSLRL 路线，但对“用比赛轨迹/动作数据做 imitation 或 motion prior”很有参考价值。数据很大，页面标的是数百 GB 级。(bit-bots.github.io)
通用 humanoid locomotion
7. roboterax/humanoid-gym
https://github.com/roboterax/humanoid-gym
Isaac Gym + rsl_rl 风格，重点是 humanoid locomotion、zero-shot sim2real、sim2sim 到 MuJoCo。它的默认任务是多帧低层控制，和我们 375 obs -> 22 action 的思路很接近。(github.com)

rohanpsingh/LearningHumanoidWalking
https://github.com/rohanpsingh/LearningHumanoidWalking
MuJoCo + PPO 的 humanoid walking 工程，结构里有 envs/、tasks/、robots/、PD control、reward/termination。适合参考“人形机器人环境抽象”和 reward/termination 设计。(github.com)
我的建议：先重点扒 booster_train + booster_assets。尤其是 booster_assets 的 K1 motion CSV 和 joint order，可以直接帮助我们下一步做 imitation/AMP 或默认姿态校验；booster_train 里 K1 BeyondMimic 任务可以参考 reward、motion tracking、课程配置。