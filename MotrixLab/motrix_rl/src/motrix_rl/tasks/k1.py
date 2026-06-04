# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.rslrl.cfg import RslrlCfg
from motrix_rl.skrl.config import SkrlCfg


class skrl:
    @rlcfg("k1-flat-terrain-walk")
    @dataclass
    class K1WalkFlatSkrlPpo(SkrlCfg):
        def __post_init__(self):
            runner = self.runner
            runner.models.policy.hiddens = [256, 128, 64]
            runner.models.value.hiddens = [256, 128, 64]
            runner.agent.rollouts = 24
            runner.agent.learning_epochs = 5
            runner.agent.mini_batches = 4
            runner.agent.learning_rate = 1e-3
            runner.trainer.timesteps = 30000


class rslrl:
    @rlcfg("k1-flat-terrain-walk")
    @dataclass
    class K1WalkFlatRslrlPpo(RslrlCfg):
        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            self.num_envs = 4096
            runner.seed = 1
            runner.max_iterations = 10000
            runner.num_steps_per_env = 24
            runner.save_interval = 50
            runner.experiment_name = "k1_g1_style_walk"
            runner.actor.hidden_dims = [32]
            runner.critic.hidden_dims = [32]
            runner.actor.init_noise_std = 0.8

            algo.learning_rate = 1e-3
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 4
            algo.entropy_coef = 0.01
            algo.desired_kl = 0.01

    @rlcfg("k1-ball-navigate")
    @dataclass
    class K1BallNavigateRslrlPpo(RslrlCfg):
        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            self.num_envs = 256
            runner.seed = 42
            runner.max_iterations = 3000
            runner.num_steps_per_env = 24
            runner.save_interval = 50
            runner.experiment_name = "k1_ball_navigate"
            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]
            runner.actor.init_noise_std = 0.5

            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 3
            algo.entropy_coef = 0.001
            algo.desired_kl = 0.008

    @rlcfg("k1-point-navigate")
    @dataclass
    class K1PointNavigateRslrlPpo(RslrlCfg):
        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            self.num_envs = 256
            runner.seed = 42
            runner.max_iterations = 3000
            runner.num_steps_per_env = 24
            runner.save_interval = 50
            runner.experiment_name = "k1_point_navigate"
            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]
            runner.actor.init_noise_std = 0.5

            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 3
            algo.entropy_coef = 0.001
            algo.desired_kl = 0.008
