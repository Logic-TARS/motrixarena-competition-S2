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
            runner.obs_groups = {"actor": ["policy"], "critic": ["privileged"]}
            runner.actor.class_name = "MLPModel"
            runner.actor.hidden_dims = [512, 256, 128]
            runner.actor.noise_std_type = "scalar"
            runner.actor.state_dependent_std = False
            runner.critic.class_name = "MLPModel"
            runner.critic.hidden_dims = [512, 256, 128]
            runner.actor.init_noise_std = 0.3

            algo.learning_rate = 1e-3
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 4
            algo.entropy_coef = 0.0005
            algo.desired_kl = 0.01

    @rlcfg("k1-flat-terrain-walk-speed")
    @dataclass
    class K1WalkSpeedRslrlPpo(RslrlCfg):
        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            self.num_envs = 4096
            runner.seed = 1
            runner.max_iterations = 10000
            runner.num_steps_per_env = 24
            runner.save_interval = 50
            runner.experiment_name = "k1_speed_recovery"
            runner.obs_groups = {"actor": ["policy"], "critic": ["privileged"]}
            runner.actor.class_name = "MLPModel"
            runner.actor.hidden_dims = [512, 256, 128]
            runner.actor.noise_std_type = "scalar"
            runner.actor.state_dependent_std = False
            runner.critic.class_name = "MLPModel"
            runner.critic.hidden_dims = [512, 256, 128]
            runner.actor.init_noise_std = 0.3

            algo.learning_rate = 1e-3
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 4
            algo.entropy_coef = 0.0005
            algo.desired_kl = 0.01

    @rlcfg("k1-getup")
    @dataclass
    class K1GetupRslrlPpo(RslrlCfg):
        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            self.num_envs = 2048
            runner.seed = 1
            runner.max_iterations = 10000
            runner.num_steps_per_env = 24
            runner.save_interval = 50
            runner.experiment_name = "k1_full_body_getup"
            runner.obs_groups = {"actor": ["policy"], "critic": ["privileged"]}
            runner.actor.class_name = "MLPModel"
            runner.actor.hidden_dims = [512, 256, 128]
            runner.actor.noise_std_type = "scalar"
            runner.actor.state_dependent_std = False
            runner.actor.init_noise_std = 0.6
            runner.critic.class_name = "MLPModel"
            runner.critic.hidden_dims = [512, 256, 128]

            algo.learning_rate = 5.0e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 4
            algo.entropy_coef = 0.001
            algo.desired_kl = 0.01
