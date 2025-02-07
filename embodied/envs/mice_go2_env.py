import torch
import genesis as gs
import numpy as np
import embodied
from gymnasium import spaces


class Go2Env(embodied.Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self._initialize_environment(num_envs, show_viewer)

        self._done = torch.ones((num_envs,), device=self.device, dtype=torch.bool)
        self._episode_reward = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self._episode_length = torch.zeros((num_envs,), device=self.device, dtype=torch.int32)

        # 添加 observation_space 和 action_space 属性
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(self.num_obs,), dtype=np.float32),
            "reward": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "is_first": spaces.Discrete(2),
            "is_last": spaces.Discrete(2),
            "is_terminal": spaces.Discrete(2),
        })

        self.action_space = spaces.Dict({
            "action": spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32),
            "reset": spaces.Discrete(2),
        })

    @property
    def obs_space(self):
        """定义环境的观察空间"""
        return {
            'image': embodied.Space(np.float32, (self.num_envs, self.num_obs)),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
    def act_space(self):
        """定义环境的动作空间"""
        return {
            'action': embodied.Space(np.float32, (self.num_envs, self.num_actions)),
            'reset': embodied.Space(bool),
        }

    def _initialize_environment(self, num_envs, show_viewer):
        self.dt = 0.02
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )
        self.scene.build(n_envs=num_envs)

        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.env_cfg["num_actions"], self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.env_cfg["num_actions"], self.motor_dofs)

    def reset(self):
        self._done[:] = False
        self._episode_reward[:] = 0.0
        self._episode_length[:] = 0

        # Reset all environments
        obs = self._reset_envs(torch.arange(self.num_envs, device=self.device))
        return obs

    def _reset_envs(self, env_ids):
        self.robot.set_dofs_position(
            position=torch.zeros((len(env_ids), self.env_cfg["num_actions"]), device=self.device),
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(env_ids)

        obs = torch.zeros((self.num_envs, self.obs_cfg['num_obs']), device=self.device)
        return {'obs': obs, 'reward': 0.0, 'is_first': True, 'is_last': False, 'is_terminal': False}

    def step(self, action):
        """与Dreamer兼容的step方法"""
        actions = action['action']
        reset = action['reset']

        # 执行动作和环境步进
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        self.robot.control_dofs_position(self.actions, self.motor_dofs)  # 控制机器人
        self.scene.step()  # 进行一步模拟

        # 获取状态、奖励和终止信息
        reward = self.compute_reward()
        done = self.check_done()

        obs = torch.zeros((self.num_envs, self.obs_cfg['num_obs']), device=self.device) 
        # 假设 obs 是原始的整数型数据 
        normalized_image = obs / 255.0 # 将整数值规范化到 0.0 到 1.0

        # 返回包含奖励和状态的字典
        return {
            'image': normalized_image,
            'reward': reward,
            'is_first': self._episode_length == 0,
            'is_last': done,
            'is_terminal': done,
        }

    def _apply_action(self, actions):
        clipped_actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        target_dof_pos = clipped_actions * self.env_cfg["action_scale"]
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

    def _collect_observations(self):
        obs = torch.zeros((self.num_envs, self.obs_cfg['num_obs']), device=self.device)
        reward = self._compute_reward()
        done = reward < self.reward_cfg["done_threshold"]  # Example done condition
        info = {}

        return obs, reward, done, info

    def _compute_reward(self):
        reward = torch.zeros((self.num_envs,), device=self.device)

        # 计算追踪奖励：目标位置到机器人当前位置的距离
        target_position = torch.tensor(self.env_cfg["target_pos"], device=self.device)
        robot_position = self.robot.get_global_positions()
        dist = torch.norm(robot_position - target_position, dim=1)
        reward += self.reward_cfg["distance_weight"] * (1.0 - dist)

        # 计算机器人掉落的惩罚
        base_height = self.robot.get_global_positions()[:, 2]
        reward -= self.reward_cfg["fall_penalty"] * (base_height < self.reward_cfg["fall_threshold"])

        self._episode_reward += reward
        return reward

    def check_done(self):
        # 终止条件：超出最大步数或掉落
        return self._episode_length >= self.env_cfg["max_episode_length"] or self._episode_reward < self.reward_cfg["fall_penalty"]

    def render(self):
        return self.scene.render()
