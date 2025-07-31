__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Union

import numpy as np

from gymnasium import utils
from envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 0.1,
    "elevation": 10,
    # "azimuth": 135,
}

# 推杆起始位姿 <key qpos='0 0 0 0 0 -0.84318 0 0 1.8875 1.50816 -0.03927 -0.86405'/>
# push_init_pos = [0,0,0,0,0,-0.84318,0,0,1.8875,1.50816,-0.03927,-0.86405]

class PusherEnv(MujocoEnv, utils.EzPickle):
    r"""
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "CyberMice_head-fixed-250627.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 0.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_near_weight,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )
        self._reward_near_weight = reward_near_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(63,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        # print(action)
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    # def _get_rew(self, action):
    #     # 1) 用 joint_name2id 拿索引
    #     axle_idx = self.model.joint("axle_y").id
    #     axle_angle = float(self.data.qpos[axle_idx])

    #     # 2) 定义目标角度和误差
    #     target = 0.75
    #     err = abs(axle_angle - target)

    #     # 3) 设计 reward：离目标越远惩罚越大
    #     reward_angle = -err * 10.0      # 样例权重 10，可调
    #     reward_ctrl  = -np.sum(np.square(action)) * self._reward_control_weight

    #     reward = reward_angle + reward_ctrl + 5.0
    #     info = {
    #         "axle_angle": axle_angle,
    #         "reward_angle": reward_angle,
    #         "reward_ctrl": reward_ctrl,
    #     }
    #     return reward, info
    
    def _get_rew(self, action):
        # 1) 当前关节角
        axle_idx = self.model.joint("axle_y").id
        axle_angle = float(self.data.qpos[axle_idx])

        # 2) 目标角度
        target = 0.75

        # 3) 计算本步距离变化（正值表示朝目标靠近）
        if not hasattr(self, "_prev_axle_angle"):
            # 第一步，初始化 prev
            self._prev_axle_angle = axle_angle
        # 变化量（绝对距离减少的量）
        prev_err = abs(self._prev_axle_angle - target)
        curr_err = abs(axle_angle - target)
        delta = max(prev_err - curr_err, 0.0)
        self._prev_axle_angle = axle_angle

        # 4) 设计全正向 reward：
        #    - 按距离减少量奖励（比例放大）
        #    - 到达目标一次性大奖励
        reward = 0.0
        reward += 100.0 * delta           # 每减少 1 radian 得 100 分
        if curr_err < 0.01:                # 距离足够近
            reward += 1000.0               # 一次性“完成”大奖励
        # （不再包含任何负向惩罚项）

        # 5) info 里仍保存角度、delta 供监控
        info = {
            "axle_angle": axle_angle,
            "delta_error": delta,
        }
        return reward, info

    def reset_model(self):
        # qpos = push_init_pos + self.np_random.uniform(
        #     low=-0.005, high=0.005, size=self.model.nv
        # )

        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:],
                self.data.qvel.flatten()[:],
                self.get_body_com("RCarpi"),
                self.get_body_com("RFPhalange3_3"),
                self.get_body_com("Axle"),
            ]
        )
