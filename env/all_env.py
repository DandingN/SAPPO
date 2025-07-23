#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/2 上午10:40
# @function: the script is used to do something.
# @version : V1
from env import MyNewEnv
import numpy as np


class AllEnv(MyNewEnv):
    def __init__(self, env_config):
        super(AllEnv, self).__init__(env_config)

    def observe(self):
        x, y = self.positions
        # 剩余步数归一化
        remaining_steps_normalized = self.remaining_steps / self.max_episode_steps

        # 坐标归一化
        pos_x_normalized = x / (self.size - 1)
        pos_y_normalized = y / (self.size - 1)

        observation = np.array(
            self.grid.flatten().tolist() + [remaining_steps_normalized, pos_x_normalized, pos_y_normalized],
            dtype=np.float32
        )
        return observation
