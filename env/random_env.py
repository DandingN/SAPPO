#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/2 下午8:15
# @function: the script is used to do something.
# @version : V1
import random
from env import MyNewEnv
import numpy as np


class RandomEnv(MyNewEnv):
    def __init__(self, env_config):
        super(RandomEnv, self).__init__(env_config)

    def reset(self, seed=None, options=None):
        observation = self.observe()
        info = {}
        self._current_step = 0

        # 单局超过目标阈值，则更换地图继续训练
        target = self.get_target()
        if not self.test:
            self.num = random.randint(1, 1000)
        self.threshold = 0
        self.grid = self.generate_grid(self.num)
        self.positions = self.start_pos  # 初始化智能体位置
        self.reset_color_calculator()
        return observation, info
