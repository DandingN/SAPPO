#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/4/30 下午4:59
# @function: the script is used to do something.
# @version : V1
from types import SimpleNamespace
import numpy as np
from stable_baselines3.common.env_checker import check_env
from env import MyNewEnv


def get_object():
    return np.load("data/object/object.npy")


env_config = SimpleNamespace(**{
    "env_id": "new_env_id",
    "start_pos": [0, 0],
    "grid_size": 10,
    "observation_size": 3,
    "max_episode_steps": 31,
    "repeat_punish": 0,
    "zero_punish": 0,
    "object_percent": 90,
    "object_list": get_object(),
    "render": False,
    "render_mode": "rgb_array",
    "test_mode": False,
    "model_dir": "./models/ppo/"
})

env = MyNewEnv(env_config)
check_env(env)

