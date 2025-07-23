#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/9 上午10:35
# @function: the script is used to do something.
# @version : V1
import glob
import os

from env import MyNewEnv, AllEnv, RandomEnv

import numpy as np
from stable_baselines3.common.monitor import Monitor
from types import SimpleNamespace
from sb3_contrib import RecurrentPPO
from tqdm import tqdm
from utils import evaluate_policy


def get_path_from_actions(actions, start=(0, 0), size=10):
    """
    从给定的动作序列生成路径坐标。

    :param actions: 动作序列，列表形式，每个元素为 0~3 的整数
    :param start: 初始坐标，默认为 (0, 0)
    :param size: 网格大小（size × size 的方格），默认为 5
    :return: 路径坐标列表，例如 [(0,0), (0,1), ...]
    """
    x, y = start
    path = [[x, y]]

    for action in actions:
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(size - 1, y + 1)
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0~3.")

        path.append([x, y])

    return path


# noinspection PyShadowingNames
def test(env_num, model_name, obj, model):
    env = Monitor(MyNewEnv(SimpleNamespace(**{
        "env_id": "MyNewEnv",  # 环境名
        "num": env_num,
        "start_pos": [0, 0],  # 起始位置
        "grid_size": 10,  # 网格环境尺寸
        "observation_size": 5,  # 观察空间尺寸
        "max_episode_steps": 31,  # 回合最大步数
        "repeat_punish": 0,  # 重复惩罚
        "zero_punish": 0,  # 零值惩罚
        "object_percent": 0.9,  # 精确算法目标值比率
        "object_list": obj,  # 环境目标值池
        "render": False,  # 渲染选项
        "render_mode": "rgb_array",  # 渲染模式
        "test_mode": True,  # 测试模式
        "model_dir": "./models/ppo/"  # 模型位置
    })), f"log/test/{model_name}")

    episode_rewards, episode_length, episode_actions = evaluate_policy(model, env, n_eval_episodes=30, return_episode_rewards=True)
    # 获取最大奖励的索引
    best_index = np.argmax(episode_rewards)

    # 获取对应的动作序列
    best_actions = episode_actions[best_index]
    path = get_path_from_actions(best_actions, start=(0, 0), size=10)

    return max(episode_rewards)


def test_model(models):
    obj = np.load("data/object/object.npy")
    for m in tqdm(models, desc="Processing Models", total=len(models)):
        m_name = m["name"]
        model = RecurrentPPO.load(f"{m['path']}{m['name']}")
        results = []
        for i in tqdm(range(1, 2), desc=f"Testing {m_name}", total=1000):
            results.append(
                test(
                    i, m["name"], obj, model
                )
            )
        np.savetxt(f"result/{m['model_name']}_{m_name}.txt", results)


if __name__ == "__main__":
    model_name = "model_2821000_steps"

    models = [
        {
            "name": f"{model_name}",
            "path": f"model/best/",
            "model_name": model_name
        }
    ]

    test_model(models)
