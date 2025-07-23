#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/1 下午7:04
# @function: the script is used to do something.
# @version : V1
from env import MyNewEnv, AllEnv, RandomEnv

import numpy as np
from stable_baselines3.common.monitor import Monitor
from types import SimpleNamespace
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
from utils import evaluate_policy


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

    episode_rewards, episode_length = evaluate_policy(model, env, n_eval_episodes=30, return_episode_rewards=True)
    return max(episode_rewards)


def test_model(models):
    obj = np.load("data/object/object.npy")
    for m in tqdm(models, desc="Processing Models", total=len(models)):
        m_name = m["name"]
        model = RecurrentPPO.load(f"{m['path']}{m['name']}")
        results = []
        for i in tqdm(range(1, 1001), desc=f"Testing {m_name}", total=1000):
            results.append(
                test(
                    i, m["name"], obj, model
                )
            )
        np.savetxt(f"result/{m['model_name']}_{m_name}.txt", results)