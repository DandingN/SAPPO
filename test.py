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
# from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
from utils import evaluate_policy


def get_object():
    return np.load("data/object/object.npy")


# noinspection PyShadowingNames
def test(env_num, model_name, observation_size, repeat_punish, object_percent, env_):
    env = Monitor(env_(SimpleNamespace(**{
        "env_id": "MyNewEnv",  # 环境名
        "num": env_num,
        "start_pos": [0, 0],  # 起始位置
        "grid_size": 10,  # 网格环境尺寸
        "observation_size": observation_size,  # 观察空间尺寸
        "max_episode_steps": 31,  # 回合最大步数
        "repeat_punish": repeat_punish,  # 重复惩罚
        "zero_punish": 0,  # 零值惩罚
        "object_percent": object_percent,  # 精确算法目标值比率
        "object_list": get_object(),  # 环境目标值池
        "render": False,  # 渲染选项
        "render_mode": "rgb_array",  # 渲染模式
        "test_mode": True,  # 测试模式
        "model_dir": "./models/ppo/"  # 模型位置
    })), f"log/test/{model_name}")

    episode_rewards, episode_length = evaluate_policy(model, env, n_eval_episodes=30, return_episode_rewards=True)
    return max(episode_rewards)


if __name__ == '__main__':
    # models = [
    #     # {
    #     #     "name": "Recurrent_PPO_20250502_2",
    #     #     "observation_size": 5,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.9,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "Recurrent_PPO_20250502_5",
    #     #     "observation_size": 5,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.95,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "2_7",
    #     #     "observation_size": 3,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.9,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "RecurrentPPO_20250502_8",
    #     #     "observation_size": 5,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.9,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "2_9",
    #     #     "observation_size": 10,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.9,
    #     #     "env": AllEnv,
    #     # },
    #     #
    #     # {
    #     #     "name": "2_10",
    #     #     "observation_size": 3,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.95,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "2_11",
    #     #     "observation_size": 5,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.95,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "2_12",
    #     #     "observation_size": 10,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.95,
    #     #     "env": AllEnv,
    #     # },
    #
    #     # {
    #     #     "name": "RecurrentPPO_20250502_15",
    #     #     "observation_size": 3,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.9,
    #     #     "env": RandomEnv,
    #     # },
    #     # {
    #     #     "name": "RecurrentPPO_20250502_16",
    #     #     "observation_size": 5,
    #     #     "repeat_punish": 0,
    #     #     "object_percent": 0.9,
    #     #     "env": RandomEnv,
    #     # },
    #
    #     # {
    #     #     "name": "2_17",
    #     #     "observation_size": 3,
    #     #     "repeat_punish": -0.1,
    #     #     "object_percent": 0.9,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "RecurrentPPO_20250502_18",
    #     #     "observation_size": 5,
    #     #     "repeat_punish": -0.1,
    #     #     "object_percent": 0.9,
    #     #     "env": MyNewEnv,
    #     # },
    #     #
    #     # {
    #     #     "name": "RecurrentPPO_20250502_19",
    #     #     "observation_size": 3,
    #     #     "repeat_punish": -0.1,
    #     #     "object_percent": 0.9,
    #     #     "env": MyNewEnv,
    #     # },
    #     # {
    #     #     "name": "RecurrentPPO_20250502_20",
    #     #     "observation_size": 5,
    #     #     "repeat_punish": -0.1,
    #     #     "object_percent": 0.9,
    #     #     "env": MyNewEnv,
    #     # },
    #
    # ]

    models = [
        {
            "name": "RecurrentPPO_20250503_1",
            "observation_size": 5,
            "repeat_punish": 0,
            "object_percent": 0.9,
            "env": MyNewEnv,
        },
        # {
        #     "name": "RecurrentPPO_20250503_5",
        #     "observation_size": 5,
        #     "repeat_punish": 0,
        #     "object_percent": 0.9,
        #     "env": MyNewEnv,
        # },
        # {
        #     "name": "RecurrentPPO_20250503_6",
        #     "observation_size": 5,
        #     "repeat_punish": 0,
        #     "object_percent": 0.9,
        #     "env": MyNewEnv,
        # },
        # {
        #     "name": "Recurrent_PPO_20250503_7",
        #     "observation_size": 5,
        #     "repeat_punish": 0,
        #     "object_percent": 0.9,
        #     "env": MyNewEnv,
        # },
        # {
        #     "name": "Recurrent_PPO_20250503_8",
        #     "observation_size": 5,
        #     "repeat_punish": 0,
        #     "object_percent": 0.9,
        #     "env": MyNewEnv,
        # },
        # {
        #     "name": "Recurrent_PPO_20250503_9",
        #     "observation_size": 5,
        #     "repeat_punish": 0,
        #     "object_percent": 0.9,
        #     "env": MyNewEnv,
        # },
    ]

    for m in tqdm(models, desc="Processing Models", total=len(models)):
        model_name = m["name"]
        model = RecurrentPPO.load(f"model/{m['name']}")
        output_file = f"results_{m['name']}.txt"

        with open(output_file, "w") as f:
            for i in tqdm(range(1, 1001), desc=f"Testing {model_name}", total=1000):
                temp = test(i, m["name"], m["observation_size"], m["repeat_punish"], m["object_percent"], m["env"])
                f.write(f"{temp}\n")
