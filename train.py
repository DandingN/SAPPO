#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/4/30 下午7:01
# @function: the script is used to do something.
# @version : V1
import glob
import os

from tqdm import tqdm

from env import MyNewEnv

from types import SimpleNamespace
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from test_model import test_model
from stable_baselines3.common.callbacks import CheckpointCallback


def get_object():
    return np.load("data/object/object.npy")


def make_env(config):
    return SubprocVecEnv([lambda: MyNewEnv(config) for _ in range(config.n_envs)])


def print_info(env_conf, model_conf):
    print("-----environment config-----")
    print(f"observation_size:{env_conf.observation_size}")
    print(f"object_percent:{env_conf.object_percent}")
    print(f"repeat_punish:{env_conf.repeat_punish}")
    print(f"n_envs:{env_conf.n_envs}")
    print("-----model config-----")
    print(f"total_timesteps:{model_conf.total_timesteps}")
    print(f"n_steps:{model_conf.n_steps}")
    print(f"policy_kwargs:{model_conf.policy_kwargs}")


if __name__ == "__main__":
    env_config = SimpleNamespace(**{
        "env_id": "MyNewEnv",  # 环境名
        "num": np.random.randint(1, 1000),
        "start_pos": [0, 0],  # 起始位置
        "grid_size": 10,  # 网格环境尺寸
        "observation_size": 5,  # todo 观察空间尺寸------------------------------------------------------------------------------------
        "max_episode_steps": 31,  # 回合最大步数
        "repeat_punish": 0,  # 重复惩罚
        "zero_punish": 0,  # 零值惩罚
        "object_percent": 0.9,  # 精确算法目标值比率
        "object_list": get_object(),  # 环境目标值池
        "render": False,  # 渲染选项
        "render_mode": "rgb_array",  # 渲染模式
        "test_mode": False,  # 测试模式
        "model_dir": "./models/ppo/",  # 模型位置
        "n_envs": 10  # 并行环境数量
    })
    env = make_env(env_config)

    model_config = SimpleNamespace(**{
        "policy": "MlpLstmPolicy",  # 策略
        "env": env,  # 环境
        "learning_rate": 3e-4,  # 学习率

        # nohup python3 train.py > 20250504_12.txt 2>&1 &
        "tensorboard_log": f"log/RecurrentPPO/{20250504_15}",  # todo TensorBoard日志路径（若为None则不记录）
        "n_steps": 28,  # todo 每次更新中每个环境运行的步数
        "total_timesteps": 403_0000,  # 指定模型与环境交互的总次数

        "batch_size": 128,  # Minibatch size
        "n_epochs": 10,  # 优化代理损失函数时的迭代轮次
        "gamma": 0.99,  # 折扣因子
        "gae_lambda": 0.95,  # 广义优势估计（GAE）的衰减因子，平衡偏差与方差
        "clip_range": 0.2,  # 策略更新的剪切范围（PPO-Clip核心参数，控制概率比的截断区间）
        "clip_range_vf": None,  # 值函数的剪切范围（OpenAI实现特有参数，依赖奖励缩放）
        "normalize_advantage": True,  # 是否对优势函数进行标准化处理
        "ent_coef": 0.0,  # 熵系数（Entropy Coefficient），用于鼓励策略探索
        "vf_coef": 0.0,  # 值函数损失的权重系数
        "max_grad_norm": 0.5,  # 梯度裁剪的最大范数值（防止梯度爆炸）
        "use_sde": False,  # 用gSDE替代随机动作噪声
        "sde_sample_freq": -1,  # 噪声参数的重采样频率
        "target_kl": None,  # 策略更新的KL散度阈值（限制新旧策略差异，增强稳定性）
        "stats_window_size": 100,  # 统计窗口大小（用于计算平均成功率、奖励等指标）
        # "policy_kwargs": dict(
        #     lstm_hidden_size=128,
        #     n_lstm_layers=1,
        #     lstm_kwargs=dict(
        #         dropout=0.2
        #     )
        # ),  # 策略网络的附加参数（如网络结构配置）
        "policy_kwargs": None,
        "verbose": 1,  # 输出详细程度（0:无输出，1:信息级，2:调试级）
        "seed": 79811,  # 伪随机数生成器的种子值
        "device": "auto",  # 计算设备（cpu/cuda，auto表示自动选择GPU）
        "init_setup_model": True,  # 是否在初始化时构建计算图
        "callback": None,  # 回调函数（可多个），在算法运行每一步被调用，用于监控或干预训练流程
        "log_interval": 1,  # 日志记录间隔
        "tb_log_name": "RecurrentPPO",  # TensorBoard日志名称——用于标识本次训练运行的名称
        "reset_num_timesteps": False,  # 是否重置时间步计数器（影响日志中的步数统计）
        "progress_bar": True  # 是否显示进度条（使用tqdm和rich库实现可视化进度追踪）
    })

    # 输出环境及模型信息
    print_info(env_config, model_config)

    # region
    model = RecurrentPPO(
        policy=model_config.policy,
        env=model_config.env,
        learning_rate=model_config.learning_rate,
        n_steps=model_config.n_steps,
        batch_size=model_config.batch_size,
        n_epochs=model_config.n_epochs,
        gamma=model_config.gamma,
        gae_lambda=model_config.gae_lambda,
        clip_range=model_config.clip_range,
        clip_range_vf=model_config.clip_range_vf,
        normalize_advantage=model_config.normalize_advantage,
        ent_coef=model_config.ent_coef,
        vf_coef=model_config.vf_coef,
        max_grad_norm=model_config.max_grad_norm,
        use_sde=model_config.use_sde,
        sde_sample_freq=model_config.sde_sample_freq,
        target_kl=model_config.target_kl,
        stats_window_size=model_config.stats_window_size,
        tensorboard_log=model_config.tensorboard_log,
        policy_kwargs=model_config.policy_kwargs,
        verbose=model_config.verbose,
        seed=model_config.seed,
        device=model_config.device,
        _init_setup_model=model_config.init_setup_model
    )
    # endregion

    model_name = f"{model_config.tensorboard_log.split('/')[-1]}"

    model_num = 20
    save_freq = model_config.total_timesteps // model_num // env_config.n_envs
    # 设置每隔40万步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,  # 保存间隔步数
        save_path=f"models/{model_name}",  # 保存路径（与TensorBoard日志路径一致）
        name_prefix="model",  # 文件名前缀（如 "model_400000_steps"）
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )

    model.learn(
        total_timesteps=model_config.total_timesteps,
        callback=checkpoint_callback,
        log_interval=model_config.log_interval,
        tb_log_name=model_config.tb_log_name,
        reset_num_timesteps=model_config.reset_num_timesteps,
        progress_bar=model_config.progress_bar
    )

    model.save(model_name)

    # 动态获取所有保存的模型文件
    save_path = f"models/{model_name}"
    model_files = glob.glob(os.path.join(save_path, "model_*.zip"))
    model_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))  # 按步数排序

    # 取最后10个模型
    last_n_models = model_files[-10:]

    models = [
        {
            "name": f"{file.split('.')[0].split('/')[-1]}",
            "path": f"models/{model_name}/",
            "model_name": model_name
        } for file in last_n_models
    ]

    test_model(models)
