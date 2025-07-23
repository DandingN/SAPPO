#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/2 下午8:45
# @function: the script is used to do something.
# @version : V1
import warnings
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int], list[list]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episodes to evaluate the agent.
    :param deterministic: Whether to use deterministic or stochastic actions.
    :param render: Whether to render the environment or not.
    :param callback: Callback function to do additional checks.
    :param reward_threshold: Minimum expected reward per episode.
    :param return_episode_rewards: If True, returns per-episode rewards, lengths, and actions.
    :param warn: Whether to warn about missing Monitor wrapper.
    :return: Mean reward and std, or (episode_rewards, episode_lengths, episode_actions) if return_episode_rewards is True.
    """
    is_monitor_wrapped = False
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "Consider wrapping it first.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_actions = []  # 新增：用于存储每个 episode 的动作序列

    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_actions = [[] for _ in range(n_envs)]  # 每个环境当前 episode 的动作序列

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)

        current_rewards += np.maximum(rewards, 0)
        current_lengths += 1

        # 记录动作
        for i in range(n_envs):
            current_actions[i].append(actions[i])  # 保存每个 step 的动作

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if done:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_actions.append(current_actions[i])  # 保存当前 episode 的动作序列
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    current_actions[i] = []  # 重置当前 episode 的动作序列

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"

    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_actions  # 返回三元组
    return mean_reward, std_reward
