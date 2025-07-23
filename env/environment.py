# !/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/3/4 下午3:40
# @function: the script is used to do something.
# @version : V1
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from utils import ColorCalculator, get_model_name, display_frames_as_gif
from utils.draw_plot import draw_path


class MyNewEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, env_config):
        super(MyNewEnv, self).__init__()
        self.env_id = env_config.env_id
        self.start_pos = env_config.start_pos  # 智能体起始位置设定
        self.positions = self.start_pos

        self.repeat_punish = env_config.repeat_punish  # 重复探索的惩罚
        self.zero_punish = env_config.zero_punish  # 零值惩罚

        self.size = env_config.grid_size  # 网格环境长宽
        self.observation_size = env_config.observation_size  # 观察空间大小
        self.window_size = self.observation_size // 2
        self.observation_size_ = (self.observation_size ** 2) + 1 + 2  # 观察窗口 + 剩余步数 + 当前位置

        self.observation_space = Box(low=0, high=1, shape=[self.observation_size_, ], dtype=np.float32)
        self.action_space = Discrete(4)

        self.num = env_config.num
        self.threshold = 0
        self.object_list = env_config.object_list
        self.object_percent = env_config.object_percent

        self.grid = self.generate_grid(self.num)  # 初始化网格

        self.max_episode_steps = env_config.max_episode_steps  # 单episode最大步数
        self.steps = []
        self._current_step = 0  # 当前episode所处步数
        self.remaining_steps = self.max_episode_steps - self._current_step  # 当前episode剩余步数

        self.render_mode = env_config.render_mode  # 渲染模式
        self.is_render = env_config.render  # 是否渲染
        self.screen = None
        self.clock = None
        self.cell_width = 30
        self.screen_width = self.size * self.cell_width
        # 颜色计算器
        self.color_calculator = ColorCalculator(
            cmap_name="rainbow",
            max_val=np.max(self.grid),
            min_val=np.min(self.grid)
        )
        self.frames = []  # 视频帧数组
        self.test = env_config.test_mode  # 测试模式
        self.model_folder = env_config.model_dir  # 模型路径

    def generate_grid(self, num):
        """
        生成符合高斯分布的网格
        :return: 高斯分布的网格
        """
        grid = np.load(f"./data/npy/{num}.npy") * 100
        grid /= np.sum(grid)
        grid *= 100
        # num_centers = np.random.randint(1, 4)  # 随机选择1到3个中心点
        # grid = generate_gaussian_grid(self.size, num_centers)
        return np.where(grid == 0, self.zero_punish, grid)

    def reset_color_calculator(self):
        self.color_calculator = ColorCalculator(
            cmap_name="rainbow",
            max_val=np.max(self.grid),
            min_val=np.min(self.grid)
        )

    def observe(self):
        x, y = self.positions
        window = []
        # 提取窗口
        for dx in range(-self.window_size, self.window_size + 1):
            for dy in range(-self.window_size, self.window_size + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    value = self.grid[nx, ny]
                else:
                    value = 0  # 越界部分用0填充
                window.append(value)

        # 剩余步数归一化
        remaining_steps_normalized = self.remaining_steps / self.max_episode_steps

        # 坐标归一化
        pos_x_normalized = x / (self.size - 1)
        pos_y_normalized = y / (self.size - 1)

        observation = np.array(
            window + [remaining_steps_normalized, pos_x_normalized, pos_y_normalized],
            dtype=np.float32
        )
        return observation

    def get_target(self):
        return self.object_list[self.num - 1] * self.object_percent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.observe()
        info = {}
        self._current_step = 0

        # 单局超过目标阈值，则更换地图继续训练
        target = self.get_target()
        if self.threshold > target and self.test is False:
            self.num = random.randint(1, 1000)
        self.threshold = 0
        self.grid = self.generate_grid(self.num)
        self.positions = self.start_pos  # 初始化智能体位置
        self.reset_color_calculator()
        return observation, info

    def step(self, action):
        info = {
            "episode_rewards": 0
        }

        x, y = self.positions

        self.steps.append(self.positions)

        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.size - 1, y + 1)

        self.positions = [x, y]
        rewards = self.grid[x, y]
        self.threshold += max(rewards, 0)

        if self.test:
            info["episode_rewards"] = 0 if self.grid[x, y] <= 0 else self.grid[x, y]

        self.grid[x, y] = self.repeat_punish

        self._current_step += 1
        self.remaining_steps = self.max_episode_steps - self._current_step  # 当前episode剩余步数
        terminated = self._current_step >= self.max_episode_steps
        truncated = self._current_step >= self.max_episode_steps
        observation = self.observe()
        # # 测试模式存储gif
        # if truncated is True and self.test and self.render_mode == "rgb_array":
        #     model_name = get_model_name(self.model_folder)  # 获取当前测试模型的名称
        #     display_frames_as_gif(self.frames, "gif/" + model_name)
        #     draw_path(steps=self.steps, grid=self.generate_grid(self.num), file_path="gif/" + model_name)
        #
        # # 测试时保存视频帧
        # if self.test and self.render_mode == "rgb_array":
        #     self.frames.append(self.render())

        return observation, rewards, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if not self.is_render:
            return None

        if self.screen is None:
            pygame.init()
            if self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_width))

        pygame.display.set_caption("Grid Search")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))  # 设置背景色为黑色

        # 按照背景色绘制网格环境
        for x in range(self.size):
            for y in range(self.size):
                pygame.draw.rect(
                    self.screen,
                    self.color_calculator.value_to_color(self.grid[x, y]),
                    (y * self.cell_width, x * self.cell_width, self.cell_width, self.cell_width)
                )

        # 绘制智能体 及 智能体编号
        font = pygame.font.Font(None, 16)

        # 绘圆
        pygame.draw.circle(
            self.screen,
            color=(255, 255, 255),
            center=(
                self.positions[1] * self.cell_width + self.cell_width // 2,
                self.positions[0] * self.cell_width + self.cell_width // 2
            ),
            radius=self.cell_width // 3,
        )

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            # 将屏幕内容转换为RGB数组并返回
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()

        pygame.quit()
