#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2024/11/26 下午2:38
# @function: 随机生成训练场景，且可绘制场景图
# @version : V1
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def generate_gaussian_grid(size, num_centers):
    # todo 加障碍物密度
    grid = np.zeros((size, size))
    num_centers = num_centers  # 随机选择1到3个中心点
    centers = np.random.randint(size // 5, size - size // 5, (num_centers, 2))
    cov = [
        [size // np.random.randint(1, size), 0],
        [0, size // np.random.randint(1, size)]
    ]  # 协方差矩阵

    for center in centers:
        rv = multivariate_normal(mean=center, cov=cov)
        for x in range(size):
            for y in range(size):
                grid[x, y] += rv.pdf([x, y])

    grid /= np.sum(grid)  # 使和为1
    grid = np.round(grid, 2)  # 保留两位小数
    return grid


def plot_envs(num_files):
    for i in range(num_files):
        # 加载文件
        file_path = f"../data/{i + 1}.npy"
        grid = np.load(file_path)

        # 创建图像
        # plt.figure(figsize=(4, 4), dpi=150)  # 增加分辨率
        plt.imshow(grid, cmap='rainbow')  # 使用更美观的颜色映射
        plt.colorbar(label='Value')  # 缩短图例标签
        plt.title(f"Env {i + 1}", fontsize=16)  # 增加标题字体大小

        # 保存图像
        plt.savefig(f"../data/img/env_{i + 1}.png", bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    for i in range(1000):
        np.save("../data/" + str(i + 1) + ".npy", generate_gaussian_grid(10, np.random.randint(1, 4)))
    # 调用函数生成图像
    plot_envs(1000)
    pass
