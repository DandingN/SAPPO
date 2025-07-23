#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/3/6 下午12:24
# @function: the script is used to do something.
# @version : V1
import os.path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import matplotlib


def draw_path(steps, grid, file_path):
    plt.figure(figsize=(10, 8), dpi=100)

    # 创建掩膜并设置0值为白色
    vmin, vmax = grid.min(), grid.max()
    masked_grid = grid.astype(float)
    masked_grid[grid == 0] = np.nan  # 将0替换为NaN
    cmap = matplotlib.colormaps["OrRd"]
    cmap.set_bad('#DFFAFF')  # 设置NaN值的颜色为白色

    # 绘制网格数值（使用热图）
    plt.imshow(masked_grid,
               cmap=cmap,
               origin='upper',
               extent=[0, grid.shape[1], grid.shape[0], 0],
               vmin=vmin,
               vmax=vmax)
    plt.colorbar(label="POS")

    # 设置坐标轴刻度（显示在格子边缘）
    ax = plt.gca()
    # 设置刻度位置（每个单元格的边界）
    ax.set_xticks(np.arange(grid.shape[1] + 1))
    ax.set_yticks(np.arange(grid.shape[0] + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # 隐藏刻度线（设置长度为0）
    ax.tick_params(which='both', length=1)
    ax.grid(color='#343434', linewidth=1)  # 显示网格线

    # 转换路径坐标（假设steps是[[row0,col0], [row1,col1], ...]）
    if len(steps) > 0:
        steps = np.array(steps)
        x = steps[:, 1] + 0.5  # 列索引 -> x坐标（中心点）
        y = steps[:, 0] + 0.5  # 行索引 -> y坐标（中心点）

        # 路径连线
        plt.plot(x, y, 'k-', linewidth=4, zorder=2)  # 改为黑色

        # 加载无人机图片
        # drone_img = plt.imread("utils/assets/drone.png")
        drone_img = plt.imread("assets/drone.png")
        imagebox = OffsetImage(drone_img, zoom=0.2)  # 调整zoom参数控制图片大小
        ab = AnnotationBbox(imagebox, (x[0], y[0]), frameon=False, zorder=4)
        ax.add_artist(ab)

        # 箭头终点（当路径长度>1时绘制）
        if len(steps) > 1:
            # 计算最后一段方向向量
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]

            # 绘制方向箭头（在终点位置）
            plt.quiver(x[-1] - dx * 0.8, y[-1] - dy * 0.8, dx, dy,  # 起点略微后退
                       scale_units='xy', angles='xy', scale=1,
                       color='black', width=0.007, headwidth=5,
                       headlength=4, headaxislength=4,
                       zorder=4)

    plt.title("The path of the agent")
    # plt.legend(loc='lower right')

    # todo 文件路径修改
    # 使用plt.savefig()保存图像
    num = 1
    while os.path.exists(os.path.join(file_path, str(num) + ".png")):
        num += 1
    # plt.savefig(os.path.join(file_path, str(num) + ".png"))

    plt.show()


if __name__ == "__main__":
    # DQN
    s = [
        [0, 0],
        [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [5, 3], [6, 3], [6, 4],
        [6, 5], [7, 5], [7, 6], [7, 7], [6, 7], [5, 7], [4, 7], [3, 7], [3, 6], [3, 5],
        [3, 4], [4, 4], [5, 4], [5, 5], [4, 5], [3, 5], [2, 5], [2, 6], [2, 7], [2, 8]
    ]

    # ql
    s = [
        [0, 0],
        [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [5, 2], [6, 2], [6, 3], [5, 3],
        [5, 4], [6, 4], [6, 5], [6, 6], [5, 6], [4, 6], [4, 7], [3, 7], [3, 8], [3, 9],
        [4, 9], [4, 8], [5, 8], [5, 7], [5, 6], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2]
    ]

    # ppo
    s = [
        [0, 0],
        [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [5, 3], [6, 3], [6, 4],
        [7, 4], [7, 5], [7, 6], [7, 7], [6, 7], [6, 6], [6, 5], [5, 5], [5, 4], [4, 4],
        [4, 5], [3, 5], [3, 6], [3, 7], [2, 7], [2, 8], [3, 8], [4, 8], [4, 7], [4, 6],
    ]

    g = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1.03092784, 0, 0],
        [0, 0, 0, 0, 0, 1.03092784, 2.06185567, 2.06185567, 2.06185567, 1.03092784],
        [0, 0, 0, 1.03092784, 1.03092784, 2.06185567, 3.09278351, 4.12371134, 3.09278351, 1.03092784],
        [0, 1.03092784, 2.06185567, 2.06185567, 2.06185567, 2.06185567, 2.06185567, 2.06185567, 2.06185567, 1.03092784],
        [0, 1.03092784, 3.09278351, 4.12371134, 3.09278351, 2.06185567, 1.03092784, 1.03092784, 1.03092784, 0],
        [0, 1.03092784, 2.06185567, 3.09278351, 3.09278351, 3.09278351, 3.09278351, 2.06185567, 1.03092784, 0],
        [0, 0, 0, 1.03092784, 2.06185567, 3.09278351, 4.12371134, 3.09278351, 1.03092784, 0],
        [0, 0, 0, 0, 1.03092784, 2.06185567, 2.06185567, 2.06185567, 1.03092784, 0],
        [0, 0, 0, 0, 0, 0, 1.03092784, 0, 0, 0]
    ])

    draw_path(s, g, "")
