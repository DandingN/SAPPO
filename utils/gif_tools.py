#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2024/11/11 下午4:55
# @function: 处理GIF文件
# @version : V1
import multiprocessing
import os
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation


def display_frames_as_gif(frames: list, folder_name: str) -> None:
    """
    将一系列图像帧显示为GIF动画并保存至文件中
    :param frames: 包含图像帧的列表
    :param folder_name: 存储文件夹位置
    """
    # 判断文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    n = 1
    while os.path.exists(os.path.join(folder_name, str(n) + ".gif")):
        n += 1

    gif_path = os.path.join(folder_name, str(n) + ".gif")

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i: int) -> None:
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save(gif_path, writer="pillow", fps=10)


def load_gif(file_path: str) -> list:
    """
    加载 GIF 文件并返回每一帧的图像列表
    :param file_path: GIF文件路径
    :return 图像列表
    """
    gif = Image.open(file_path)
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(len(frames))  # 移动到下一帧
    except EOFError:
        pass  # 到达 GIF 的末尾
    return frames


def combine_gifs_and_mp4(directory: str, output_path: str, max_per_row: int = 5) -> None:
    """
    将多个 GIF 合并为一个 GIF，同时播放，每行最多 max_per_row 个 GIF
    :param directory: 输入文件路径
    :param output_path: 输出路径
    :param max_per_row: 每行最大图像数
    """

    # 获取指定文件夹下以指定前缀开头的 GIF 文件列表
    gif_paths = []
    for filename in os.listdir(directory):
        if filename.endswith('.gif'):
            gif_paths.append(os.path.join(directory, filename))

    # 加载所有 GIF 文件
    all_frames = [load_gif(path) for path in gif_paths]

    # 获取所有 GIF 的最大帧数
    max_frames = max(len(frames) for frames in all_frames)

    # 获取所有 GIF 的最大宽度和高度
    max_width = max(frame.size[0] for frames in all_frames for frame in frames)
    max_height = max(frame.size[1] for frames in all_frames for frame in frames)

    # 计算行数和列数
    num_gifs = len(gif_paths)
    num_rows = (num_gifs + max_per_row - 1) // max_per_row  # 向上取整
    num_cols = min(num_gifs, max_per_row)

    # 创建一个空的图像列表来存储合并后的帧
    combined_frames = []

    for i in range(max_frames):
        # 创建一个新的空白图像，大小为所有 GIF 的最大宽度和高度之和
        combined_frame = Image.new('RGBA', (max_width * num_cols, max_height * num_rows))

        for j, frames in enumerate(all_frames):
            if i < len(frames):
                frame = frames[i]
            else:
                frame = frames[-1]  # 如果 GIF 的帧数不够，使用最后一帧

            # 计算当前帧的位置
            row = j // max_per_row
            col = j % max_per_row

            # 将当前帧粘贴到组合图像中
            combined_frame.paste(frame, (col * max_width, row * max_height))

        # 将组合帧转换为 numpy 数组
        combined_frames.append(np.array(combined_frame))

    # 保存合并后的 GIF
    imageio.mimsave(output_path + ".gif", combined_frames, 'GIF', duration=0.1)

    # 保存合并后的 MP4
    imageio.mimsave(output_path + ".mp4", combined_frames, 'MP4', fps=10)


def create_gif_folder(folder_name):
    # 判断文件夹是否存在
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
