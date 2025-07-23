#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2024/10/31 15:03
# @function: the script is used to do something.
# @version : V1
import os
import random
from typing import Tuple, Dict, List

from matplotlib import cm


# noinspection PyTypeChecker
class ColorCalculator:
    def __init__(self, cmap_name: str, max_val: float, min_val: float) -> None:
        """
        初始化颜色计算器
        :param cmap_name: matplotlib 色带名称
        :param max_val: 最大值
        :param min_val: 最小值
        """
        self.color_band = cm.get_cmap(cmap_name, None)

        self.max_val = max_val
        self.min_val = min_val

    def value_to_color(self, value: float) -> Tuple[int, int, int]:
        """
        将值转换为颜色
        :param value: 值
        :return: 颜色数组
        """
        normalized_value = (value - self.min_val) / (self.max_val - self.min_val)
        color = tuple(int(round(component * 255)) for component in self.color_band(normalized_value)[:3])
        return color


def get_start_pos(num_agents: int, grid_size: int) -> Dict[str, List[int]]:
    """
    返回随机初始位置
    :param num_agents: 智能体个数
    :param grid_size: 网格空间长宽
    :return: 智能体位置字典
    """
    start_pos = {}
    for i in range(num_agents):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        start_pos[f'agent_{i}'] = [x, y]
    return start_pos


def get_model_name(directory: str) -> str or None:
    """
    获取文件夹下最后一个以seed为开头的文件夹名
    :param directory: test文件夹路径
    :return: 最新模型名称
    """
    # 获取目录下所有文件和文件夹的列表
    all_items = os.listdir(directory)

    # 筛选出以 "seed" 开头的文件夹
    seed_folders = [item for item in all_items if
                    item.startswith("seed") and os.path.isdir(os.path.join(directory, item))]

    # 如果没有找到符合条件的文件夹，返回 None
    if not seed_folders:
        return

    # 按字母顺序排序，并返回最后一个文件夹的名字
    seed_folders.sort()
    return seed_folders[-1]
