#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/8 下午4:40
# @function: the script is used to do something.
# @version : V1
# 给定的数列
numbers = [
    2, 2, 1, 3, 2, 2, 2, 1, 2, 2, 2, 3, 1, 1, 1, 2, 3, 3, 3, 2, 2, 3, 2, 1, 3, 3, 1, 3, 3, 1, 3, 2, 1, 2, 2, 3, 1, 2, 2, 1, 1, 1, 3, 1, 1, 1, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 1, 3, 3, 3, 3, 2, 3, 1, 1, 2, 2, 1, 2, 3, 1, 1, 2, 3, 3, 3, 1, 1, 3, 1, 3, 1, 3, 2, 1, 2, 1, 3, 1, 2, 1, 1, 1, 2, 3, 1, 1, 3, 3, 2
]
# 初始化计数器
count_1 = 0
count_2 = 0
count_3 = 0

# 循环数列并计数
for number in numbers:
    if number == 1:
        count_1 += 1
    elif number == 2:
        count_2 += 1
    elif number == 3:
        count_3 += 1

# 输出结果
print(f"1的数量: {count_1}")
print(f"2的数量: {count_2}")
print(f"3的数量: {count_3}")
