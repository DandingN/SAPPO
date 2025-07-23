#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/6/27 上午10:48
# @function: the script is used to do something.
# @version : V1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 设置高斯分布的参数
mean = [0, 0]          # 均值
covariance = [[1, 0.5],
              [0.5, 2]]  # 协方差矩阵

# 创建网格点
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 创建多元正态分布对象
rv = multivariate_normal(mean, covariance)

# 计算概率密度
Z = rv.pdf(pos)

# 绘制三维曲面图
fig = plt.figure(figsize=(12, 5))

# 3D曲面图
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
ax1.set_title('3D Gaussian Distribution')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Probability Density')

# 二维等高线图
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(contour)
ax2.set_title('2D Gaussian Contour')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True)

plt.tight_layout()
plt.show()