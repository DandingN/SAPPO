#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/6/27 下午8:56
# @function: the script is used to do something.
# @version : V1
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

# 设置顶刊级别的绘图参数
plt.rcParams.update({
    # 字体设置 - 使用专业的衬线字体
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,

    # 数学公式字体
    'mathtext.fontset': 'stix',  # 科技出版物常用

    # 坐标轴设置
    'axes.linewidth': 0.8,  # 边框线宽
    'axes.edgecolor': '#2f2f2f',  # 深灰色边框
    'axes.labelcolor': '#2f2f2f',
    'axes.titlelocation': 'center',  # 标题居中
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',

    # 网格和背景
    'axes.grid': True,
    'grid.color': '#f0f0f0',
    'grid.alpha': 0.7,
    'grid.linestyle': '--',

    # 刻度设置
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': '#2f2f2f',
    'ytick.color': '#2f2f2f',

    # 图形设置
    'figure.dpi': 300,
    'figure.autolayout': True,
    'figure.titleweight': 'bold',
})

# 创建自定义颜色方案
nasa_colors = ['#0067A5', '#A4D7F4', '#E9C46A', '#E76F51']
palette = {
    "Our Model": nasa_colors[0],
    "w/o Observation Space": nasa_colors[1],
    "w/o LSTM": nasa_colors[2],
    "w/o Threshold Switching": nasa_colors[3]
}

obj = np.load("data/object/object.npy")[200:300]

df_tem = pd.read_excel("excel_results/result.xlsx", sheet_name='原始结果')

# 最好的模型
df_best = df_tem.loc[200:299, "051_2821000"]

df_temp = pd.read_excel("data/result/results.xlsx", sheet_name='Sheet1')

# 无观测空间
df_without_obs = df_temp.loc[200:299, "20250502_9"]
# 无LSTM
df_without_lstm = df_temp.loc[200:299, "20250501_8"]
# 无阈值切换机制
df_without_threshold = df_temp.loc[200:299, "20250502_16"]

# 将 obj_sub 转换为 Series
obj_series = pd.Series(obj, index=df_best.index)

# 执行除法并将结果乘以100以显示百分比
df_best_divided = (df_best / obj_series).clip(upper=1) * 100
df_without_obs_divided = (df_without_obs / obj_series).clip(upper=1) * 100
df_without_lstm_divided = (df_without_lstm / obj_series).clip(upper=1) * 100
df_without_threshold_divided = (df_without_threshold / obj_series).clip(upper=1) * 100

# 合并为 DataFrame
results_df = pd.DataFrame({
    "Our Model": df_best_divided,
    "w/o Observation Space": df_without_obs_divided,
    "w/o LSTM": df_without_lstm_divided,
    "w/o Threshold Switching": df_without_threshold_divided
})

# 将DataFrame转换为长格式以便Seaborn处理
results_melted = results_df.melt(var_name="Model Type", value_name="Performance Score")

# 创建图形和子图
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300, facecolor='white')

# 绘制箱线图
sns.boxplot(
    x="Model Type",
    y="Performance Score",
    data=results_melted,
    palette=palette,
    fliersize=0,
    width=0.7,
    boxprops=dict(alpha=0.95, linewidth=1.2, edgecolor='#404040'),
    whiskerprops=dict(color='#404040', linewidth=1.2),
    capprops=dict(color='#404040', linewidth=1.2),
    medianprops=dict(color='#404040', linewidth=1.8),
    showfliers=False,
    ax=ax
)

# 添加数据点 - 使用抖动(jitter)避免重叠
sns.stripplot(
    x="Model Type",
    y="Performance Score",
    data=results_melted,
    palette=palette,
    jitter=0.2,
    size=4.5,
    alpha=0.7,
    edgecolor='#404040',
    linewidth=0.4,
    ax=ax
)

# 添加标题和标签
ax.set_title("Ablation Study Results", fontsize=13, pad=15)
ax.set_ylabel("Performance Score (%)", fontsize=11, labelpad=8)
ax.set_xlabel("Model Variant", fontsize=11, labelpad=8)

# 添加顶部和右侧边框
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_color('#e0e0e0')
ax.spines['right'].set_color('#e0e0e0')

# 添加均值标记
for i, model in enumerate(results_df.columns):
    mean_val = results_df[model].mean()
    ax.plot([i - 0.25, i + 0.25], [mean_val, mean_val],
            color='#202020', linestyle='-', linewidth=2.5,
            solid_capstyle='round')
    ax.text(i + 0.36, mean_val, f'{mean_val:.1f}',
            fontsize=10, verticalalignment='center',
            color='#202020', fontweight='bold')

# 优化刻度标签
ax.tick_params(axis='both', which='major', labelsize=10, pad=5)

# 添加网格线
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.4)

# 添加图例
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='s', color='w', label='Our Model',
           markerfacecolor=palette["Our Model"], markersize=10),
    Line2D([0], [0], marker='s', color='w', label='w/o Observation',
           markerfacecolor=palette["w/o Observation Space"], markersize=10),
    Line2D([0], [0], marker='s', color='w', label='w/o LSTM',
           markerfacecolor=palette["w/o LSTM"], markersize=10),
    Line2D([0], [0], marker='s', color='w', label='w/o Threshold',
           markerfacecolor=palette["w/o Threshold Switching"], markersize=10)
]
# ax.legend(handles=legend_elements, loc='lower right',
#           ncol=1, frameon=True,
#           framealpha=0.9, fontsize=10)

# 调整布局
plt.tight_layout()

# 保存为出版质量的图片
plt.savefig('ablation_study.png', dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()
