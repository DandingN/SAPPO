#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/8 下午7:53
# @function: the script is used to do something.
# @version : V1
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 设置Matplotlib使用SimSun字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 调整全局字体大小
plt.rcParams.update({'font.size': 16})

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
    "本文模型": df_best_divided,
    "无观测空间模块": df_without_obs_divided,
    "无LSTM模块": df_without_lstm_divided,
    "无阈值切换模块": df_without_threshold_divided
})

# 将DataFrame转换为长格式以便Seaborn处理
results_melted = results_df.melt(var_name="模型类型", value_name="性能得分")

# 绘制箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(
    x="模型类型",
    y="性能得分",
    data=results_melted,
    palette='gray',
    fliersize=0,
    width=0.6,
    boxprops=dict(facecolor='none', edgecolor='.3'),  # 不填充颜色，只保留边框
    whiskerprops=dict(color='.3'),
    capprops=dict(color='.3'),
    medianprops=dict(color='.3'),
    showfliers=False  # 不显示异常值
)
sns.stripplot(x="模型类型", y="性能得分", data=results_melted, color=".3", dodge=True, size=3)  # 添加数据点

plt.title("消融实验")
plt.ylabel("性能得分(%)")
plt.xlabel("模型类型")
plt.tight_layout()
plt.show()

print(df_best)



