import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
plt.rcParams['mathtext.fontset'] = 'stix'       # 数学字体兼容

# 1. 加载所有 .npy 文件（6个）
npy_files = [f'data/npy/{i}.npy' for i in range(2, 8)]
data_list = [np.load(f) for f in npy_files]

# 2. 检查所有矩阵的形状是否一致
shapes = [data.shape for data in data_list]
assert all(shape == shapes[0] for shape in shapes), "所有矩阵的尺寸必须一致"

# 3. 确定全局的值范围
global_min = min(np.min(data) for data in data_list)
global_max = max(np.max(data) for data in data_list)

# 4. 创建 2x3 子图布局（启用自动布局）
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), constrained_layout=True)
axes = axes.ravel()  # 展平为一维数组

# 5. 绘制每个子图（无标题）
for ax, data in zip(axes, data_list):
    im = ax.imshow(data, cmap='gray_r', vmin=global_min, vmax=global_max)

    # 隐藏刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])

    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')  # 边框颜色
        spine.set_linewidth(1.5)  # 边框线宽

# 6. 添加全局颜色条（自动布局会自动调整位置）
cbar = fig.colorbar(im, ax=axes, label='POS', shrink=0.935)
cbar.ax.tick_params(labelsize=28)  # 设置刻度字体大小为14
cbar.set_label('POS', fontdict={'size': 28})  # 设置标签字体大小和加粗


# 7. 手动调整子图间距（重点：减小行间距）
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.1)

# 8. 保存和显示图像
# plt.savefig('merged_6_subplots_less_space.png', dpi=300, bbox_inches='tight')
plt.show()
