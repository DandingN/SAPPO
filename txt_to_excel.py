import pandas as pd
import os
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Side

# 文件路径定义
excel_path = 'excel_results/result.xlsx'
txt_folder = 'result'

# 需要排除的列
exclude_columns = {'Instance', 'Runtime', 'Object'}

# 1. 读取原始Excel中的"原始结果"
df_original = pd.read_excel(excel_path, sheet_name='原始结果')
df = df_original.copy()

# 获取现有列名集合
existing_columns = set(df.columns)

# 2. 读取所有txt文件并添加为新列，但仅当该列不存在于现有列中时
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith('.txt'):
        model_fullname = os.path.splitext(txt_file)[0]

        # 提取简化的模型名称
        parts = model_fullname.split('_')

        if len(parts) < 4:
            raise ValueError(f"文件名 {txt_file} 格式不支持，请确保包含类似 202505xxx_model_xxxxxx_steps 的结构")

        date_part = parts[0][6:]       # 如 "047" 或 "0410"
        step_part = parts[-2]          # 如 "2216500"

        simplified_name = f"{date_part}_{step_part}"

        if simplified_name in existing_columns:
            print(f"列 {simplified_name} 已存在，跳过对应txt文件的处理：{txt_file}")
            continue

        file_path = os.path.join(txt_folder, txt_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            values = [float(line.strip()) for line in f.readlines()]

        # 确保长度一致
        if len(values) != len(df):
            raise ValueError(f"文件 {txt_file} 的行数与原始数据不一致，请检查！")

        df[simplified_name] = values

# 写回 "原始结果" sheet（覆盖）
with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='原始结果', index=False)
print("✅ 已成功将所有模型结果写入 Excel 的【原始结果】sheet。")


# 计算每隔100个数据的平均值（排除 Instance/Runtime/Object 列）
def calculate_averages_every_n(column_data, n=100):
    return [np.mean(column_data[i:i + n]) for i in range(0, len(column_data), n)]


summary_dict = {}

# 构建统计分数 DataFrame
cols_to_process = [col for col in df.columns if col not in exclude_columns]
for col in cols_to_process:
    summary_dict[col] = calculate_averages_every_n(df[col].values)

# 添加分段标签
num_segments = (len(df) + 99) // 100  # 向上取整
summary_dict['Segment'] = [str((i + 1) * 100) for i in range(num_segments)]

# 将 Segment 放到第一列
summary_df = pd.DataFrame(summary_dict)
summary_df = summary_df[['Segment'] + cols_to_process]

# 四舍五入保留4位小数
summary_df[cols_to_process] = summary_df[cols_to_process].round(4)

# 创建评估得分 sheet：模型得分 / Max 得分
evaluation_df = summary_df.copy()

# 获取 Max 列的值
max_values = evaluation_df['Max'].values

# 对非 Segment 和 Max 的列进行占比计算
for col in evaluation_df.columns:
    if col not in ['Segment', 'Max']:
        evaluation_df[col] = (evaluation_df[col] / max_values).round(4)


# 给统计分数和评估得分添加平均值行
def add_average_row(df: pd.DataFrame, segment_col='Segment'):
    avg_row = {'Segment': 'Average'}
    for col in df.columns:
        if col == segment_col:
            continue
        avg_row[col] = round(df[col].mean(), 4)
    return df._append(avg_row, ignore_index=True)


summary_df = add_average_row(summary_df)
evaluation_df = add_average_row(evaluation_df)

# ===== 新增：按列名排序 =====
def sort_columns(df, keep_first=None):
    if keep_first is None:
        keep_first = []
    other_cols = sorted([col for col in df.columns if col not in keep_first])
    return df[keep_first + other_cols]

# 排序各个DataFrame的列
df = sort_columns(df, keep_first=['Instance'])         # 原始结果保留 Instance 在前
summary_df = sort_columns(summary_df, keep_first=['Segment'])  # 统计分数保留 Segment 在前
evaluation_df = sort_columns(evaluation_df, keep_first=['Segment'])  # 评估得分保留 Segment 在前

# 写入 Excel：原始结果 + 统计分数 + 评估得分，并设置样式
with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    summary_df.to_excel(writer, sheet_name='统计分数', index=False)
    evaluation_df.to_excel(writer, sheet_name='评估得分', index=False)

# 加载工作簿并对指定表格应用样式
wb = load_workbook(excel_path)


def apply_styles(ws):
    for column_cells in ws.columns:
        length = max(len(str(cell.value)) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length + 2
        for cell in column_cells:
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = Border(left=Side(style='thin'),
                                 right=Side(style='thin'),
                                 top=Side(style='thin'),
                                 bottom=Side(style='thin'))


# 应用样式到所有需要格式化的表单
apply_styles(wb['原始结果'])
apply_styles(wb['统计分数'])
apply_styles(wb['评估得分'])

wb.save(excel_path)

print("✅ 已成功生成【原始结果】、【统计分数】和【评估得分】三个sheet，并应用了适当的格式。")
