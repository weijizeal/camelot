import os
import pandas as pd
from collections import defaultdict

def group_csv_by_prefix(input_folder):
    """
    遍历指定目录中的所有 CSV 文件，并按文件名前缀分组。
    返回一个字典，键是前缀，值是属于该前缀的 CSV 文件列表。
    """
    groups = defaultdict(list)  # 用于存储前缀与对应文件的映射关系

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            prefix = "_".join(file.split("_")[:-2])  # 提取前缀
            groups[prefix].append(file)  # 将文件归类到对应的前缀列表中

    return groups

def save_csv_to_excel(input_folder, output_folder):
    """
    将按前缀分组的 CSV 文件分别保存到单独的 Excel 文件中。
    """
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（若不存在）

    # 获取按前缀分组的 CSV 文件
    grouped_files = group_csv_by_prefix(input_folder)

    for prefix, files in grouped_files.items():
        excel_path = os.path.join(output_folder, f"{prefix}.xlsx")  # 输出 Excel 文件路径
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for file in files:
                csv_path = os.path.join(input_folder, file)
                df = pd.read_csv(csv_path)  # 读取 CSV 文件为 DataFrame
                sheet_name = os.path.splitext(file)[0]  # 工作表名
                df.to_excel(writer, sheet_name=sheet_name, index=False)  # 保存到 Excel

        print(f"Saved {excel_path}")

# 示例调用
if __name__ == "__main__":
    input_folder = "output_tables"  # 存放 CSV 文件的目录
    output_folder = "excel_outputs"  # 存放 Excel 文件的目录
    save_csv_to_excel(input_folder, output_folder)
