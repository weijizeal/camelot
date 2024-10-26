import camelot
import openai
import pandas as pd

from openai import OpenAI
client = OpenAI(api_key="sk-e68a6a508fae4a3aa10963376ded8544", base_url="https://api.deepseek.com")

# 提取PDF中的表格
pdf_path = "优彩环保三季度报告.PDF"
tables = camelot.read_pdf(pdf_path, pages='all', bottom_threshold=100, top_threshold=90)

print(f"Total tables extracted: {len(tables)}")

# 遍历提取的表格并输出解析报告
for i, table in enumerate(tables):
    print(f"\nTable {i + 1}:")
    print(f"Shape: {table.shape}")
    print(f"Parsing Report: {table.parsing_report}")
    print(table.df)  # 打印表格的内容
tables.export("tmp_csv")
# # 初始化合并表格的存储容器
# merged_tables = []
# current_table = tables[0].df  # 将第一个表格作为初始表

# # 遍历并检查表格是否需要合并
# for i in range(1, len(tables)):
#     table = tables[i].df

#     # 调用大模型API判断是否需要合并
#     need_merge = check_need_merge(current_table.head().to_string(), table.head().to_string())

#     if need_merge:
#         # 如果需要合并，则将表格拼接
#         current_table = pd.concat([current_table, table.iloc[1:]], ignore_index=True)
#     else:
#         # 保存当前表格，并开始新的合并表
#         merged_tables.append(current_table)
#         current_table = table

# # 保存最后一个表格
# merged_tables.append(current_table)

# # 打印合并后的表格信息
# for idx, merged_table in enumerate(merged_tables):
#     print(f"\nMerged Table {idx + 1}:")
#     print(merged_table)
