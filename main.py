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
    print(table._bbox)  # 打印表格的边界框
    
tables.export("tmp_csv")
