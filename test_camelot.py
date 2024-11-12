import camelot
import os
import pandas as pd

# 调用函数并传入PDF文件路径
pdf_path = "C:\\Users\\weiji\\Documents\\projects\\camelot\\pdfs\\格力半年度报告.PDF"

# 读取PDF文件中的表格
tables = camelot.read_pdf(
    pdf_path, pages='28', 
    bottom_threshold=130, top_threshold=90, 
    number_of_hearder_rows=6,
    strip_text='\n',  # 去除换行符
)

print(f"Total tables extracted: {len(tables)}")

# 输出解析的表格内容并保存非空表格为 CSV 文件
output_dir = os.path.join(os.getcwd(), "output_test")
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

def is_table_empty(df):
    """
    检查 DataFrame 是否为空，或者只有无意义的分隔符（如空格或逗号）。
    """
    # 检查是否为空表（所有值为空或仅包含标点符号）
    if df.empty:
        return True
    # 检查每个单元格是否都为空或仅包含标点符号
    return df.map(lambda x: str(x).strip() in ['', ',', '.']).all().all()

# 计数器：用于统计非空表的数量
non_empty_table_count = 0

# 遍历所有表格并保存非空表格为 CSV
for i, table in enumerate(tables):
    df = table.df  # 获取表格的 DataFrame

    # 打印表格信息和内容
    print(f"\nTable {i + 1}:")
    print(f"Shape: {df.shape}")
    print(f"Parsing Report: {table.parsing_report}")
    print(df)

    # 判断表格是否为空，如果不为空则保存为 CSV 文件，并增加计数器
    if not is_table_empty(df):
        output_file = os.path.join(output_dir, f"{pdf_name}_table_{i + 1}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Saved: {output_file}")
        non_empty_table_count += 1  # 记录非空表格数量
    else:
        print(f"Skipped empty table {i + 1}.")

# 打印非空表格的数量
print(f"\nTotal non-empty tables saved: {non_empty_table_count}")
