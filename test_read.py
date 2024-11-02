import os
import json
import pandas as pd

def is_table_empty(df):
    """
    检查 DataFrame 是否为空，或者只有无意义的分隔符（如空格或逗号）。
    """
    # 检查是否为空表（所有值为空或仅包含标点符号）
    if df.empty:
        return True
    # 检查每个单元格是否都为空或仅包含标点符号
    return df.map(lambda x: str(x).strip() in ['', ',', '.']).all().all()

def read_all_tables_from_json(file_path):
    """
    从 JSON 文件中读取所有表格的标题和数据。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_tables_data = json.load(f)

        # 遍历所有表格，打印标题和数据
        for table_data in all_tables_data:
            table_number = table_data["table_number"]
            candidate_title = "\n".join(table_data["candidate_title"])  # 还原多行标题
            df = pd.DataFrame(table_data["data"])  # 将数据还原为 DataFrame

            print(f"\nTable {table_number}:")
            print(f"candidate_title:\n{candidate_title}")
            title = extract_table_title(candidate_title,df)
            print(f"title:\n{title}")
            # print(f"\nData:\n{df}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# 生成 PPT 综合内容
def extract_table_title(candidate_title,df):
    """
    提取表格标题
    """
    try:
        # 将内容与图片描述结合后，传递给大模型
        prompt = f"""
        目标：你需要从表格上方的文字中，提取出表格的标题，并且用前几行数据作为表格的参照。
        步骤：从后边的文字往前找，找到第一个可以作为表格标题的文字，输出这个文字作为表格标题。
        注意：标题要用表格上方文字中的原句

        表格上方文字：
        {candidate_title}
        
        表格前几行数据：
        {df.head(10).to_string(index=False)}

        输出表格标题的时候不要附加标记或者有多余解释。直接输出提取的表格标题即可。
        """

        from openai import OpenAI

        client = OpenAI(api_key="sk-e68a6a508fae4a3aa10963376ded8544", base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", 'content': prompt},
            ],
            stream=False
        )


        # 获取API返回的内容
        extracted_content = response.choices[0].message.content
        print(f"提取的表格标题为：{extracted_content}")
        return extracted_content

    except Exception as e:
        print(f"错误：不能获取图片描述提取修改内容，错误为：{e}")
        return f"错误：不能获取图片描述提取修改内容，错误为：{e}"

if __name__ == '__main__':
    # 设置要读取的文件路径
    output_dir = os.path.join(os.getcwd(), "output_test")
    pdf_name = "东方电子2024三季度报告"  # 根据你的PDF文件名调整
    json_file = os.path.join(output_dir, f"{pdf_name}_all_tables.json")

    # 检查文件是否存在
    if os.path.exists(json_file):
        read_all_tables_from_json(json_file)
    else:
        print(f"File {json_file} does not exist.")
