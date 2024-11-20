import camelot
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def is_table_empty(df):
    """
    检查 DataFrame 是否为空，或者只有无意义的分隔符（如空格或逗号）。
    """
    # 检查是否为空表（所有值为空或仅包含标点符号）
    if df.empty:
        return True
    # 检查每个单元格是否都为空或仅包含标点符号
    return df.map(lambda x: str(x).strip() in ['', ',', '.']).all().all()

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

def extract_tables_from_pdf(pdf_path, output_dir,output_titles_dir):
    """
    从 PDF 中提取表格并保存为 CSV 文件到指定的输出目录。
    """
    print(f"Processing {pdf_path}...")
    try:
        # 提取 PDF 中的表格
        tables = camelot.read_pdf(pdf_path, pages='all', bottom_threshold=130, top_threshold=90,number_of_hearder_rows=6,strip_text='\n ')
        print(f"[{pdf_path}] Total tables extracted: {len(tables)}")

        # 遍历提取的表格并保存为 CSV 文件
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        title_record = []
        count = 1
        for table in tables:
            if not is_table_empty(table.df):
                title = extract_table_title(table.candidate_title,table.df)
                title_record.append({
                    "candidate_title":table.candidate_title,
                    "title":title
                })
                output_file = os.path.join(output_dir, f"{pdf_name}_{count}_{title}.csv")
                count = count + 1
                table.df.to_csv(output_file, index=False, header=False, encoding='utf-8-sig')

        output_titles_file = os.path.join(output_titles_dir, f"{pdf_name}.json")
        with open(output_titles_file, 'w') as f:
            json.dump(title_record, f, ensure_ascii=False, indent=4)
        print(f"Saved tables from {pdf_path} to {output_file}")
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")

def extract_from_folder(folder_path, output_dir, output_titles_dir ,max_workers=4):
    """
    从指定文件夹中提取所有 PDF 的表格数据，并保存到一个固定输出目录。
    """
    # 获取文件夹中的所有 PDF 文件
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith( '.PDF') or f.endswith( '.pdf')]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs( output_titles_dir, exist_ok=True)
    
    # 使用线程池并行提取表格
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_tables_from_pdf, pdf, output_dir, output_titles_dir): pdf for pdf in pdf_files}

        # 等待所有任务完成，并处理结果
        for future in as_completed(futures):
            pdf = futures[future]
            try:
                future.result()
                print(f"Completed: {pdf}")
            except Exception as e:
                print(f"Error processing {pdf}: {e}")

# 示例调用：从文件夹中提取表格数据，并保存到固定输出目录
if __name__ == "__main__":
    print("Start PDFs processed.")
    # 当前目录下的 'pdfs' 文件夹
    folder_path = os.path.join(os.getcwd(), "pdf_三季度")

    # 当前目录下的 'output_tables' 文件夹
    output_dir = os.path.join(os.getcwd(), "output_三季度")

    # 输出提取文件标题和候选标题的文件
    output_titles_dir = os.path.join(os.getcwd(), "output_titles")
    
    extract_from_folder(folder_path, output_dir,output_titles_dir , max_workers=8)
    print("All PDFs processed.")
