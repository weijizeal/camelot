import camelot
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_tables_from_pdf(pdf_path, output_dir):
    """
    从 PDF 中提取表格并保存为 CSV 文件到指定的输出目录。
    """
    print(f"Processing {pdf_path}...")
    try:
        # 提取 PDF 中的表格
        tables = camelot.read_pdf(pdf_path, pages='all', bottom_threshold=130, top_threshold=90, number_of_hearder_rows=6)
        
        print(f"[{pdf_path}] Total tables extracted: {len(tables)}")

        # 遍历提取的表格并保存为 CSV 文件
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, table in enumerate(tables):
            output_file = os.path.join(output_dir, f"{pdf_name}_table_{i + 1}.csv")
            table.to_csv(output_file)

    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")

def extract_from_folder(folder_path, output_dir, max_workers=4):
    """
    从指定文件夹中提取所有 PDF 的表格数据，并保存到一个固定输出目录。
    """
    # 获取文件夹中的所有 PDF 文件
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith( '.PDF') or f.endswith( '.pdf')]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 使用线程池并行提取表格
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_tables_from_pdf, pdf, output_dir): pdf for pdf in pdf_files}

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
    folder_path = os.path.join(os.getcwd(), "pdfs")

    # 当前目录下的 'output_tables' 文件夹
    output_dir = os.path.join(os.getcwd(), "output_tables")

    extract_from_folder(folder_path, output_dir, max_workers=8)
    print("All PDFs processed.")
