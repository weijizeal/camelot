# -*- coding: utf-8 -*-

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader, PdfWriter

from .core import TableList
from .parsers import Stream, Lattice
from .utils import (
    TemporaryDirectory,
    get_page_layout,
    get_text_objects,
    get_rotation,
    is_url,
    download_url,
)
import pandas as pd
import re 

class PDFHandler(object):
    """Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.

    """

    def __init__(self, filepath, pages="1", password=None):
        if is_url(filepath):
            filepath = download_url(filepath)
        self.filepath = filepath
        if not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""
        else:
            self.password = password
            if sys.version_info[0] < 3:
                self.password = self.password.encode("ascii")
        self.pages = self._get_pages(self.filepath, pages)

    def _get_pages(self, filepath, pages):
        """Converts pages string to list of ints.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.

        Returns
        -------
        P : list
            List of int page numbers.

        """
        page_numbers = []
        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            instream = open(filepath, "rb")
            infile = PdfReader(instream, strict=False)
            if infile.is_encrypted:
                infile.decrypt(self.password)
            if pages == "all":
                page_numbers.append({"start": 1, "end": len(infile.pages)})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = infile.getNumPages()
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})
            instream.close()
        P = []
        for p in page_numbers:
            P.extend(range(p["start"], p["end"] + 1))
        return sorted(set(P))

    def _save_page(self, filepath, page, temp):
        """Saves specified page from PDF into a temporary directory.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        page : int
            Page number.
        temp : str
            Tmp directory.

        """
        with open(filepath, "rb") as fileobj:
            infile = PdfReader(fileobj, strict=False)
            if infile.is_encrypted:
                infile.decrypt(self.password)
            fpath = os.path.join(temp, f"page-{page}.pdf")
            froot, fext = os.path.splitext(fpath)
            p = infile.pages[page - 1]
            outfile = PdfWriter()
            outfile.add_page(p)
            with open(fpath, "wb") as f:
                outfile.write(f)
            layout, dim = get_page_layout(fpath)
            # fix rotated PDF
            chars = get_text_objects(layout, ltype="char")
            horizontal_text = get_text_objects(layout, ltype="horizontal_text")
            vertical_text = get_text_objects(layout, ltype="vertical_text")
            rotation = get_rotation(chars, horizontal_text, vertical_text)
            if rotation != "":
                fpath_new = "".join([froot.replace("page", "p"), "_rotated", fext])
                os.rename(fpath, fpath_new)
                instream = open(fpath_new, "rb")
                infile = PdfReader(instream, strict=False)
                if infile.is_encrypted:
                    infile.decrypt(self.password)
                outfile = PdfWriter()
                p = infile.getPage(0)
                if rotation == "anticlockwise":
                    p.rotateClockwise(90)
                elif rotation == "clockwise":
                    p.rotateCounterClockwise(90)
                outfile.add_page(p)
                with open(fpath, "wb") as f:
                    outfile.write(f)
                instream.close()

    #      (0, 842)  -------------------->  (595.2, 842)  
    #     |                                      |
    #     |                                      |
    #     |                                      |
    #     |                                      |
    #  (0, 0)  -----------------------> (595.2, 0)  
    
    # 需要注意的是上一个页面的底部到当前页面的顶部，如果这个距离小于一个阈值，那么这两个表格应该合并。
    # 1.计算出顶部和底部空白所占的比例 如 顶部 77/871 底部 84/871
    # 2.计算出顶部y轴和底部y轴的坐标范围 如 顶部842 - 842 * 77/871 = 767 底部 842 * 84/871 = 81
    # 3.顶部可以下移，底部可以上移，将顶部数字变小，底部数字变大 如 顶部 767 - 17 = 750 底部 81 + 10 = 91
    # 4.得出顶部和顶部的y的范围 如 pre_y <= 91 和 底部 cur_y >= 757

    def check_need_merge(self,p, pre_page_last_table, cur_table, **kwargs):
        pre_bbox = pre_page_last_table._bbox
        cur_table_bbox = cur_table._bbox
        
        pre_y_bottom = pre_bbox[1]
        cur_y_top = cur_table_bbox[3]
        
        pdf_height = self.page_height(p)
        
        bottom_threshold = kwargs.get('bottom_threshold', 100)
        top_threshold = pdf_height - kwargs.get('top_threshold', 90)
        
        if cur_table.page - pre_page_last_table.page > 1:
            return False
        
        if pre_y_bottom <= bottom_threshold and cur_y_top >= top_threshold:
            return True
        else:
            return False

    def page_height(self, p):
        layout, dimensions = get_page_layout(p)
        pdf_width, pdf_height = dimensions
        return pdf_height
    
    def parse(
        self, flavor="lattice", suppress_stdout=False, layout_kwargs={}, **kwargs
    ):
        """Extracts tables by calling parser.get_tables on all single
        page PDFs in threads, then performs merging and sets titles.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice' or 'stream').
        suppress_stdout : str (default: False)
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of pdfminer.layout.LAParams kwargs.
        kwargs : dict
            Additional options for parsing.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.
        """
        tables = []
        page_tb_list_list = []  # Store tables for each page
        is_first_tab_merge = []  # Store if the first table on each page was merged

        with TemporaryDirectory() as tempdir:
            # Save pages to temporary directory
            for p in self.pages:
                self._save_page(self.filepath, p, tempdir)

            pages = [os.path.join(tempdir, f"page-{p}.pdf") for p in self.pages]
            
            import time 
            start_time = time.perf_counter()  # 开始计时
            # 多线程处理页面表格
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {
                    executor.submit(
                        self.extract_tables_thread, kwargs, flavor, p, suppress_stdout, layout_kwargs
                    ): idx
                    for idx, p in enumerate(pages)
                }

                # 收集线程结果
                page_tb_list_list = [None] * len(pages)
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        tables_extracted = future.result()
                        page_tb_list_list[idx] = tables_extracted
                    except Exception as e:
                        print(f"Error processing page {idx + 1}: {e}")
            part1_time = time.perf_counter() - start_time  # 结束计时并计算耗时
            file_name = self.filepath.split('/')[-1]
            print(f"{file_name} - Part 1 多线程处理页面表格耗时: {part1_time:.2f} seconds")
            
            start_time = time.perf_counter()  # 开始计时
            # 处理提取的表格并且合并
            pre_page_last_table = None
            last_in_table = None
            for idx, current_page_tables in enumerate(page_tb_list_list):
                if not current_page_tables:
                    is_first_tab_merge.append(False)
                    continue
                
                if len(tables) > 0:
                    last_in_table = tables[-1]
                
                tables.extend(current_page_tables)
                current_page_first_table = current_page_tables[0]
                if pre_page_last_table:
                    # 检查现在的页面是否需要合并
                    first_merge = self.merge_cross_table(
                        pages[idx], tables, pre_page_last_table, last_in_table, current_page_first_table, **kwargs
                    )
                    is_first_tab_merge.append(first_merge)
                else:
                    is_first_tab_merge.append(False)

                pre_page_last_table = current_page_tables[-1]
            part2_time = time.perf_counter() - start_time  # 结束计时并计算耗时
            print(f"{file_name} - Part 2 处理提取的表格并且合并耗时: {part2_time:.2f} seconds")
            
            start_time = time.perf_counter()  # 开始计时
            # 多线程处理表格标题设置
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {
                    executor.submit(
                        self.set_title_for_table, kwargs, pages[idx], pages,
                        page_tb_list_list, idx, is_first_tab_merge[idx], current_page_tables
                    ): idx
                    for idx, current_page_tables in enumerate(page_tb_list_list) if current_page_tables
                }

                # 等待所有线程完成
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()  # 捕获线程中的异常
                    except Exception as e:
                        print(f"Error setting title for page {idx + 1}: {e}")
            part3_time = time.perf_counter() - start_time  # 结束计时并计算耗时
            print(f"{file_name} - Part 3 多线程处理表格标题设置耗时: {part3_time:.2f} seconds")
        
        return TableList(sorted(tables))

    def extract_tables_thread(self, kwargs,flavor, page, suppress_stdout, layout_kwargs):
        """Extract tables from a single page with thread-safe lock."""
        parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
        
        return parser.extract_tables(
            page, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
        )

    def set_title_for_table(self, kwargs, p, pages, page_tb_list_list, idx, is_frist_table_merge, t):
        if is_frist_table_merge == False:
            page_height = self.page_height(p)
            top_threshold = page_height - kwargs.get('top_threshold', 90)
            if t[0]._bbox[3] >= top_threshold and idx > 0: # 表格的顶部大于阈值
                        # 判断上一页有没有表格，如果有那么获取它的页面底部y坐标否则0
                pre_page_last_table_bottom_y = page_tb_list_list[idx-1][-1]._bbox[1] if len(page_tb_list_list[idx-1]) > 0 else 0
                        # 当前表格顶部归零，即从上一页底部网上找，找到距离最近的标题，但是不超过上一页最后一个表的底部y坐标 10回调值
                t[0].candidate_title = self.find_table_title(0, pages[idx-1],pre_page_last_table_bottom_y,kwargs)
            else:
                        # 在当表格y坐标网上找，找到距离最近的标题 当前表格顶部y t[0]._bbox[3] 找到空白部分之下 10是回调值
                t[0].candidate_title = self.find_table_title(t[0]._bbox[3], p,page_height,kwargs)
                    
        if len(t) > 1:
                    # 当前也页面之后的每个表的标题，在上个表格底部之下的区域寻找
            for i in range(1, len(t)):
                pre_page_last_table_bottom_y = t[i-1]._bbox[1]
                        # 当前表格顶部坐标 t[i]._bbox[3]之上，上个表格底部坐标之下
                t[i].candidate_title = self.find_table_title(t[i]._bbox[3], p,pre_page_last_table_bottom_y,kwargs)

    def _generate_layout(self, filename):
        layout, dimensions = get_page_layout(filename)
        horizontal_text = get_text_objects(layout, ltype="horizontal_text")
        vertical_text = get_text_objects(layout, ltype="vertical_text")
        return horizontal_text, vertical_text

    
    def find_table_title(self, self_table_top_y, page, pre_page_last_table_bottom_y,kwargs):
        number_of_hearder_rows = int(kwargs.get("number_of_hearder_rows", 5))
        # print("number_of_hearder_rows",number_of_hearder_rows)
        h_txt,v_txt = self._generate_layout(page)
        
        distance_list = []
        for obj in h_txt:
            if pre_page_last_table_bottom_y > obj.bbox[1] > self_table_top_y and obj.get_text().strip() != "":
                distance = abs(obj.bbox[1] - self_table_top_y)
                distance_list.append((obj, distance))
                
        distance_list = sorted(distance_list, key=lambda x: x[1])
        closest_titles = distance_list[:number_of_hearder_rows]
        closest_titles.reverse()
        title_arr = [obj[0].get_text().strip() for obj in closest_titles]
        title = " ".join(title_arr)
        return title
        
    def merge_cross_table(self,p, tables, pre_page_last_table, last_in_table, current_page_first_table, **kwargs):
        need_merge = self.check_need_merge(p, pre_page_last_table, current_page_first_table, **kwargs)
        if need_merge == True: # 合并表格
            tables.remove(current_page_first_table)
            for i in range(len(tables)):
                if tables[i] == last_in_table:
                    tables[i].df = pd.concat([last_in_table.df, current_page_first_table.df])
                    break
            return True
        else:
            return False
