# -*- coding: utf-8 -*-

import os
import sys

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
PAGE_HEIGHT = 842

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

    def check_need_merge(self,pre_table, cur_table, **kwargs):
        pre_bbox = pre_table._bbox
        cur_table_bbox = cur_table._bbox
        
        pre_y_bottom = pre_bbox[1]
        cur_y_top = cur_table_bbox[3]
        # 打印这些值
        print(f"pre_y_bottom: {pre_y_bottom}")
        print(f"cur_y_top: {cur_y_top}")
        
        bottom_threshold = kwargs.get('bottom_threshold', 100)
        top_threshold = PAGE_HEIGHT - kwargs.get('top_threshold', 90)
        print(f"bottom_threshold: {bottom_threshold}")
        print(f"top_threshold: {top_threshold}")
        
        if pre_y_bottom <= bottom_threshold and cur_y_top >= top_threshold:
            return True
        else:
            return False

    def parse(
        self, flavor="lattice", suppress_stdout=False, layout_kwargs={}, **kwargs
    ):
        """Extracts tables by calling parser.get_tables on all single
        page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice' or 'stream').
            Lattice is used by default.
        suppress_stdout : str (default: False)
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        tables = []
        with TemporaryDirectory() as tempdir:
            for p in self.pages:
                self._save_page(self.filepath, p, tempdir)
            pages = [os.path.join(tempdir, f"page-{p}.pdf") for p in self.pages]
            parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
            
            pre_table=None
            last_table=None
            for p in pages:
                if len(tables) > 0:
                    last_table = tables[-1]
                    
                t = parser.extract_tables(
                    p, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
                )
                tables.extend(t)
                if len(t) <= 0:
                    continue
                
                current_table = t[0]
                if pre_table:
                    if pre_table.df.shape[1] == current_table.df.shape[1]:
                        self.merge_cross_table(tables, pre_table, last_table, current_table, **kwargs)
                        
                pre_table = t[-1]

        return TableList(sorted(tables))

    def merge_cross_table(self, tables, pre_table, last_table, current_table, **kwargs):
        need_merge = self.check_need_merge(pre_table, current_table, **kwargs)
        if need_merge == True: # 合并表格
            tables.remove(current_table)
            for i in range(len(tables)):
                if tables[i] == last_table:
                    tables[i].df = pd.concat([last_table.df, current_table.df])
                    break
