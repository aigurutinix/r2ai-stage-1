import re

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import docx
from docx.document import Document as _Docx
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell, _Row
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

from FisReader.document import Article
from FisReader.document import LAW_TYPES_LIST, Article
from FisReader.document import Document
from FisReader.reader.pdf_reader import PDFReader
from LegalTreeConstruct.document_constructor import DocumentConstructor
from GeneralTreeConstruct.tree_construct import StructuredDocumentTreeBuilder
from docx2python import docx2python
from docx2python.depth_collector import Par
from docx2python.iterators import is_tbl, iter_at_depth, iter_tables
from html_to_markdown import convert_to_markdown


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def w_tag(local_name: str) -> str:
    return f"{{{W_NS}}}{local_name}"


def w_attr(element, name: str, default: Optional[str] = None) -> Optional[str]:
    if element is None:
        return default
    return element.get(w_tag(name), default)


def int_to_alpha(value: int, upper: bool = False) -> str:
    if value <= 0:
        return str(value)

    chars = []
    while value > 0:
        value -= 1
        chars.append(chr(ord("A" if upper else "a") + (value % 26)))
        value //= 26
    return "".join(reversed(chars))


def int_to_roman(value: int, upper: bool = False) -> str:
    if value <= 0:
        return str(value)

    table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = []
    for number, numeral in table:
        while value >= number:
            result.append(numeral)
            value -= number
    roman = "".join(result)
    return roman if upper else roman.lower()


def format_number(value: int, fmt: str) -> str:
    if fmt == "decimal":
        return str(value)
    if fmt == "decimalZero":
        return f"{value:02d}"
    if fmt == "lowerLetter":
        return int_to_alpha(value, upper=False)
    if fmt == "upperLetter":
        return int_to_alpha(value, upper=True)
    if fmt == "lowerRoman":
        return int_to_roman(value, upper=False)
    if fmt == "upperRoman":
        return int_to_roman(value, upper=True)
    return str(value)


@dataclass
class LevelDef:
    ilvl: int
    start: int = 1
    num_fmt: str = "decimal"
    lvl_text: str = ""


@dataclass
class NumberingDef:
    abstract_num_id: str
    levels: dict[int, LevelDef] = field(default_factory=dict)


@dataclass
class NumInstance:
    num_id: str
    abstract_num_id: str
    start_overrides: dict[int, int] = field(default_factory=dict)


class TableArrayTools():
    @staticmethod
    def validate_table_array(anything: Union[list[list[str]], Any]) -> bool:
        """ Check if any input is a valid table array """
        if not isinstance(anything, list):
            return False

        for row in anything:
            if not isinstance(row, list):
                return False

        for row in anything:
            for cell in row:
                if not isinstance(cell, str):
                    return False
        return True

    @staticmethod
    def merge_table_arrays(upper: list[list[str]], lower: list[list[str]]) -> list[list[str]]:
        # it might be simple at first, but it's not
        # it shall grows with time
        return upper + lower
    
    @staticmethod
    def is_mergeable(upper: list[list[str]], lower: list[list[str]]) -> bool:
        # it might be simple at first, but it's not
        # it shall grows with time
        if __class__.validate_table_array(upper) and __class__.validate_table_array(lower):
            return len(upper[0]) == len(lower[0])
        return False
    
    @staticmethod
    def make_html_table_str(table_array: list[list[str]]) -> str:
        html_table_str = "<table>"
        for row in table_array:
            html_row_str = "<tr>"
            for cell in row:
                html_row_str += "<td>" + cell.replace("\n", "<br>") + "</td>"
            html_row_str += "</tr>"
            html_table_str += html_row_str
        html_table_str += "</table>"
        return html_table_str
    
    @staticmethod
    def docx2python_to_table_array(table: list[list[list[Par]]], line_break: str = "<br>") -> list[list[str]]:
        table_array = []
        for row in table:
            row_array = []
            for cell in row:
                cell_str = line_break.join(["\n".join(p.run_strings) for p in cell])
                row_array.append(cell_str)
            table_array.append(row_array)
        return table_array

    @staticmethod
    def docx_to_table_array(table: Table, line_break: str = "<br>") -> list[list[str]]:
        table_array = []
        for row in table.rows:
            row_array = []
            for cell in row.cells:
                cell_str = __class__._docx_cell_to_text(cell, line_break)
                row_array.append(cell_str)
            table_array.append(row_array)
        return table_array
    
    @staticmethod
    def _docx_cell_to_text(cell: _Cell, line_break: str = "<br>") -> str:
        elements = cell._element.xpath(".//w:t | .//w:br | .//w:p")
        texts = []

        for element in elements:
            if element.tag.endswith("br"):
                texts.append(line_break)
            elif element.tag.endswith("p"):
                texts.append(" ")
            else:
                texts.append(element.text)

        text = "".join(texts)
        return line_break.join(" ".join(line.split()) for line in text.split(line_break))


class WordReader:
    def __init__(self) -> None:
        self.document_constructor = DocumentConstructor()

    def _parse_numbering_element(self, root) -> tuple[dict[str, NumberingDef], dict[str, NumInstance]]:
        abstract_defs = {}
        num_instances = {}

        for abstract in root.findall(f"./{w_tag('abstractNum')}"):
            abstract_id = w_attr(abstract, "abstractNumId")
            if abstract_id is None:
                continue

            numbering_def = NumberingDef(abstract_num_id=abstract_id)
            for lvl in abstract.findall(f"./{w_tag('lvl')}"):
                ilvl_text = w_attr(lvl, "ilvl")
                if ilvl_text is None:
                    continue

                start_elem = lvl.find(f"./{w_tag('start')}")
                num_fmt_elem = lvl.find(f"./{w_tag('numFmt')}")
                lvl_text_elem = lvl.find(f"./{w_tag('lvlText')}")

                level = LevelDef(
                    ilvl=int(ilvl_text),
                    start=int(w_attr(start_elem, "val", "1")),
                    num_fmt=w_attr(num_fmt_elem, "val", "decimal"),
                    lvl_text=w_attr(lvl_text_elem, "val", ""),
                )
                numbering_def.levels[level.ilvl] = level

            abstract_defs[abstract_id] = numbering_def

        for num in root.findall(f"./{w_tag('num')}"):
            num_id = w_attr(num, "numId")
            abstract_elem = num.find(f"./{w_tag('abstractNumId')}")
            abstract_num_id = w_attr(abstract_elem, "val")
            if num_id is None or abstract_num_id is None:
                continue

            instance = NumInstance(num_id=num_id, abstract_num_id=abstract_num_id)
            for override in num.findall(f"./{w_tag('lvlOverride')}"):
                ilvl_text = w_attr(override, "ilvl")
                start_override = override.find(f"./{w_tag('startOverride')}")
                if ilvl_text is None or start_override is None:
                    continue
                start_val = w_attr(start_override, "val")
                if start_val is not None:
                    instance.start_overrides[int(ilvl_text)] = int(start_val)

            num_instances[num_id] = instance

        return abstract_defs, num_instances

    def _render_level_text(self, template: str, level_defs: dict[int, LevelDef], counters: dict[int, int]) -> str:
        def replace(match: re.Match[str]) -> str:
            level_number = int(match.group(1)) - 1
            level_def = level_defs.get(level_number)
            current_value = counters.get(level_number)
            if level_def is None or current_value is None:
                return match.group(0)
            return format_number(current_value, level_def.num_fmt)

        return re.sub(r"%(\d+)", replace, template)

    def _get_paragraph_numbering_label(self, paragraph: Paragraph, numbering_context: dict[str, Any]) -> Optional[str]:
        ppr = paragraph._p.find(f"./{w_tag('pPr')}")
        if ppr is None:
            return None

        num_pr = ppr.find(f"./{w_tag('numPr')}")
        if num_pr is None:
            return None

        ilvl_elem = num_pr.find(f"./{w_tag('ilvl')}")
        num_id_elem = num_pr.find(f"./{w_tag('numId')}")
        num_id = w_attr(num_id_elem, "val")
        ilvl_text = w_attr(ilvl_elem, "val")
        if num_id is None or ilvl_text is None:
            return None

        ilvl = int(ilvl_text)
        instance = numbering_context["num_instances"].get(num_id)
        if instance is None:
            return None

        abstract_def = numbering_context["abstract_defs"].get(instance.abstract_num_id)
        if abstract_def is None:
            return None

        level_def = abstract_def.levels.get(ilvl)
        if level_def is None:
            return None

        counters = numbering_context["state"].setdefault(num_id, {})
        for deeper_level in [level for level in counters if level > ilvl]:
            del counters[deeper_level]

        for parent_level in range(ilvl):
            if parent_level not in counters:
                parent_def = abstract_def.levels.get(parent_level)
                if parent_def:
                    counters[parent_level] = instance.start_overrides.get(parent_level, parent_def.start)

        if ilvl not in counters:
            counters[ilvl] = instance.start_overrides.get(ilvl, level_def.start)
        else:
            counters[ilvl] += 1

        label = self._render_level_text(level_def.lvl_text, abstract_def.levels, counters).strip()
        return label or None

    def _get_numbering_context(self, doc: _Docx) -> Optional[dict[str, Any]]:
        try:
            numbering_root = doc.part.numbering_part.element
        except Exception:
            return None

        try:
            abstract_defs, num_instances = self._parse_numbering_element(numbering_root)
        except Exception:
            return None

        return {
            "abstract_defs": abstract_defs,
            "num_instances": num_instances,
            "state": {},
        }
    
    def __call__(self, document, for_llm, chunk_type):
        if chunk_type == "legal":
            return self.extract(document=document, for_llm=for_llm)    
        elif chunk_type == "general":
            return self.extract_general(document=document, doc_id=document.doc_id)

    def iter_block_items(self, parent):
        if isinstance(parent, _Docx):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        elif isinstance(parent, _Row):
            parent_elm = parent._tr
        else:
            raise ValueError("something's not right")
        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)
            elif child.tag == qn('w:sdt'):
                sdtContent = child.find(qn('w:sdtContent'))
                if sdtContent is not None:
                    for sdt_child in sdtContent.iterchildren():
                        if isinstance(sdt_child, CT_P):
                            yield Paragraph(sdt_child, parent)
                        elif isinstance(sdt_child, CT_Tbl):
                            yield Table(sdt_child, parent)
                
    def _print_tc(self, cell: list[Par]) -> str:
        """Print a table cell as a string on one line."""
        ps = ["".join(p.run_strings).replace("\n", " ") for p in cell]
        return "\n\n".join(ps)


    def _join_and_enclose_with_pipes(self, strings: list[str]) -> str:
        """Join strings with pipes and enclose with pipes."""
        return "|" + "|".join(strings) + "|"


    def _print_text(self, tbl: list[list[list[Par]]]) -> str:
        """Text in this list [[[Par]]] is not a table. It's just text."""
        all_cells = iter_at_depth(tbl, 2)
        return "\n\n".join(self._print_tc(tc) for tc in all_cells)


    def _print_tbl(self, tbl: list[list[list[Par]]]) -> str:
        """Text in this list [[[Par]]] is a table."""
        rows_as_string_lists = [[self._print_tc(tc) for tc in tr] for tr in tbl]
        rows_as_string_lists.insert(1, ["---"] * len(rows_as_string_lists[0]))
        rows_as_strings = [
            self._join_and_enclose_with_pipes(row) for row in rows_as_string_lists
        ]
        return "\n".join(rows_as_strings)

    def iter_unique_cells(self, cells):
        prior_cell = None
        for c in cells:
            if c == prior_cell:
                continue
            yield c
            prior_cell = c

    def para2text(self, p):
        elements = p._element.xpath(".//w:t | .//w:br | .//w:p")
        texts = []

        for element in elements:
            if element.tag.endswith("br"):
                texts.append("\n")
            elif element.tag.endswith("p"):
                texts.append(" ")
            else:
                texts.append(element.text)

        text = "".join(texts)
        return "\n".join(" ".join(line.split()) for line in text.splitlines())

    def getText(self, document, for_llm):
        doc = docx.Document(str(document.file_path))
        numbering_context = self._get_numbering_context(doc)

        # Use docx2python to extract tables
        tables = []
        try:
            with docx2python(str(document.file_path), duplicate_merged_cells=False) as extraction:
                tables_pars = extraction.document_pars
            for possible_table in iter_tables(tables_pars):
                if is_tbl(possible_table):
                    table_array = TableArrayTools.docx2python_to_table_array(possible_table)
                    tables.append(table_array)
        except Exception as e:
            print(e)
            tables = []

        text_elements = []
        count_tables = 0
        for block in self.iter_block_items(doc):
            if isinstance(block, Paragraph):
                text = self.para2text(block)
                if numbering_context is not None:
                    label = self._get_paragraph_numbering_label(block, numbering_context)
                    if label and text and not text.startswith(label):
                        text = f"{label} {text}"
                if text:
                    text_elements.append(text)
            elif isinstance(block, Table):
                if tables:
                    # Use the extracted table from docx2python if available
                    text_elements.append(tables[count_tables])
                    count_tables+=1
                else:
                    # Fallback to use the docx library to extract table content
                    try:
                        # I haven't handled merged cells yet, merged cells will be duplicated
                        table_array = TableArrayTools.docx_to_table_array(block)
                        text_elements.append(table_array)
                    except Exception as e:
                        print(e)

        # reduced_text_elements = [text_elements[0]]
        # for i in text_elements[1:]:
        #     if TableArrayTools.is_mergeable(reduced_text_elements[-1], i):
        #         reduced_text_elements[-1] = TableArrayTools.merge_table_arrays(reduced_text_elements[-1], i)
        #     else:
        #         reduced_text_elements.append(i)

        if for_llm:
            for idx in range(1, len(text_elements)):
                if TableArrayTools.is_mergeable(text_elements[idx-1], text_elements[idx]):
                    text_elements[idx] = text_elements[idx-1][:3] + text_elements[idx]
            
        # Convert table array to html
        # table arrays could appear in the middle of text
        for idx in range(len(text_elements)):
            if TableArrayTools.validate_table_array(text_elements[idx]):
                text_elements[idx] = TableArrayTools.make_html_table_str(text_elements[idx])

        # print(*text_elements, sep="\n")
        return text_elements

    def extract_code(self, document, text_elements):
        for i in range(0, min(20, len(text_elements))):
            match = re.search(r"SỐ:\s*([A-ZĐƯƠĂÂÊ0-9/-]+)", text_elements[i].upper())

            if match:
                document.code = match.group(1)
                break
        return document

    def extract_type_and_title(self, document, text_elements):
        for i in range(0, min(20, len(text_elements))):
            # print(f"text_elements: {text_elements[i]}")
            for law_type in LAW_TYPES_LIST:
                if text_elements[i].lower().find(law_type) != -1:
                    part_element = text_elements[i].split("\n")
                    # remove "" in part_element
                    part_element = [x for x in part_element if x != ""]
                    if part_element[0].isupper():
                        if len(part_element) > 1 and part_element[-1] != "":
                                document.law_type = part_element[0].lower()
                                document.title = part_element[1].lower()
                        else:
                            document.law_type = text_elements[i]
                            document.title = text_elements[i + 1]
            if document.law_type != "" and document.title != "":
                break
        if document.title == "":
            for i in range(0, min(20, len(text_elements))):
                match = re.search(r"V\/v\s+(.+?)(?=\s{2,}|\t)", text_elements[i])

                if match:
                    document.law_type = "CÔNG VĂN"
                    document.title = match.group(1)
                    break
        return document
    
    def extract_tree_structure(self, document, for_llm):
        def pre_process_texts(text_elements):
            results = []
            for id, text in enumerate(text_elements):
                text = text.strip()
                if text:
                    results.append({
                        "id": id,
                        "text": text,
                        "edited": False,
                    })
            return results
        # doc
        text_elements = self.getText(document, for_llm=for_llm)
        document_code = document.code
        text_elements = pre_process_texts(text_elements)
        try:
            nodes = self.document_constructor.node_constructor(text_elements, document_code)
            tree_structure = self.document_constructor.construct_tree(nodes)
            document.tree_structure = tree_structure
        except Exception as e:
            print(e)
            document.tree_structure = []
        return document

    def extract_based_documents(self, document: Document, text_elements) -> Document:
        '''
        Hàm trích xuất các văn bản liên quan từ văn bản gốc (căn cứ)
        Inputs:
            document: Document.
        Outputs:
            related_docs: List. Danh sách các law_code liên quan được trích xuất.
        '''
        document.based_documents = []
        set_based_documents = set()
        law_types_pattern = "|".join(LAW_TYPES_LIST)
        pattern = rf'căn\s+cứ\s+({law_types_pattern})[^;]*?số\s+([^\s;,]+)'
        for text_element in text_elements:
            matches = re.findall(pattern, text_element, re.IGNORECASE)
            for match in matches:
                law_type, law_code = match
                set_based_documents.add(law_code.strip())
        document.based_documents = list(set_based_documents)
        return document
    
    def extract_created_date(self, document, text_elements):
        for i in range(0, min(20, len(text_elements))):
            match = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text_elements[i])

            if match:
                day, month, year = match.groups()
                document.created_date = f"{day}/{month}/{year}"
                break
        return document

    def article_chunking(self, document, text):
        articles = []
        article = Article(doc_id=document.doc_id, code=document.code, law_type=document.law_type)
        appendix = None

        index = 0

        for paragraph in text:
            match = re.search(r"^(Điều\s+\d+)\.", paragraph[:12])

            if match:
                if article:
                    article.text = article.text[:-1]
                    articles.append(article)

                number = match.group(1)
                article = Article(
                    doc_id=document.doc_id,
                    title_article=paragraph[len(number) + 2 :],
                    code=document.code,
                    law_type=document.law_type,
                    number_article=number,
                )

            elif article:
                if "Nơi nhận:" in paragraph.strip()[:12]:
                    article.text = article.text[:-1]
                    articles.append(article)
                    article = Article(doc_id=document.doc_id, code=document.code, law_type=document.law_type)

                article.text = article.text + paragraph + " "

            if len(paragraph) <= 20 and paragraph[:7].lower() == "phụ lục":
                if article:
                    article.text = article.text[:-1]
                    articles.append(article)
                    article = None

                if appendix:
                    appendix.text = appendix.text[:-1]
                    articles.append(appendix)

                title = paragraph

                if (index + 1) < len(text):
                    title = text[index + 1]

                appendix = Article(
                    doc_id=document.doc_id,
                    title_article=title,
                    code=document.code,
                    law_type=document.law_type,
                    number_article=paragraph,
                )
            else:
                if appendix:
                    appendix.text = appendix.text + paragraph + " "

            index += 1

        if article:
            article.text = article.text[:-1]
            articles.append(article)

        if appendix:
            appendix.text = appendix.text[:-1]
            articles.append(appendix)

        document.articles = articles
        return document
    
    def extract(self, document, for_llm):
        text_elements = self.getText(document, for_llm=False)

        content = document.content
        for text in text_elements:
            if self.document_constructor.level_detector.check_table(text):
                md_table = self.document_constructor.normalize_table(text)
                content += "\n" + md_table
            else:
                content += "\n" + text

        document.content = content
        document = self.extract_type_and_title(document, text_elements)
        document = self.extract_code(document, text_elements)
        document = self.extract_created_date(document, text_elements)
        document = self.extract_based_documents(document, text_elements)
        
        # Trigger LegalTreeConstruct here
        document = self.extract_tree_structure(document, for_llm)

        if document.law_type and document.law_type.lower() in LAW_TYPES_LIST:
            document = self.article_chunking(document, text_elements)
        return document

    # for general document
    def general_get_flat_tree(self, document) -> list[dict]:
        """
        Extract level info for general text documents.
        Does NOT modify getText().
        
        Returns:
            [
                {
                    "index": int,
                    "text": str,
                    "level": int | None
                }
            ]
        """
        file_path = document.file_path
        doc = docx.Document(str(file_path))
        numbering_context = self._get_numbering_context(doc)
        # Use docx2python to extract tables
        tables = []
        try:
            with docx2python(str(file_path), duplicate_merged_cells=False) as extraction:
                tables_pars = extraction.document_pars
            for possible_table in iter_tables(tables_pars):
                if is_tbl(possible_table):
                    table_array = TableArrayTools.docx2python_to_table_array(possible_table)
                    tables.append(table_array)
        except Exception as e:
            print(e)
            tables = []

        results = []
        paragraph_index = 0
        count_tables = 0

        for block in self.iter_block_items(doc):

            if isinstance(block, Paragraph):
                text = self.para2text(block)
                if not text:
                    continue

                if numbering_context is not None:
                    label = self._get_paragraph_numbering_label(block, numbering_context)
                    if label and not text.startswith(label):
                        text = f"{label} {text}"

                level = -1

                p = block._p
                if p.pPr is not None and p.pPr.numPr is not None:
                    try:
                        ilvl = p.pPr.numPr.ilvl
                        if ilvl is not None:
                            level = int(ilvl.val) + 1  # Word level starts at 0
                    except Exception:
                        level = -1

                results.append({
                    "index": paragraph_index,
                    "text": text,
                    "level": level,
                    "type": "nội dung"
                })

                paragraph_index += 1

            elif isinstance(block, Table):
                if tables:
                    # Use the extracted table from docx2python if available
                    table_array = tables[count_tables]
        

                    results.append({
                        "index": paragraph_index,
                        "text": table_array,
                        "level": -1,
                        "type": "table"
                    })
                    count_tables+=1
                else:
                    # Fallback to use the docx library to extract table content
                    try:
                        # I haven't handled merged cells yet, merged cells will be duplicated
                        table_array = TableArrayTools.docx_to_table_array(block)
                        results.append(table_array)
                    except Exception as e:
                        print(e)

            for idx in range(0, len(results)-1):
                if TableArrayTools.is_mergeable(results[idx]["text"], results[idx+1]["text"]):
                    results[idx]["text"] = results[idx]["text"][:3] + results[idx+1]["text"]
            
            for idx in range(len(results)):
                if TableArrayTools.validate_table_array(results[idx]["text"]):
                    results[idx]["text"] = convert_to_markdown(TableArrayTools.make_html_table_str(
                        results[idx]["text"]),
                        escape_misc=False, 
                        ).strip()

        return results

    # build nested tree with children
    def extract_general(self, document: Document, doc_id: str):
        flat_tree = self.general_get_flat_tree(document)

        tree_builder = StructuredDocumentTreeBuilder()
        
        for item in flat_tree:
            document.content += "\n" + item["text"]

        document.tree_structure = tree_builder.build(flat_tree)
        
        return document
        
