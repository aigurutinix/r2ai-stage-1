import re
from typing import List
from .tree_builder import TreeBuilder
from .level_detector import LevelDetector
from .token_counter import TokenCounter
from .tree_exporter import TreeExporter
import unicodedata

from html_to_markdown import convert_to_markdown

class DocumentConstructor:
    def __init__(self, output_dir="./"):
        self.output_dir = output_dir
        self.token_counter = TokenCounter()
        self.level_detector = LevelDetector()
        self.tree_builder = TreeBuilder()
        self.is_edit_article = False

    def normalize_vietnamese(self,text):
        return unicodedata.normalize('NFC', text)

    def check_edit_article(self, text: str) -> bool:
        # using regex to check if the text is an article with start of article include "sửa đổi", "bổ sung", "thay đổi", "thay thế", "bãi bỏ", "đình chỉ", "hủy bỏ" like điều 1. Sửa đổi ...
        edit_article_patterns = [
           r'(?i)^(Điều\s+\d+\.\s*)(sửa đổi|bổ sung|thay đổi|thay thế|bãi bỏ|hủy bỏ|chỉnh sửa|đình chỉ)(.*?)(điều|nghị định|luật|thông tư|khoản|mục|chương)\b',
           r'(?i)^((\d+|\w+)(\.|\))\s*)(sửa đổi|bổ sung|thay đổi|thay thế|bãi bỏ|hủy bỏ|chỉnh sửa|đình chỉ)(.*?)(điều|nghị định|luật|thông tư|khoản|mục|chương)\b'

        ]
        for pattern in edit_article_patterns:
            if re.search(pattern, text.strip(), re.IGNORECASE):
                return True
        return False
    
    def check_is_edit_content(self, text: str) -> bool:
        if text[0] in ["“"]:
            return True
        return False
    
    def check_is_edit_content_end(self, text: str) -> bool:
        if (len(text) > 2 and text[-2] in ["”"]) or text.find("”") != -1:
            return True
        return False

    def pre_process(self, paragraphs: List) -> List:
        '''
        Pre-process the paragraphs before build tree. Recognize where paragraph is edit article.
        Inputs:
            paragraphs: List. List of paragraphs.
        Outputs:
            List. List paragraphs where edit article is recognized. If all paragraphs are not edit article, return [].
            
        '''
        list_edit_paragraph = []
        pre_status = False
        for p in paragraphs:
            if self.check_edit_article(p.get("text")):
                if not p.get("edited"): 
                    list_edit_paragraph.append(p)
                    pre_status = True                    
            elif pre_status:
                p["edited"] = False
                list_edit_paragraph.append(p)
        return list_edit_paragraph
    def normalize_table(self, text: str) -> str:
        """ Expect text is a table in html format, convert it to markdown format
            HTML validator and table validator are not implemented, 
            so we assume text is a valid table in html format
        """
        return convert_to_markdown(text, escape_misc=False).strip()

    def node_constructor(self, paragraphs: List, document_code: str) -> List:
        '''
        Construct node before build tree.
        Inputs:
            paragraphs: List. List of paragraphs.
            document_code: str. The code of the document.
        Outputs:
            List. A dictionary containing the tree and mapping id-content.
        '''
        nodes = []
        mapping_id_content = {}
        pre_level = None
        is_edit_article = False
        id_p = 0        
        content_table = ""
        for  paragraph in paragraphs:
            is_edit = False
            id = paragraph.get('id')
            content = paragraph.get('text')
            content_table = ""
            if content:
                content = self.normalize_vietnamese(content)
                is_edit_content = self.check_is_edit_content(content)
                is_edit_content_end = self.check_is_edit_content_end(content)
                level = self.level_detector.get_level(text=content)   
                if level.get("type") == "table":
                    content_table = self.normalize_table(content)
                if is_edit_content or is_edit :
                    is_edit = True
                    level.update({"type": "nội dung", "level_id": 7})
                if is_edit_content_end:
                    is_edit = False
                if id_p == 1:
                    level['level_id'] = -1
                else:         
                    if level.get("type") in ["title"]:
                        if pre_level is not None and pre_level.get("type") in ["mục", "chương", "phần", "điều"]:
                            level['level_id'] = pre_level['level_id'] + 0.5
                    # if is_edit_article == False :
                    if level.get("type") in ["điều", "khoản"] :
                        is_edit_article = self.check_edit_article(content)
                        if is_edit_article:
                            is_edit = True
                            mapping_id_content[str(id)] = content
                    pre_level = level
                    # else:
                    #     is_edit = True
                    #     level = pre_level
                    #     level['level_id'] = -1
                
                nodes.append({
                    "id": str(id),
                    "level": level,
                    "content": content,
                    "children": [],
                    "is_edit": is_edit,
                    "document_code": document_code
                }) 
                if content_table:
                    nodes[-1]["content"] = content_table
                    nodes[-1]["content_raw"] = content_table

                else:
                    nodes[-1]["content_raw"] = content
                id_p += 1
            else:
                continue
        return nodes

    def construct_tree(self, nodes: List):
        '''
        Construct tree from nodes.
        Inputs:
            nodes: List. List of nodes.
        Outputs:
            List. List of tree.
        '''
        
        tree = self.tree_builder.build_tree(nodes)
        _ = [self.token_counter.calculate_tokens_recursive(node) for node in tree]
        return tree
    def construct_closure_table(self, nodes: List):
        '''
        Construct closure table from tree.
        Inputs:
            tree: List. List of tree.
        Outputs:
            List. List of closure table.
        '''
        tree = self.construct_tree(nodes)
        
        return self.tree_builder.build_closure_table(tree)
    
    def construct_adjacency_list(self, nodes: List):
        '''
        Construct adjacency list from tree.
        Inputs:
            tree: List. List of tree.
        Outputs:
            List. List of adjacency list.
        '''
        tree = self.construct_tree(nodes)
        return self.tree_builder.build_adjacency_list(tree)
        
    def export_tree_nodes(self, data: dict, path_save: str):
        TreeExporter.save_nodes(data, path_save)
        
    def export_tree_edges(self, data: dict, path_save: str):
        TreeExporter.save_edges(data, path_save)
        