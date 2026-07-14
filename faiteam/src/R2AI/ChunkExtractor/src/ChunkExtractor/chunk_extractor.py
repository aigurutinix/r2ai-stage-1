from FisReader.document import Document
from .TreeExtractor.tree_chunks_extractor import TreeChunkExtractor
import tiktoken
from collections import deque
import json
from copy import deepcopy
from semantic_text_splitter import TextSplitter
import warnings

class ChunkExtractor:
    def __init__(
        self, 
        tiktoken_model_name: str = "cl100k_base", 
        chunk_size: int = 2048, 
        chunk_overlap: int = 512,
        tree_chunk_extractor: TreeChunkExtractor = None
    ):
        self.tiktoken_model_name = tiktoken_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = TextSplitter.from_tiktoken_model(
            model="gpt-4", capacity=2048, overlap=512
        )
        if tree_chunk_extractor is None:
            warnings.warn("TreeChunkExtractor is not provided, using default TreeChunkExtractor")
            self.tree_chunk_extractor = TreeChunkExtractor()
        else:
            self.tree_chunk_extractor = tree_chunk_extractor

    def get_chunks_in_tree(
        self, 
        document: Document,
        doc_id: str,
        original_file_path: str, 
        file_name: str
    ):
        tree_structure = document.tree_structure
        chunks = self.tree_chunk_extractor.process(
            data_tree = tree_structure,
        )
        post_process_chunks = []
        document_code = document.code
        document_law_type = document.law_type
        document_law_title = document.title
        # print(f"law_type: {document_law_type}")
        if document_law_title:
            name_doc = document_law_type+" " + document_code + " " + document_law_title
        else:
            name_doc = document_law_type+" " + document_code

        chunk_id = 0
        for chunk in chunks:

            # Remove first table
            if "cộng hòa xã hội chủ nghĩa việt nam" in chunk.get("content").lower() and chunk.get("type") == "table":
                continue
            
            content_path = chunk.get("content_path", [])
            type_chunk = chunk.get("type")
            num_level = chunk.get("num_level")
            if len(content_path) > 1:
                content_path_str = "\n".join(content_path[:-1])
            else:
                content_path_str = ""
            content_path_str = name_doc + " " + content_path_str
            chunk_content = chunk.get("content") if content_path_str == "" else content_path_str + "\n" + chunk.get("content")
            article_id = chunk.get("article")
            list_article = chunk.get("list_article")
            chunk_object = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "original_file_path": original_file_path,
                "file_name": file_name,
                "law_title": document.title,
                "law_type": document.law_type,
                "law_code": document.code,
                "article_number": article_id,
                "article_title": "",
                "text": chunk_content,
                "ocr_meta": "",
                "part": "", # Phần
                "chapter": "", # Chương
                "section": "", # Mục
                "subsection": "", # Tiểu mục
                "clause": "", # Khoản
                "item": "", # Điểm
                "sub_item": "", # Tiết
            }
            # Update article number only — chunk_id is always incremented once per chunk
            if list_article:
                for _id in list_article:
                    article_id += "_" + _id
                chunk_object["article_number"] = article_id
            elif article_id:
                chunk_object["article_number"] = article_id
            elif type_chunk in ["điều", "điều_ver2"]:
                chunk_object["article_number"] = num_level

            post_process_chunks.append(chunk_object)
            chunk_id += 1

        print(f"Number of chunks: {len(post_process_chunks)}")

        return post_process_chunks
    
    def get_chunks_by_article(
        self, 
        document: Document,
        doc_id: str,
        original_file_path: str, 
        file_name: str
    ):
        list_chunks = []
        chunk_id = 0
        for article in document.articles:
            for text in self.text_splitter.chunks(article.text):
                chunk_object = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "original_file_path": original_file_path,
                    "file_name": file_name,
                    "law_title": document.title,
                    "law_type": document.law_type,
                    "law_code": document.code,
                    "article_number": article.number_article,
                    "article_title": article.title_article,
                    "text": article.title_article + text,
                    "ocr_meta": "",
                    "part": "", # Phần
                    "chapter": "", # Chương
                    "section": "", # Mục
                    "subsection": "", # Tiểu mục
                    "clause": "", # Khoản
                    "item": "", # Điểm
                    "sub_item": "", # Tiết
                }
                # print(f'check: {chunk_object.get("text")}')
                list_chunks.append(chunk_object)
                chunk_id += 1
        return list_chunks
    
    
    def get_chunks_no_article(
        self, 
        document: Document,
        doc_id: str,
        original_file_path: str, 
        file_name: str
    ):
        list_chunks = []
        for chunk_id, text in enumerate(self.text_splitter.chunks(document.content)):
            chunk_object = {
                "doc_id": doc_id,
                
                "chunk_id": chunk_id,
                "original_file_path": original_file_path,
                "file_name": file_name,
                "law_title": document.title,
                "law_type": document.law_type,
                "law_code": document.code,
                "article_number": "",
                "article_title": "",
                "text": text,
                "ocr_meta": "",
                "part": "", # Phần
                "chapter": "", # Chương
                "section": "", # Mục
                "subsection": "", # Tiểu mục
                "clause": "", # Khoản
                "item": "", # Điểm
                "sub_item": "", # Tiết
            }
            list_chunks.append(chunk_object)
        return list_chunks

    def get_ocr_result(
        self, 
        document: Document,
    ):
        openai_tokenizer = tiktoken.get_encoding(self.tiktoken_model_name)
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        text_ocr_segment_list = []
        for article_index, article in enumerate(document.articles):
            text_ocr_segment = deque()

            for text_ocr in article.text_ocr_list:
                if len(openai_tokenizer.encode(" ".join([i["text"] for i in text_ocr_segment]))) >= chunk_size:
                    text_ocr_segment_list.append((article_index, deepcopy(text_ocr_segment)))
                    while len(openai_tokenizer.encode(" ".join([i["text"] for i in text_ocr_segment]))) > chunk_overlap:
                        text_ocr_segment.popleft()

                text_ocr_segment.append(text_ocr)

            if len(text_ocr_segment) > 0:
                text_ocr_segment_list.append((article_index, deepcopy(text_ocr_segment)))
        return text_ocr_segment_list
    
    def get_chunks_ocr_by_article(
        self, 
        document: Document,
        doc_id: str,
        original_file_path: str, 
        file_name: str
    ):
        list_chunks = []
        text_ocr_segment_list = self.get_ocr_result(document)

        current_article_index = -1
        chunk_id = 0
        for article_index, text_ocr_segment in text_ocr_segment_list:
            if current_article_index != article_index:
                current_article_index = article_index

            article = document.articles[current_article_index]
            chunk_object = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "original_file_path": original_file_path,
                "file_name": file_name,
                "law_title": document.title,
                "law_type": document.law_type,
                "law_code": document.code,
                "article_number": article.number_article,
                "article_title": article.title_article,
                "text": document.law_type + " " + document.code + "\n" + document.title + " " + article.number_article + " " + article.title_article + " " + " ".join([i["text"] for i in text_ocr_segment]),
                "ocr_meta": json.dumps([{"page_number": i["page"], "coords": i["coords"], "page_size": i["page_size"]} for i in text_ocr_segment]),
            }

            list_chunks.append(chunk_object)
            chunk_id += 1

        return list_chunks

    def get_chunks_ocr_no_article(
        self, 
        document: Document,
        doc_id: str,
        original_file_path: str, 
        file_name: str, 
    ):
        openai_tokenizer = tiktoken.get_encoding(self.tiktoken_model_name)
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap

        text_ocr_segment_list = []
        text_ocr_segment = deque()
        for text_ocr in document.content_ocr:
            if len(openai_tokenizer.encode(" ".join([i["text"] for i in text_ocr_segment]))) >= chunk_size:
                text_ocr_segment_list.append(deepcopy(text_ocr_segment))
                while len(openai_tokenizer.encode(" ".join([i["text"] for i in text_ocr_segment]))) > chunk_overlap:
                    text_ocr_segment.popleft()

            text_ocr_segment.append(text_ocr)

        if len(text_ocr_segment) > 0:
            text_ocr_segment_list.append(deepcopy(text_ocr_segment))
        list_chunks = []
        for chunk_id, text_ocr_segment in enumerate(text_ocr_segment_list):
            chunk_object = {
                "doc_id": doc_id, 
                "chunk_id": chunk_id,
                "original_file_path": original_file_path,
                "file_name": file_name,
                "law_title": document.title,
                "law_type": document.law_type,
                "law_code": document.code,
                "article_number": "",
                "article_title": "",
                "text": " ".join([i["text"] for i in text_ocr_segment]),
                "ocr_meta": json.dumps([{"page": i["page"], "coords": i["coords"]} for i in text_ocr_segment]),
            }
            list_chunks.append(chunk_object)
        return list_chunks
