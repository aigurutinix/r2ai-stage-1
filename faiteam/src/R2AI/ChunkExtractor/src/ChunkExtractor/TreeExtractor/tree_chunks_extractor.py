import json
from typing import List
from semantic_text_splitter import TextSplitter
import tiktoken

class TreeChunkExtractor:
    def __init__(self,):
        self.adjacency_list = {}
        self.node_list = {}
        self.article = ""
        self.text_splitter = TextSplitter.from_tiktoken_model(
            model="gpt-4", capacity=2048, overlap=512
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def merge_chunks(self, data_tree: List = [], list_article: List = [], token_limit: int = 1024) -> List[dict]:
        chunks = []
        content_path = []
        if data_tree == [] or data_tree is None:
            return chunks
        for node in data_tree:
            node_id = node.get("id")
            num_tokens = node.get("num_tokens")
            content = node.get("content")
            level = node.get("level")
            childs = node.get("children")
            is_edit = node.get("is_edit")
            type = node.get("type")
            article_belong = node.get("number")

            if type in ["điều"]:
                list_article.append(node.get("number"))
                # print(list_article)
            chunks.append(content)
            if childs != [] and childs is not None:
                content_child = self.merge_chunks(childs, list_article)
                if content_child is not None:
                    chunks.extend(content_child)
        return chunks
    
    def chunk_table(self, content: str, header_limit: int = 4, max_length_chunk: int = 7000, separator: str = "\n"):
        
        header_rows = content.split(separator)[:header_limit]
        header = separator.join(header_rows)

        if len(self.encoding.encode(header)) > max_length_chunk/2:
            header_limit = 1
            header = separator.join(content.split(separator)[:header_limit])

        content = separator.join(content.split(separator)[header_limit:])

        new_texts = []
        current_text = ""
        for line in content.split(separator):
            if len(self.encoding.encode(header + separator + current_text + separator + line)) + 1 <= max_length_chunk:
                current_text += separator + line
            else:
                new_texts.append(header + current_text)
                current_text = line

        if current_text != "":
            new_texts.append(header + separator + current_text)

        # for text in new_texts:
        #     print(text)
        #     print("---------------------")
        return new_texts
        
    def extract_chunks(self, data: List, content_path: List, chunks: List, article:str = "", recent_title="")-> List:
        if data == [] or data is None:
            return
        
        for i, node in enumerate(data):
            node_id = node.get("id")
            num_tokens = node.get("num_tokens") or 0
            content = node.get("content") or ""
            level = node.get("level")
            childs = node.get("children")
            is_edit = node.get("is_edit")
            # metadata = node.get("metadata")
            type = node.get("type")
            if type == "title":
                recent_title = content
            content_path.append(content)
            document_code = node.get("document_code")
            if type in ["điều","điều_ver2"]:
                article = node.get("number")
            if num_tokens <= 1024 or type in ["khoản"] or type in ["table"]:
                list_article = []
                if type in ["table"]:
                    chunks.append({
                        "id": node_id,
                        "content": content,
                        "level": level,
                        "num_tokens": num_tokens,
                        "type": type,
                        "article": article,
                        "list_article": list_article,
                        "document_code": document_code,
                        "content_path": content_path.copy(),
                    })
                else:
                    if type in ["khoản"]:
                        node["article"] = article
                    list_article = []
                    content_child = self.merge_chunks(childs, list_article)
                    if content_child is not None:
                        content += "\n" + "\n".join(content_child)

                    chunks.append({
                        "id": node_id,
                        "content": content,
                        "level": level,
                        "num_tokens": num_tokens,
                        "type": type,
                        "article": article,
                        "list_article": list_article,
                        "document_code": document_code,
                        "content_path": content_path.copy()
                    })
            else:
                level = node.get("level")
                if level == -1:
                    chunks.append({
                        "id": node_id,
                        "content": content,
                        "level": level,
                        "num_tokens": num_tokens,
                        "type": "title",
                        "article": "",
                        "list_article": "",
                        "document_code": document_code,
                        "content_path": []
                    })

                elif type in ["điều", "điều_ver2"]:
                    chunks.append({
                        "id": node_id,
                        "content": content,
                        "level": level,
                        "num_tokens": len(self.encoding.encode(content)),
                        "type": type,
                        "article": article,
                        "list_article": [],
                        "document_code": document_code,
                        "content_path": content_path.copy(),
                    })
                    
                self.extract_chunks(childs, content_path, chunks, article=article, recent_title=recent_title)
            if content_path != []:
                content_path.pop()
        
    def _pack_small_chunks(self, chunks: List[dict], min_tokens: int = 7000) -> List[dict]:
        if not chunks:
            return chunks

        DIEU_TYPES = {"điều", "điều_ver2", "table"}
        result = []
        current = chunks[0].copy()
        current_tokens = len(self.encoding.encode(current.get("content") or ""))

        for next_chunk in chunks[1:]:
            current_type = current.get("type")
            next_type = next_chunk.get("type")
            if current_type in DIEU_TYPES:
                print(f"Current chunk ID {current.get('id')} type {current_type} tokens {current_tokens}")
            next_tokens = len(self.encoding.encode(next_chunk.get("content") or ""))

            if (current_type not in DIEU_TYPES
                    and next_type not in DIEU_TYPES
                    and current_tokens + next_tokens < min_tokens):
                current["content"] = (current.get("content") or "").rstrip() + "\n" + (next_chunk.get("content") or "").lstrip()
                current_tokens += next_tokens
                current["num_tokens"] = current_tokens
            else:
                print(f"Điều content: {current.get('content')}")
                current["id"] = len(result)
                result.append(current)
                current = next_chunk.copy()
                current_tokens = next_tokens

        current["id"] = len(result)
        result.append(current)
        return result

    def process(self, data_tree: List, max_length=7000):
        chunks = []
        self.extract_chunks(data_tree, [], chunks)
        new_chunks = []

        for chunk in chunks:
            text = chunk.get("content")
            if text in ["", None, " "]:
                continue
            length = len(self.encoding.encode(text))
            chunk_type = chunk.get("type")

            new_texts = []
            if chunk_type == "table":
                if length > max_length:
                    new_texts = self.chunk_table(content=text, max_length_chunk=2048)

            elif chunk_type != "table" and length > max_length:
                new_texts = self.text_splitter.chunks(text)

            if new_texts:
                for i, text in enumerate(new_texts):
                    new_chunks.append({
                        "id": len(new_chunks),
                        "content": text,
                        "level": chunk.get("level"),
                        "num_tokens": len(self.encoding.encode(text)),
                        "type": chunk.get("type"),
                        "article": chunk.get("article"),
                        "list_article": chunk.get("list_article"),
                        "document_code": chunk.get("document_code"),
                        "content_path": chunk.get("content_path"),
                    })
            else:
                new_chunks.append(chunk)

        for chunk in new_chunks:
            if chunk.get("type") == "table":
                print(f"Table chunk ID {chunk.get('id')} tokens {chunk.get('num_tokens')}")
                print(chunk.get("content")[:100])
            
        return self._pack_small_chunks(new_chunks, min_tokens=max_length)

    def process_fill_max_length_by_level(self, data_tree: List, max_length=7000) -> List[dict]:
        """
        Build chunks in top-down document order and greedily fill each chunk
        up to ``max_length``.

        Behaviour:
        - Oversized nodes are split first so every unit is <= ``max_length``.
        - Chunks are then merged strictly in extraction order (preorder),
          regardless of whether the next node is an ``điều`` or ``khoản``.
        - Breadcrumb context for appended nodes is injected when the path
          changes so downstream consumers do not lose hierarchy information.
        """
        raw_chunks = []
        self.extract_chunks(data_tree, [], raw_chunks)

        split_chunks = []
        splitter = TextSplitter.from_tiktoken_model(
            model="gpt-4",
            capacity=max_length,
            overlap=0,
        )

        for chunk in raw_chunks:
            text = (chunk.get("content") or "").strip()
            if not text:
                continue

            chunk_type = chunk.get("type")
            if chunk_type == "table":
                if len(self.encoding.encode(text)) > min(max_length, 2048):
                    parts = self.chunk_table(
                        content=text,
                        max_length_chunk=min(max_length, 2048),
                    )
                else:
                    parts = [text]
            elif len(self.encoding.encode(text)) > max_length:
                parts = splitter.chunks(text)
            else:
                parts = [text]

            for part in parts:
                part = (part or "").strip()
                if not part:
                    continue

                split_chunks.append(
                    {
                        "id": len(split_chunks),
                        "content": part,
                        "level": chunk.get("level"),
                        "num_tokens": len(self.encoding.encode(part)),
                        "type": chunk.get("type"),
                        "article": chunk.get("article"),
                        "list_article": list(chunk.get("list_article") or []),
                        "document_code": chunk.get("document_code"),
                        "content_path": list(chunk.get("content_path") or []),
                        "_last_content_path": list(chunk.get("content_path") or []),
                    }
                )

        packed_chunks = []
        current_chunk = None

        def collect_articles(chunk_data: dict) -> List[str]:
            articles = []
            for article in [chunk_data.get("article"), *(chunk_data.get("list_article") or [])]:
                if article and article not in articles:
                    articles.append(article)
            return articles

        def append_with_context(base_chunk: dict, next_chunk: dict) -> str:
            next_text = (next_chunk.get("content") or "").strip()
            if not next_text:
                return (base_chunk.get("content") or "").strip()

            last_path = base_chunk.get("_last_content_path") or []
            next_path = next_chunk.get("content_path") or []

            last_breadcrumb = "\n".join(last_path[:-1]).strip()
            next_breadcrumb = "\n".join(next_path[:-1]).strip()

            if next_breadcrumb and next_breadcrumb != last_breadcrumb:
                next_text = f"{next_breadcrumb}\n{next_text}"

            base_text = (base_chunk.get("content") or "").rstrip()
            return f"{base_text}\n{next_text}".strip()

        for chunk in split_chunks:
            if current_chunk is None:
                current_chunk = chunk.copy()
                continue

            merged_text = append_with_context(current_chunk, chunk)
            merged_tokens = len(self.encoding.encode(merged_text))

            if merged_tokens > max_length:
                current_chunk["id"] = len(packed_chunks)
                current_chunk.pop("_last_content_path", None)
                packed_chunks.append(current_chunk)
                current_chunk = chunk.copy()
                continue

            merged_articles = collect_articles(current_chunk)
            for article in collect_articles(chunk):
                if article not in merged_articles:
                    merged_articles.append(article)

            current_chunk["content"] = merged_text
            current_chunk["num_tokens"] = merged_tokens
            current_chunk["list_article"] = merged_articles
            current_chunk["article"] = merged_articles[0] if merged_articles else current_chunk.get("article", "")
            current_chunk["_last_content_path"] = list(chunk.get("content_path") or [])

        if current_chunk is not None:
            current_chunk["id"] = len(packed_chunks)
            current_chunk.pop("_last_content_path", None)
            packed_chunks.append(current_chunk)

        return packed_chunks
