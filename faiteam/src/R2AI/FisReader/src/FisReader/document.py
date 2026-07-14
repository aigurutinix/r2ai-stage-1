from pathlib import Path
from dataclasses import dataclass, field
from typing import TypedDict

LAW_TYPES_LIST = [
    "hiến pháp",
    "luật",
    "pháp lệnh",
    "nghị định",
    "thông tư",
    "nghị quyết",
    "thông tư liên tịch",
    "quyết định",
    "bộ luật",
    "chỉ thị",
    "quy định"
]


@dataclass
class Article:
    doc_id: int = 0
    code: str = ""
    law_type: str = ""
    number_article: str = ""
    title_article: str = ""
    text: str = ""

    class TextOcrType(TypedDict):
        text: str
        coords: list[int]
        page: int
    text_ocr_list: list[TextOcrType] = field(default_factory=list)


class Document:
    is_ocr: int = 0

    law_type: str = "văn bản pháp luật"
    title: str = ""
    code: str = ""
    created_date: str = ""

    content: str = ""
    content_ocr: list[dict] = []
    articles: list[Article] = []
    tree_structure: list[dict] = []
    based_documents: list[str] = []

    def __init__(self, file_path: Path, doc_id: int) -> None:
        self.file_path = file_path
        self.doc_id = doc_id