"""Build FisReader Document + legal tree from a parquet row (plain text)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from FisReader.document import Document
from LegalTreeConstruct.document_constructor import DocumentConstructor


class TextDocumentBuilder:
    """Convert a parquet row into a Document with LegalTreeConstruct tree."""

    def __init__(self, document_constructor: DocumentConstructor | None = None) -> None:
        self.document_constructor = document_constructor or DocumentConstructor()

    @staticmethod
    def _paragraphs_from_content(content: str) -> list[dict]:
        return [
            {"id": idx, "text": text, "edited": False}
            for idx, text in enumerate(content.split("\n"))
            if text.strip()
        ]

    def build(self, row: dict[str, Any] | Any) -> Document:
        """Build Document from a pandas Series or dict with parquet columns."""
        if hasattr(row, "to_dict"):
            row = row.to_dict()

        doc_id = int(row["id"])
        content = str(row.get("content") or "").strip()
        document_number = str(row.get("document_number") or "").strip()
        title = str(row.get("title") or "").strip()
        legal_type = str(row.get("legal_type") or "văn bản pháp luật").strip()

        document = Document(
            file_path=Path(f"parquet/{doc_id}.txt"),
            doc_id=doc_id,
        )
        document.code = document_number
        document.title = title
        document.law_type = legal_type
        document.content = content

        if not content:
            document.tree_structure = []
            return document

        paragraphs = self._paragraphs_from_content(content)
        try:
            nodes = self.document_constructor.node_constructor(
                paragraphs,
                document_number,
            )
            document.tree_structure = self.document_constructor.construct_tree(nodes)
        except Exception:
            document.tree_structure = []

        return document
