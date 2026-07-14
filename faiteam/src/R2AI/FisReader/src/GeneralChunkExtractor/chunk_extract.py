from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from FisReader.document import Document
from FisReader.reader.docx_reader import WordReader


class GeneralChunkExtractor:
    """
    Compatibility wrapper for "GeneralChunkExtractor".

    The upstream codebase in this workspace doesn't contain the original
    GeneralChunkExtractor package, but FisReader already has general tree
    construction via GeneralTreeConstruct inside WordReader.extract_general().
    """

    def __init__(self, reader: Optional[WordReader] = None) -> None:
        self._reader = reader or WordReader()

    def extract_tree(
        self,
        file_path: Path,
        doc_id: int,
        *,
        for_llm: bool = False,  # reserved for future parity
        extra: Optional[dict[str, Any]] = None,  # reserved
    ):
        _ = (for_llm, extra)
        document = Document(file_path=file_path, doc_id=doc_id)
        self._reader.extract_general(document=document, doc_id=str(doc_id))
        return document.tree_structure

