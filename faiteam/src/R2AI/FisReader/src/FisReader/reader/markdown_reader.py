import re
import subprocess
from pathlib import Path
from typing import Iterator

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def detect_articles(text: str) -> list[str]:
    article_pattern = r"(\*\*Điều\s+\d+\.\s+.+|#+\s+Điều\s+\d+\.\s+.+)$"
    lines = text.splitlines(keepends=True)

    positive_lines = []

    for i in range(len(lines) - 1):
        if re.match(article_pattern, lines[i]):
            positive_lines.append(lines[i])
            positive_lines.append(lines[i + 1])

    if re.match(article_pattern, lines[-1]):
        positive_lines.append(lines[-1])

    return positive_lines


class MarkdownProcessor:
    @staticmethod
    def _assert_chunk_size_and_overlap(chunk_size: int, chunk_overlap: int) -> None:
        assert chunk_size > chunk_overlap, "chunk_size must be greater than chunk_overlap"
        assert chunk_size > 0, "chunk_size must be greater than 0"
        assert chunk_overlap > 0, "chunk_overlap must be greater than 0"

    @staticmethod
    def by_langchain(markdown_file: Path, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
        """https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/"""
        __class__._assert_chunk_size_and_overlap(chunk_size, chunk_overlap)

        content = markdown_file.read_text(encoding="utf-8")

        headers_to_split_on = [("#" * i, "Header {i}") for i in range(1, 7)]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_splits = markdown_splitter.split_text(content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i in text_splitter.split_documents(md_splits):
            yield i.page_content

    @staticmethod
    def by_chunk_of_lines(markdown_file: Path, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
        __class__._assert_chunk_size_and_overlap(chunk_size, chunk_overlap)

        buffer = []
        with markdown_file.open("r", encoding="utf-8") as fp:
            for line in fp:
                buffer.append(line)
                if len(buffer) == chunk_size:
                    yield "".join(buffer)
                    buffer = buffer[-chunk_overlap:]
        if buffer:
            yield "".join(buffer)


class FileConverter:
    @staticmethod
    def use_pandoc(almost_any_doc_file: Path, target_folder: Path):
        md_file = target_folder / almost_any_doc_file.with_suffix(".md").name
        convert_cmd = f'pandoc -s "{almost_any_doc_file}" -t markdown_mmd -o "{md_file}"'
        try:
            subprocess.check_output(convert_cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print(e)
        return md_file

    @staticmethod
    def use_pymupdf4llm(almost_any_pdf_file: Path, target_folder: Path):
        md_file = target_folder / almost_any_pdf_file.with_suffix(".md").name
        try:
            doc = pymupdf4llm.to_markdown(almost_any_pdf_file)
            md_file.write_text(doc, encoding="utf-8")
        except Exception as e:
            print(e)
        return md_file

    @staticmethod
    def use_any(almost_any_file: Path, target_folder: Path):
        if almost_any_file.suffix == ".pdf":
            return __class__.use_pymupdf4llm(almost_any_file, target_folder)
        else:
            return __class__.use_pandoc(almost_any_file, target_folder)
