from pathlib import Path

from .document import Document
from .reader.docx_reader import WordReader
from .reader.doc2docx import send_file_to_doc2docx, send_file_to_docx2pdf

import httpx
import asyncio

class DocumentFactory:
    def __init__(self, *args, **kwargs):
        if kwargs.get("ocr_endpoint") is not None:
            self.ocr_endpoint = kwargs["ocr_endpoint"]
        else:
            self.ocr_endpoint = None

        if kwargs.get("docx_converter_endpoint") is not None:
            self.docx_converter_endpoint = kwargs["docx_converter_endpoint"]
        else:
            self.docx_converter_endpoint = None

        self.word_reader = WordReader()
    
    async def _convert_pdf_to_docx(self, pdf_path: Path) -> Path:
        """
        Converts a PDF file to a DOCX file using an external API.

        Args:
            pdf_path: The path to the input PDF file.

        Returns:
            The path to the converted DOCX file.
        """
        if not self.ocr_endpoint:
            raise ValueError("OCR endpoint not configured for PDF to DOCX conversion.")

        # Construct the URL for the conversion API
        convert_url = f"{self.ocr_endpoint}/api/convert"

        try:
            async with httpx.AsyncClient() as client:
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_path.name, f, "application/pdf")}
                    response = await client.post(convert_url, files=files, timeout=300)
                    response.raise_for_status()  # Raise an exception for bad status codes

            # Save the converted DOCX file
            docx_path = pdf_path.with_suffix(".docx")
            with open(docx_path, "wb") as f:
                f.write(response.content)
            return docx_path
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to convert PDF to DOCX: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during PDF to DOCX conversion: {e}")
        
    def _convert_docx_to_html(self, docx_path: Path) -> Path:
        """
        Converts a DOCX file to an HTML file using an external API.

        Args:
            docx_path: The path to the input DOCX file.

        Returns:
            The path to the converted HTML file.
        """
        if not self.docx_converter_endpoint:
            raise ValueError("DOCX converter endpoint not configured for DOCX to HTML conversion.")

        # Construct the URL for the conversion API
        convert_url = f"{self.docx_converter_endpoint}/convert/docx-to-html"
        try:
            with open(docx_path, "rb") as f:
                files = {"file": (docx_path.name, f, "application/octet-stream")}
                response = httpx.post(convert_url, files=files, timeout=300)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Save the converted HTML file
            html_path = docx_path.with_suffix(".html")
            with open(html_path, "wb") as f:
                f.write(response.content)
            return html_path
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to convert DOCX to HTML: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during DOCX to HTML conversion: {e}")
    
    def doc2dox(self, file_path: Path) -> Path:
        return send_file_to_doc2docx(file_path, file_path.with_suffix(".docx"))
    
    def read(self, file_path: Path, doc_id: int, for_llm=False, return_docx_path=False, return_pdf_path=False, chunk_type = "legal"):
        if file_path.suffix.lower() == ".docx":
            doc = Document(file_path, doc_id)
            self.word_reader(doc, for_llm=for_llm, chunk_type=chunk_type)
            docx_path = file_path
        if file_path.suffix.lower() == ".pdf":
            docx_path = asyncio.run(self._convert_pdf_to_docx(file_path))
            doc = Document(docx_path, doc_id)
            self.word_reader(doc, for_llm=for_llm, chunk_type=chunk_type)
        if file_path.suffix.lower() == ".doc":
            docx_path = self.doc2dox(file_path)
            doc = Document(docx_path, doc_id)
            self.word_reader(doc, for_llm=for_llm, chunk_type=chunk_type)

        # TODO: convert to html
        html_path = None
        if self.docx_converter_endpoint:
            html_path = self._convert_docx_to_html(docx_path)

        if return_pdf_path:
            if html_path is None:
                raise ValueError(
                    "DOCX converter endpoint not configured; cannot return html_path with return_pdf_path."
                )
            pdf_path = send_file_to_docx2pdf(docx_path, docx_path.with_suffix(".pdf"))
            return doc, docx_path, pdf_path, html_path

        if return_docx_path:
            if html_path is None:
                raise ValueError(
                    "DOCX converter endpoint not configured; cannot return html_path with return_docx_path."
                )
            return doc, docx_path, html_path

        return doc
     