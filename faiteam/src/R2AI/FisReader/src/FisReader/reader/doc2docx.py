import os
from pathlib import Path
import re
import subprocess

import time
from io import BytesIO

import requests

from dotenv import load_dotenv

load_dotenv()

def send_file_to_doc2docx(file_path: Path, docx_path: Path):
        file_bytesio = BytesIO()
        with open(file_path, "rb") as f:
            file_bytesio.write(f.read())
        file_bytesio.seek(0)
        api_endpoint = os.getenv("DOC2DOCX_ENDPOINT", "http://10.15.82.36:9700/convert")
        headers = {
            "accept": "application/json",
        }
        files = {
            "file": (
                "content.doc",
                file_bytesio.read(),
                "application/msword",
            ),
        }
        response = requests.post(url=api_endpoint, headers=headers, files=files)
        if response.status_code == 200:
            content = BytesIO(response.content)
            # save to docx_path
            with open(docx_path, "wb") as f:
                f.write(content.read())
            return docx_path
        else:
            print(response.json())
            raise ValueError(response.json())

def send_file_to_docx2pdf(file_path: Path, pdf_path: Path):
        file_bytesio = BytesIO()
        with open(file_path, "rb") as f:
            file_bytesio.write(f.read())
        file_bytesio.seek(0)
        api_endpoint = os.getenv("DOCX2PDF_ENDPOINT", "http://10.15.82.36:9700/docx2pdf")
        headers = {
            "accept": "application/json",
        }
        files = {
            "file": (
                "content.docx",
                file_bytesio.read(),
                "application/msword",
            ),
        }
        response = requests.post(url=api_endpoint, headers=headers, files=files)
        if response.status_code == 200:
            content = BytesIO(response.content)
            # save to pdf_path
            with open(pdf_path, "wb") as f:
                f.write(content.read())
            return pdf_path
        else:
            print(response.json())
            raise ValueError(response.json())