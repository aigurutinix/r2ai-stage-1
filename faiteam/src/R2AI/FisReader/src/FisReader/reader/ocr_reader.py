import os
import re

import requests

from FisReader.document import LAW_TYPES_LIST, Article, Document
from readyocr.entities import Paragraph, Figure
from readyocr.parsers.readyocr_parser import load
from docx import Document as DocxDocument

class OCRReader:
    def __init__(self, ocr_endpoint: str) -> None:
        self.ocr_endpoint = ocr_endpoint

    def extract_created_date(self, text_elements, document: "Document") -> Document:
        for i in range(0, min(10, len(text_elements))):
            match = re.search(r"ngày(.*)tháng(.*)năm(.*)", text_elements[i])

            if match:
                day, month, year = match.groups()
                day = "".join([char for char in day if char.isdigit()])
                month = "".join([char for char in month if char.isdigit()])
                year = "".join([char for char in year if char.isdigit()])
                document.created_date = f"{day}/{month}/{year}"

                print(f"created date: {document.created_date}")
                break
        return document

    def extract_code(self, text_elements, document: "Document"):
        for i in range(0, min(20, len(text_elements))):
            match = re.search(r"SỐ:(.*$)", text_elements[i].upper())

            if match:
                document.code = text_elements[i][match.start(1) : match.end(1)].replace(" ", "")
                break
        return document

    def extract_type_and_title(self, text_elements, document: "Document") -> Document:
        print("start extract")
        for i in range(0, min(20, len(text_elements))):
            if text_elements[i].lower() in LAW_TYPES_LIST:
                document.law_type = text_elements[i]

                for j in range(i + 1, min(50, len(text_elements))):
                    if text_elements[j].lower()[:6] == "căn cứ":
                        document.title = " ".join(text_elements[i + 1 : j])
                        break

                if document.title == "":
                    print("debug: ", text_elements)
                    document.title = " ".join(text_elements[i + 1 : i + 3])

                break

        if document.title == "":
            for i in range(0, min(20, len(text_elements))):
                match = re.search(r"V\/v\s+(.*)", text_elements[i])

                if match:
                    document.law_type = "CÔNG VĂN"
                    document.title = match.group(1)
                    break
        return document

    def article_chunking(self, text, document: Document) -> Document:
        articles = []
        article = Article(doc_id=document.doc_id, code=document.code, law_type=document.law_type)
        appendix = None

        index = 0

        for paragraph in text:
            match = re.search(r"^(Điều\s+\d+)\.", paragraph["text"][:12])

            if match:
                if article:
                    articles.append(article)

                number = match.group(1)
                article = Article(
                    doc_id=document.doc_id,
                    title_article=paragraph["text"][len(number) + 2 :],
                    code=document.code,
                    law_type=document.law_type,
                    number_article=number,
                )

            elif article:
                if "Nơi nhận:" in paragraph["text"][:12]:
                    articles.append(article)
                    article = Article(
                        doc_id=document.doc_id, code=document.code, law_type=document.law_type
                    )

                article.text_ocr_list.append(paragraph)

            if len(paragraph["text"]) <= 20 and paragraph["text"][:7].lower() == "phụ lục":
                if article:
                    articles.append(article)
                    article = None

                if appendix:
                    articles.append(appendix)

                title = paragraph["text"]

                if (index + 1) < len(text):
                    title = text[index + 1]["text"]

                appendix = Article(
                    doc_id=document.doc_id,
                    title_article=title,
                    code=document.code,
                    law_type=document.law_type,
                    number_article=paragraph["text"],
                )
            else:
                if appendix:
                    appendix.text_ocr_list.append(paragraph)

            index += 1

        if article:
            articles.append(article)

        if appendix:
            articles.append(appendix)

        document.articles = articles
        return document

    def extract(self, document: "Document") -> None:
        document.is_ocr = 1

        files = {"file": open(document.file_path, "rb")}
        params = {
            'pdf_2_layer': 'true'
        }

        r = requests.post(self.ocr_endpoint, files=files, params=params)

        if r.status_code != 200:
            raise Exception(f"OCR service failed: {r.text}")

        ocr_response = r.json()["result"]

        pages = load(ocr_response).pages
        
        ocr_text = []

        for page in pages:
            paragraphs = page.descendants.filter_by_class(Paragraph)
            # figures = page.descendants.filter_by_class(Figure)
            text_per_page = []

            # image_index = 0

            for paragraph in paragraphs:
                x = paragraph.bbox.x
                y = paragraph.bbox.y
                w = paragraph.bbox.width
                h = paragraph.bbox.height

                coords = [x, y, w, h]

                text_per_page.append({"text": "\n" + paragraph.text.strip(), "coords": coords, "page": paragraph.page_number, "page_size": {"width": page.width, "height": page.height}})

            # for figure in figures:
            #     image_index += 1
            #     x = figure.bbox.x
            #     y = figure.bbox.y
            #     w = figure.bbox.width
            #     h = figure.bbox.height

            #     coords = [x, y, w, h]

            #     text_per_page.append({"text": f"\n|<image_{image_index}>|", "coords": coords, "page": figure.page_number, "page_size": {"width": page.width, "height": page.height}})

            text_per_page = sorted(text_per_page, key=lambda x: x["coords"][1], reverse=False)
            ocr_text.extend(text_per_page)

        first_page = [x["text"] for x in ocr_text if x["page"] == 1]

        document = self.extract_code(first_page, document)
        document = self.extract_type_and_title(first_page, document)
        document = self.extract_created_date(first_page, document)

        if document.law_type and document.law_type.lower() in LAW_TYPES_LIST:
            document = self.article_chunking(ocr_text, document=document)

        document.content = "\n".join([t["text"] for t in ocr_text])
        document.content_ocr = ocr_text

        # open("temp.md", "w", encoding="utf8").write(document.content)

        # docx = DocxDocument()
        # docx.add_paragraph("\n".join([t["text"] for t in ocr_text]))
        # docx.save("temp.docx")

        return document

    def extract_old(self, document: "Document") -> None:
        document.is_ocr = 1

        files = {"image": open(document.file_path, "rb")}
        r = requests.post(self.ocr_endpoint, files=files)

        if r.status_code != 200:
            raise Exception("OCR service failed")

        ocr_response = r.json()["data"]

        ocr_text = []

        for item in ocr_response:
            for t in item["raw_text"]:
                t["text"] = t["text"].replace("[B]", " ")
                ocr_text.append({"text": t["text"].strip(), "coords": t["coords"], "page": item["page"]})

        first_page = [item["text"] for item in ocr_response[0]["raw_text"]]

        document = self.extract_code(first_page, document)
        document = self.extract_type_and_title(first_page, document)
        document = self.extract_created_date(first_page, document)

        if document.law_type and document.law_type.lower() in LAW_TYPES_LIST:
            document = self.article_chunking(ocr_text, document=document)

        document.content = "\n".join([t["text"] for t in ocr_text])
        document.content_ocr = ocr_text
        return document
