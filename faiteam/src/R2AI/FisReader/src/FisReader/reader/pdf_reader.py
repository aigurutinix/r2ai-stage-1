from pydoc import doc
import re

import pymupdf

from FisReader.document import LAW_TYPES_LIST, Article, Document
from FisReader.reader.ocr_reader import OCRReader


class PDFReader:
    def __init__(self, ocr_endpoint) -> None:
        self.ocr_reader = OCRReader(ocr_endpoint)

    def extract_code(self, document, text_elements):
        for i in range(0, len(text_elements)):
            match = re.search(r"SỐ:\s*([A-ZĐƯƠĂÂÊ0-9/-]+)", text_elements[i].upper())

            if match:
                document.code = text_elements[i][match.start(1) : match.end(1)]
                break
        return document

    def extract_created_date(self, document, text_elements):
        for i in range(0, min(20, len(text_elements))):
            match = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text_elements[i])

            if match:
                day, month, year = match.groups()
                document.created_date = f"{day}/{month}/{year}"
                break
        return document

    def extract_type_and_title(self, document, text_elements):
        for i in range(0, min(20, len(text_elements))):
            if text_elements[i].lower() in LAW_TYPES_LIST:
                document.law_type = text_elements[i]

                for j in range(i + 1, min(50, len(text_elements))):
                    if text_elements[j].lower()[:6] == "căn cứ":
                        document.title = " ".join(text_elements[i + 1 : j])
                        break

                if document.title == "":
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
    def article_chunking(self, document, text):
        articles = []
        article = Article(doc_id=document.doc_id, code=document.code, law_type=document.law_type)
        appendix = None

        index = 0

        for paragraph in text:
            match = re.search(r"^(Điều\s+\d+)\.", paragraph[:12])

            if match:
                if article:
                    article.text = article.text[:-1]
                    articles.append(article)

                number = match.group(1)
                article = Article(
                    doc_id=document.doc_id,
                    title_article=paragraph[len(number) + 2 :],
                    code=document.code,
                    law_type=document.law_type,
                    number_article=number,
                )

            elif article:
                if "Nơi nhận:" in paragraph.strip()[:12]:
                    article.text = article.text[:-1]
                    articles.append(article)
                    article = Article(doc_id=document.doc_id, code=document.code, law_type=document.law_type)

                article.text = article.text + paragraph + " "

            if len(paragraph) <= 20 and paragraph[:7].lower() == "phụ lục":
                if article:
                    article.text = article.text[:-1]
                    articles.append(article)
                    article = None

                if appendix:
                    appendix.text = appendix.text[:-1]
                    articles.append(appendix)

                title = paragraph

                if (index + 1) < len(text):
                    title = text[index + 1]

                appendix = Article(
                    doc_id=document.doc_id,
                    title_article=title,
                    code=document.code,
                    law_type=document.law_type,
                    number_article=paragraph,
                )
            else:
                if appendix:
                    appendix.text = appendix.text + paragraph + " "

            index += 1

        if article:
            article.text = article.text[:-1]
            articles.append(article)

        if appendix:
            appendix.text = appendix.text[:-1]
            articles.append(appendix)

        document.articles = articles
        return document

    def extract(self, document, for_llm=False):
        pdf = pymupdf.open(document.file_path)

        image_pages = 0

        for page in pdf:
            img_refs = page.get_image_info(xrefs=True)
            if img_refs != []:
                image_pages += 1

        if image_pages / len(pdf) >= 0.4:
            print("have to OCR")
            document = self.ocr_reader.extract(document)
            return document

        text_elements = []
        for page in pdf:
            text = page.get_text()
            text = [" ".join(line.split()) for line in text.split("\n") if line.strip()]
            text_elements.extend(text)

            if page.number == 0:
                document = self.extract_code(document, text)

        document = self.extract_type_and_title(document, text_elements)
        document = self.extract_created_date(document, text_elements)

        if document.law_type and document.law_type.lower() in LAW_TYPES_LIST:
            document = self.article_chunking(document, text_elements)

        document.content = " ".join(text_elements)
        return document
    
    def __call__(self, document, for_llm):
        return self.extract(document=document, for_llm=for_llm)        
