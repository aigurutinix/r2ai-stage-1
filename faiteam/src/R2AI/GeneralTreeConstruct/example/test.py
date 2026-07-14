from FisReader.reader.docx_reader import WordReader
from FisReader.document_factory import DocumentFactory
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from GeneralChunkExtractor.chunk_extract import GeneralChunkExtractor
import time

start_time = time.time()
file_path = Path("example/content.docx")
reader = WordReader()
factory = DocumentFactory()
doc = reader.extract_general(file_path, doc_id="1")
# doc_ = factory.read(file_path=file_path, doc_id="1", for_llm=True)

import json
# print(doc.tree_structure)
print(json.dumps(doc.tree_structure, indent=4, ensure_ascii=False))
# print(json.dumps(doc_.tree_structure, indent=4, ensure_ascii=False))
