from FisReader.document_factory import DocumentFactory
from ChunkExtractor.chunk_extractor import ChunkExtractor
import pathlib, json
file_path = pathlib.Path("example/Nghị-định-133-2026-NĐ-CP.docx")
doc_id = "1"

document = DocumentFactory(ocr_endpoint="http://10.15.136.31:8999", docx_converter_endpoint="http://10.15.136.50:9700").read(file_path, doc_id)
with open("tree_test.json", "w", encoding='utf8') as f:
    json.dump(document.tree_structure, f, ensure_ascii=False, indent=4)

f.close()

chunk_extractor = ChunkExtractor()

chunks = chunk_extractor.get_chunks_in_tree(document, doc_id, file_path, "data.docx")
print(chunks[0])
for chunk in chunks[-5:]:
    print("TABLE CHUNK:")
    print(chunk.get("text"))
    print("============")
