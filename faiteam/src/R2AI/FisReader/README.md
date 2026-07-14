# OCR Engine Package

This is FisReader package that extracts basic informations from text file (docx and pdf).

## Install Locally

First pull LegalTreeConstruct and install

```
git clone -b docx-process git@github.com:NlpFisTeam/LegalTreeConstruct.git
cd LegalTreeConstruct
pip install -e .
```
After git clone, you can access the codebase and simply run the following command line:

```
cd FisReader
pip install -e .
```

## Example Usage

After installing, You can view examples/example.py to get usage of FisReader.

You can run ```example/example.py``` file as following:

```
python examples/sample.py -p your_data_path -ep your_ocr_service_endpoint
```

## Basic Usage

You can import DocumentFactory from FisReader.

```
from pathlib import Path
from FisReader.document_factory import DocumentFactory
document = DocumentFactory(ocr_endpoint=your_ocr_service_endpoint).read(Path(your_file_path), doc_id=1)
```
