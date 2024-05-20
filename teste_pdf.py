
import datetime
import io
import json
import textwrap
from io import BytesIO

import openai
import pandas as pd
import PyPDF2
from langchain.schema import Document
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)


def pdf_bytes_to_documents_metadata(nome_arquivo: str, chave_arquivo: str, pdf_bytes: bytes) -> list:
    
    buffer = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(buffer)
    num_pages = len(pdf_reader.pages)
    palavras_total = 0
    bytes_total = 0
    docs = []
    document_metadata = pdf_reader.metadata
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        page_text = remover_caracteres_especiais(remover_espacos_duplos(text))

        metadata = {
            "página": page_num+1,
            "arquivo": nome_arquivo,
            "chave_arquivo": chave_arquivo,
            "título": document_metadata.title if document_metadata and document_metadata.title is not None else "desconhecido",
            "autor": document_metadata.author if document_metadata and document_metadata.author is not None else "desconhecido"
        }
     
        docs.append(Document(page_content=page_text, metadata=metadata))

        palavras = page_text.split()
        palavras_total = palavras_total + len(palavras)
        bytes_total = bytes_total + len(page_text)

    return docs, num_pages, palavras_total, bytes_total



def teste():
    narrative_texts = [elem for elem in elements if elem.category == "NarrativeText"]

    for index, elem in enumerate(narrative_texts[:5]):
        print(f"Narrative text {index + 1}:")
        print("\n".join(textwrap.wrap(elem.text, width=70)))
        print("\n" + "-" * 70 + "\n")

