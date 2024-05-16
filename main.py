import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from git import Repo
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.document_loaders.parsers import LanguageParser
from langchain.memory import (ChatMessageHistory, ConversationBufferMemory,
                              ConversationSummaryMemory)
from langchain.prompts import PromptTemplate
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel


def cria_banco_vetorial():

  repo_path = Path(__file__).resolve().parent
  print(str(repo_path))

  py_files = list(repo_path.glob("*.py"))
  py_files_count = len(py_files)
  print(f"Arquivos py: {py_files_count}")
  
  loader = GenericLoader.from_filesystem(str(repo_path),
                                        glob = "*.py",
                                        suffixes=[".py"],
                                        parser = LanguageParser(language=Language.PYTHON, parser_threshold=500),
                                        show_progress=True,
    )
    
  documents = loader.load()
  len(documents)


  documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                             chunk_size = 2000,
                                                             chunk_overlap = 200)

  texts = documents_splitter.split_documents(documents)
  len(texts)

  #text-embedding-ada-002
  embeddings=OpenAIEmbeddings(disallowed_special=(), openai_api_key=os.getenv("OPENAI_API_KEY"))

  vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory='./data')
  vectordb.persist()

  return vectordb

def cria_chat(vectordb):
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True)
    memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)
    return qa


def main():
  print("Analisador de Código Fonte")
  load_dotenv()
  vectordb = cria_banco_vetorial()
  qa = cria_chat(vectordb)

  while True:
      print("\n")
      pergunta = input("O que você quer saber? ")
      if pergunta == ".":
        break
      resposta = qa.invoke(pergunta)
      print("\n\n***\n\nResposta:\n\n"+resposta['answer'])


if __name__ == "__main__":
  main()


