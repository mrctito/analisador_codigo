import asyncio
import json
import os
import sys
from pathlib import Path

# Bibliotecas específicas
from dotenv import load_dotenv  # Carrega variáveis de ambiente de um arquivo .env
from git import Repo  # Manipula repositórios Git
# Estruturas para criar correntes de processamento de linguagem
from langchain.chains import ConversationalRetrievalChain, LLMChain
# Parser para documentos de linguagem de programação
from langchain.document_loaders.parsers import LanguageParser
from langchain.memory import (ChatMessageHistory, ConversationBufferMemory,
                              ConversationSummaryMemory)  # Gerenciamento de memória de conversação
from langchain.prompts import PromptTemplate  # Templates para prompts de IA
# Ferramentas para dividir textos em segmentos
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
# Modelos de chat usando OpenAI
from langchain_community.chat_models import ChatOpenAI
# Carregador genérico de documentos
from langchain_community.document_loaders.generic import GenericLoader
# Armazenamento vetorial usando Chroma
from langchain_community.vectorstores import Chroma
# Ferramenta para gerar embeddings usando OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel  # Modelo de dados usando Pydantic


def cria_banco_vetorial(repo_path: Path, chunk_size: int, chunk_overlap: int, persist_directory: str):
    """
    Função para criar um banco de dados vetorial (VectorDB) a partir de um repositório de código Python.
    O banco de dados vetorial permite buscar similaridades entre textos processados.

    Parâmetros:
    - repo_path: Caminho para o repositório de código-fonte.
    - chunk_size: Tamanho dos pedaços (chunks) de texto em que o código será dividido.
    - chunk_overlap: Sobreposição entre os chunks de texto.
    - persist_directory: Diretório onde o banco vetorial será salvo.
    """
    print(f"Carregando arquivos de {repo_path}")
    # Lista todos os arquivos .py no diretório
    py_files = list(repo_path.glob("*.py"))
    print(f"Arquivos py: {len(py_files)}")

    # Configura um carregador genérico para arquivos Python
    loader = GenericLoader.from_filesystem(
        str(repo_path),
        glob="*.py",  # Define o padrão de busca para arquivos .py
        suffixes=[".py"],  # Define os sufixos de arquivos a serem processados
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        show_progress=True,
    )

    # Carrega os documentos do repositório
    documents = loader.load()

    # Divide os documentos em chunks utilizando um divisor recursivo
    documents_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Divide os documentos em textos menores
    texts = documents_splitter.split_documents(documents)
    print(f"Textos separados: {len(texts)}")

    # Gera embeddings (representações vetoriais) para os textos usando OpenAI
    embeddings = OpenAIEmbeddings(
        disallowed_special=(), openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Cria o banco vetorial (VectorDB) usando Chroma e persiste os dados
    vectordb = Chroma.from_documents(
        texts, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()  # Salva o banco vetorial em disco
    return vectordb  # Retorna o banco vetorial criado


def cria_chat(vectordb):
    """
    Função para criar um sistema de chat baseado em IA com memória de conversação.
    A IA utiliza o modelo GPT-4 para interagir com os usuários.

    Parâmetros:
    - vectordb: O banco vetorial criado na função anterior, que permite buscas contextuais durante o chat.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini",
                     openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True)

    # Cria uma memória de conversação que resume as interações anteriores
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True)

    # Cria uma corrente de processamento para o chat com recuperação de contexto
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 8}), memory=memory)
    return qa  # Retorna a corrente de processamento do chat


def main():
    """
    Função principal que inicializa o ambiente, cria o banco vetorial e inicia o chat interativo com o usuário.
    """
    print("Analisador de Código Fonte")
    load_dotenv()  # Carrega variáveis de ambiente a partir de um arquivo .env

    # Define o caminho do repositório, tamanho dos chunks e diretório de persistência
    repo_path = Path(__file__).resolve().parent
    chunk_size = 2000
    chunk_overlap = 200
    persist_directory = './data'

    # Cria o banco vetorial e a corrente de chat
    vectordb = cria_banco_vetorial(
        repo_path, chunk_size, chunk_overlap, persist_directory)
    qa = cria_chat(vectordb)

    # Loop de interação contínua com o usuário
    while True:
        pergunta = input("\n================\nPode perguntar! ")
        if pergunta == ".":  # Condição de saída do loop
            break
        try:
            # Processa a pergunta usando o sistema de chat baseado em IA
            resposta = qa.invoke(pergunta)
            print("\n\n***\n\nResposta:\n\n" + resposta['answer'])
        except Exception as e:
            # Trata erros que possam ocorrer durante o processamento
            print(f"Ocorreu um erro: {e}")


if __name__ == "__main__":
    main()  # Executa a função principal se o script for rodado diretamente
