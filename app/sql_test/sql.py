import fitz  # PyMuPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
import logging
from fastapi.logger import logger as fastapi_logger
from app.chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from app.database.search import execute_query
from app.database.vector_db import DocumentLoader, TextSplitter, Embeddings, QdrantIndex
from app.audio.text_to_speech import AudioService
from fastapi.logger import logger
from app.utils import OPENAI_API_KEY, CODE


class CommandChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o-mini"
        self.prompt_template = """
        1. Você é um assistente virtual chamado Éden
        2. Seu papel é ler e entender o conteudo de um PDF que será desorganizado
        3. Retornar de maneira clara e objetiva o conteúdo do PDF de maneira organizada
        4. Dizer qual é o curso do estudante e todas as aulas, seu conteudo e tambem o dia que acontecera essa aula
        5. No fim de sua resposta, diga quais tabelas sao necessarias para guardar essas informacoes em um banco de dados relacional

        Dado o contexto acima, responda o texto a seguir: {text}
        """

    def setup_chain(self, text, history):
        history.add_user_message(text)
        history_messages = history.get_history()

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        response = chain.invoke(history_messages)
        history.add_assistant_message(response)

        return response, history

    def query_openai_api(self, text):
        prompt = self.prompt_template.format(text=text)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        response = llm.invoke(prompt)
        return response


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = "/Users/peric/projects/LangChain-study/PLANO_ENSINO_PIF_TURMA_B.pdf"
text = extract_text_from_pdf(pdf_path)
print(text)

import re

def transform_program_table(text):
    program_data = []
    # Use expressão regular para encontrar o bloco de texto da tabela "PROGRAMA"
    program_block = re.search(r"PROGRAMA:\s+(.*?)\s+CESAR School", text, re.DOTALL).group(1).strip()

    # Divida o bloco de texto em linhas
    lines = program_block.split("\n")
    
    for line in lines:
        # Use expressão regular para extrair dados das linhas
        match = re.match(r"(\d+)\s+(\d{2}/\d{2}/\d{4})\s+(\d+)\s+(\d+)\s+(.*)\s+Computador.*Sala de Aula Virtual", line)
        if (match):
            encontro, data, carga_teorica, carga_pratica, conteudo = match.groups()
            program_data.append({
                "encontro": int(encontro),
                "data": data,
                "carga_teorica": int(carga_teorica),
                "carga_pratica": int(carga_pratica),
                "conteudo": conteudo.strip()
            })
    
    return program_data

program_data = transform_program_table(text)
print(program_data)


command_chain = CommandChain(api_key=OPENAI_API_KEY)

resposta = command_chain.query_openai_api(text)

# Exiba a resposta obtida da API
print("Resposta da API da OpenAI:", resposta)
