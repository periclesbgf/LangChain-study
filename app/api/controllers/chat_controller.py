# app/controllers/chat_controller.py

import os
from datetime import datetime, timezone
from typing import List
import json
from pymongo import errors
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, BaseMessage, message_to_dict, messages_from_dict
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from langchain.docstore.document import Document
from database.vector_db import TextSplitter, Embeddings, QdrantHandler
from agent.image_handler import ImageHandler
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage
from bs4 import BeautifulSoup
from database.mongo_database_manager import MongoDatabaseManager
from database.vector_db import TextSplitter, Embeddings, QdrantHandler
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, message_to_dict, messages_from_dict, AIMessage

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage

from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from agent.prompt import CONTEXTUALIZE_SYSTEM_PROMPT
from datetime import datetime, timezone
from typing import List
import json
from pymongo import MongoClient, errors

import fitz  # PyMuPDF library
import asyncio

from utils import (
    OPENAI_API_KEY,
    MONGO_DB_NAME,
    MONGO_URI,
    )

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

class CustomMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    def __init__(self, user_email: str, disciplina: str, *args, **kwargs):
        self.user_email = user_email
        self.disciplina = disciplina
        super().__init__(*args, **kwargs)
        # Cria um índice em (session_id, user_email, timestamp) para otimizar consultas
        self.collection.create_index(
            [
                (self.session_id_key, 1),
                ("user_email", 1),
                ("timestamp", 1)
            ],
            name="session_user_timestamp_index",
            unique=False
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to MongoDB with user_email, disciplina, and timestamp."""
        try:
            self.collection.insert_one(
                {
                    self.session_id_key: self.session_id,
                    self.history_key: json.dumps(message_to_dict(message)),
                    "user_email": self.user_email,
                    "disciplina": self.disciplina,
                    "timestamp": datetime.now(timezone.utc)
                }
            )
        except errors.WriteError as err:
            print(f"Error adding message: {err}")

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Append multiple messages to MongoDB with user_email, disciplina, and timestamp."""
        try:
            documents = [
                {
                    self.session_id_key: self.session_id,
                    self.history_key: json.dumps(message_to_dict(message)),
                    "user_email": self.user_email,
                    "disciplina": self.disciplina,
                    "timestamp": datetime.now(timezone.utc)
                }
                for message in messages
            ]
            self.collection.insert_many(documents)
        except errors.WriteError as err:
            print(f"Error adding messages: {err}")

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages filtered by session_id and user_email, sorted by timestamp."""
        try:
            cursor = self.collection.find(
                {
                    self.session_id_key: self.session_id,
                    "user_email": self.user_email
                }
            ).sort("timestamp", 1)
        except errors.OperationFailure as error:
            print(f"Error retrieving messages: {error}")
            return []

        items = [json.loads(document[self.history_key]) for document in cursor]
        messages = messages_from_dict(items)
        return messages

class ChatController:
    def __init__(
        self,
        session_id: str,
        student_email: str,
        disciplina: str,
        qdrant_handler: QdrantHandler,
        image_handler: ImageHandler,
    ):
        print("Inicializando ChatController")
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )
        print("LLM inicializado")
        self.qdrant_handler = qdrant_handler
        self.image_handler = image_handler
        self.text_splitter = TextSplitter()
        self.embeddings = Embeddings().get_embeddings()
        self.chain = self._setup_chain()
        self.chain_with_history = self.__setup_chat_history()
        print("Histórico de chat inicializado")

    def _setup_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a tutor who helps students develop critical thinking and solve problems on their own.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | self.llm
        return chain

    def __setup_chat_history(self):
        chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: CustomMongoDBChatMessageHistory(
                user_email=self.student_email,
                disciplina=self.disciplina,
                connection_string=MONGO_URI,
                session_id=session_id,
                database_name=MONGO_DB_NAME,
                collection_name="chat_history",
                session_id_key="session_id",
                history_key="history",
            ),
            input_messages_key="input",
            history_messages_key="history",
        )
        return chain_with_history

    async def handle_user_message(self, user_input: str, file = None):
        print("Answering user message")

        # Configure the session ID
        config = {"configurable": {"session_id": self.session_id}}

        try:
            # Invoke the chain with the user's input and session ID
            result = await self.chain_with_history.ainvoke(
                {"input": user_input},
                config=config,
            )
            print("Result:", result)

            # Check if the result is an AIMessage
            if isinstance(result, AIMessage):
                response = result.content
            else:
                response = str(result)  # Convert to string if necessary

            return response

        except Exception as e:
            print(f"Error handling message: {e}")
            return "An error occurred while processing your message."

    async def _process_files(self, files):
        """
        Processa os arquivos enviados: extrai texto, imagens e armazena embeddings.
        :param files: Lista de arquivos enviados pelo usuário.
        """
        for file in files:
            filename = file.filename
            content = await file.read()
            if filename.lower().endswith(".pdf"):
                await self._process_pdf(content)
            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                await self._process_image(content)
            else:
                print(f"Tipo de arquivo não suportado: {filename}")

    async def _process_pdf(self, content):
        """
        Processa arquivos PDF: extrai texto e imagens, gera embeddings.
        :param content: Conteúdo binário do arquivo PDF.
        """
        import io

        # Abre o PDF usando PyMuPDF (fitz)
        pdf_document = fitz.open(stream=content, filetype="pdf")

        # Extrai texto de cada página
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()

        # Divide o texto em documentos menores
        documents = [Document(page_content=text)]
        text_docs = self.text_splitter.split_documents(documents)

        # Gera embeddings para o texto
        for doc in text_docs:
            embedding = self.embeddings.embed_query(doc.page_content)
            self.qdrant_handler.add_document(
                student_email=self.student_email,
                disciplina=self.disciplina,
                content=doc.page_content,
                embedding=embedding,
                metadata={"type": "text"},
            )

        # Extrai imagens de cada página
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Processa cada imagem extraída
                await self._process_image(image_bytes)

        pdf_document.close()

    async def _process_image(self, content):
        """
        Processa arquivos de imagem: gera descrição usando VLM, armazena embeddings.
        :param content: Conteúdo binário do arquivo de imagem.
        """
        # Obtém a descrição da imagem usando VLM
        img_base64 = self.image_handler.encode_image_bytes(content)
        description = await self.image_handler.image_summarize(img_base64)
        # Gera embedding da descrição
        embedding = self.embeddings.embed_query(description)
        # Armazena a imagem (em base64) e o embedding
        self.qdrant_handler.add_document(
            student_email=self.student_email,
            disciplina=self.disciplina,
            content=img_base64,
            embedding=embedding,
            metadata={"type": "image", "description": description},
        )

    def _retrieve_context(self, query):
        """
        Recupera documentos relevantes do banco vetorial com base na query.
        :param query: Texto de entrada do usuário.
        :return: Lista de documentos relevantes.
        """
        embedding = self.embeddings.embed_query(query)
        results = self.qdrant_handler.similarity_search(
            embedding,
            student_email=self.student_email,
            disciplina=self.disciplina,
            k=5,
        )
        context = [result['content'] for result in results]
        return context

    async def _generate_response(self, user_input, context):
        """
        Gera a resposta do assistente usando o LLM e o contexto recuperado.
        :param user_input: Texto de entrada do usuário.
        :param context: Lista de documentos relevantes.
        :return: Resposta do assistente.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Você é um tutor que ajuda os estudantes a desenvolver pensamento crítico "
                    "e resolver problemas por conta própria. Use o contexto fornecido para "
                    "auxiliar o estudante.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        # Prepara o histórico de mensagens
        history_messages = self.chat_history.messages

        prompt = prompt_template.format(
            history=history_messages,
            input=user_input + "\n\nContexto:\n" + "\n".join(context),
        )
        response = await self.llm.agenerate([prompt])
        response_text = response.generations[0].text.strip()

        return response_text

    def _save_message(self, role, content):
        """
        Salva uma mensagem no histórico de chat.
        :param role: 'user' ou 'assistant'.
        :param content: Conteúdo da mensagem.
        """
        if role == "user":
            message = BaseMessage(content=content, role="human")
        else:
            message = BaseMessage(content=content, role="ai")

        self.chat_history.add_message(message)


    # def _setup_retriever(self):
    #     print("Setting up retriever")

    #     # Dividir os documentos
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    #     splits = text_splitter.split_documents(docs)
    #     print("Documents split")
    #     # Criar o vetor de embeddings
    #     vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    #     retriever = vectorstore.as_retriever()
    #     print("Retriever created")
    #     return retriever

    # def _setup_qa_chain(self):
    #     print("Setting up QA chain")
    #     # Prompt para contextualizar a pergunta
    #     contextualize_q_system_prompt = """Given a chat history and the latest user question \
    #     which might reference context in the chat history, formulate a standalone question \
    #     which can be understood without the chat history. Do NOT answer the question, \
    #     just reformulate it if needed and otherwise return it as is."""
    #     contextualize_q_prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", contextualize_q_system_prompt),
    #             MessagesPlaceholder("chat_history"),
    #             ("human", "{input}"),
    #         ]
    #     )
    #     print("Contextualize prompt created")
    #     # Prompt para responder a pergunta
    #     qa_system_prompt = """Você é um tutor educacional que ajuda os estudantes a desenvolver pensamento crítico \
    #     e resolver problemas por conta própria. Use os seguintes trechos de contexto recuperados para responder à pergunta. \
    #     Se não souber a resposta, diga que não sabe. Use no máximo três frases e mantenha a resposta concisa.

    #     {context}"""
    #     print("QA system prompt created")
    #     qa_prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", qa_system_prompt),
    #             MessagesPlaceholder("chat_history"),
    #             ("human", "{input}"),
    #         ]
    #     )
    #     print("QA prompt created")
    #     # Criar a cadeia de perguntas e respostas com recuperação
    #     qa_chain = ConversationalRetrievalChain.from_llm(
    #         llm=self.llm,
    #         retriever=self.retriever,
    #         condense_question_prompt=contextualize_q_prompt,
    #         combine_docs_chain_kwargs={"prompt": qa_prompt}
    #     )
    #     print("QA chain created")
    #     return qa_chain

    # async def handle_user_message(self, session_id: str, user_input: str):
    #     print("Handling user message")
    #     # Recuperar o histórico de conversa
    #     chat_history = await self.db_manager.get_chat_history(session_id)
    #     print(chat_history)
    #     # Converter o histórico para o formato necessário
    #     history = []
    #     for msg in chat_history:
    #         history.append(msg)

    #     # Obter a resposta do modelo
    #     result = self.qa_chain(
    #         {"question": user_input, "chat_history": history}
    #     )

    #     response = result["answer"]
    #     print(response)
    #     # Salvar a mensagem do usuário e a resposta do assistente
    #     await self.db_manager.save_message(session_id, "user", user_input)
    #     await self.db_manager.save_message(session_id, "assistant", response)

    #     return response



