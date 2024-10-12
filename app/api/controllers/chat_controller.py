# app/controllers/chat_controller.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import ConversationalRetrievalChain
from bs4 import BeautifulSoup
from database.mongo_database_manager import MongoDatabaseManager

from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from agent.prompt import CONTEXTUALIZE_SYSTEM_PROMPT
from langchain.schema import BaseMessage, message_to_dict, messages_from_dict, AIMessage
from datetime import datetime, timezone
from typing import List
import json
from pymongo import MongoClient, errors

from utils import (
    OPENAI_API_KEY,
    MONGO_DB_NAME,
    MONGO_URI,
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
    def __init__(self, session_id: str, student_email: str, disciplina: str):
        print("Initializing ChatController")
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        print("LLM initialized")
        self.chain = self.__setup_chain()
        self.chain_with_history = self.__setup_chat_history()
        print("Chat history initialized")

    def __setup_chain(self):
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

    async def handle_user_message(self, user_input: str):
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

    def summarize_messages(self, chain_input):
        stored_messages = self.chain_with_history.get_session_history
        print("Stored messages:", stored_messages)
        if len(stored_messages) == 0:
            return False
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
                ),
            ]
        )
        summarization_chain = summarization_prompt | self.llm

        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
        print("Summary message:", summary_message)
        # self.chain_with_history.runnable.

        # self.chain_with_history.add_message(summary_message)

        return True



    # def _setup_retriever(self):
    #     print("Setting up retriever")
    #     # Carregar documentos (exemplo com WebBaseLoader)
    #     loader = WebBaseLoader(
    #         web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
    #         bs_kwargs={"features": "html.parser"}
    #     )
    #     docs = loader.load()
    #     print("Documents loaded")
    #     # Dividir os documentos
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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



