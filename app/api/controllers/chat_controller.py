# app/controllers/chat_controller.py

import os
from datetime import datetime, timezone
from typing import List, Optional
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
import uuid

import fitz  # PyMuPDF library
import asyncio

from utils import (
    OPENAI_API_KEY,
    MONGO_DB_NAME,
    MONGO_URI,
    )


class ChatController:
    def __init__(
        self,
        session_id: str,
        student_email: str,
        disciplina: str,
        qdrant_handler: QdrantHandler,
        image_handler: ImageHandler,
    ):
        print("Initializing ChatController")
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )
        print("LLM initialized")
        self.qdrant_handler = qdrant_handler
        self.image_handler = image_handler
        self.text_splitter = TextSplitter()
        self.embeddings = Embeddings().get_embeddings()
        self.chain = self._setup_chain()
        self.chain_with_history = self.__setup_chat_history()
        print("Chat history initialized")

        # Initialize MongoDB client and image collection
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]
        self.image_collection = self.db["image_collection"]  # Replace with your collection name
        print("MongoDB client initialized")

    def _setup_chain(self):
        print("Setting up chain")
        # Prompt to reformulate the question
        contextualize_q_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""

        # First part: contextualize the question
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("ai", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
        print(f"Contextualize prompt: {contextualize_q_prompt}")

        # Second part: answer the reformulated question
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a tutor who helps students develop critical thinking and solve problems on their own. 
                    The following is the reformulated version of the user question based on the chat history:""",
                ),
                ("ai", "{contextualized_question}"),  # Output of the reformulated question
                MessagesPlaceholder(variable_name="history"),  # Passes the chat history
                ("human", "Now answer the reformulated question: {contextualized_question}"),  # AI answers the reformulated question
            ]
        )
        print(f"Main prompt: {prompt}")

        # Debug function to print the output of the reformulated question
        def debug_output(output):
            # Prints the reformulated question
            reformulated_question = output.messages[-1].content
            print(f"Reformulated question: {reformulated_question}")

            # Returns values to continue processing
            return {
                "contextualized_question": reformulated_question,  # Extracts the reformulated question
                "history": output.messages  # Passes the message history
            }

        # Create a chain that processes the reformulated question and passes it to the next prompt
        chain = (
            contextualize_q_prompt
            | debug_output  # Function that prints and returns the output
            | prompt
            | self.llm  # Passes to the LLM to respond
        )
        return chain

    def __setup_chat_history(self):
        print("Setting up chat history")
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

    async def handle_user_message(self, user_input: Optional[str] = None, files=None):
        print("Handling user message")

        # Prepare the session ID configuration
        config = {"configurable": {"session_id": self.session_id}}

        try:
            # Check if files are provided
            if files:
                print("Processing uploaded files")
                await self._process_files(files)
                response = "Files have been successfully processed and added to the database."
                return response

            if user_input:
                chat_history = self.chain_with_history.get_session_history(self.session_id)

                print(f"User input: {user_input}")
                inputs = {
                    "input": user_input,
                    "chat_history": chat_history  # Ensure this is properly formatted and passed
                }
                # Invoke the chain asynchronously with the user's input and session history
                result = await self.chain_with_history.ainvoke(
                    inputs,
                    config=config,
                )
                print(f"Result: {result}")

                # Extract the AI response
                if isinstance(result, AIMessage):
                    response = result.content
                else:
                    response = str(result)  # Convert to string if necessary

                return response

            else:
                # If neither user_input nor files are provided
                response = "No input provided."
                return response

        except Exception as e:
            print(f"Error handling message: {e}")
            return "An error occurred while processing your message."

    async def _process_files(self, files):
        """
        Processes the uploaded files: extracts text, images, and stores embeddings.
        :param files: List of files uploaded by the user.
        """
        print(f"Processing {len(files)} file(s)")
        for file in files:
            filename = file.filename
            print(f"Processing file: {filename}")
            content = await file.read()
            if filename.lower().endswith(".pdf"):
                print(f"Processing PDF file: {filename}")
                await self._process_pdf(content)
            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Processing image file: {filename}")
                await self._process_image(content)
            else:
                print(f"Unsupported file type: {filename}")

    async def _process_pdf(self, content):
        """
        Processes PDF files: extracts text and images, generates embeddings.
        :param content: Binary content of the PDF file.
        """
        print("Opening PDF document")
        # Open the PDF using PyMuPDF (fitz)
        pdf_document = fitz.open(stream=content, filetype="pdf")

        # Extract text from each page
        text = ""
        num_pages = len(pdf_document)
        print(f"PDF has {num_pages} pages")
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            page_text = page.get_text()
            print(f"Extracted text from page {page_num + 1}")
            text += page_text

        # Split the text into smaller documents
        documents = [Document(page_content=text)]
        text_docs = self.text_splitter.split_documents(documents)
        print(f"Split text into {len(text_docs)} documents")

        # Generate embeddings for the text
        for idx, doc in enumerate(text_docs):
            embedding = self.embeddings.embed_query(doc.page_content)
            self.qdrant_handler.add_document(
                student_email=self.student_email,
                disciplina=self.disciplina,
                content=doc.page_content,
                embedding=embedding,
                metadata={
                    "type": "text",
                    "student_email": self.student_email,
                    "disciplina": self.disciplina,
                },
            )
            print(f"Added document {idx + 1}/{len(text_docs)} to Qdrant")

        # Extract images from each page
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            print(f"Page {page_num + 1} has {len(image_list)} images")

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Process each extracted image
                print(f"Processing image {img_index + 1}/{len(image_list)} on page {page_num + 1}")
                await self._process_image(image_bytes)

        pdf_document.close()
        print("PDF processing complete")

    async def _process_image(self, content):
        """
        Processes image files: stores the image in MongoDB, generates description using VLM,
        stores embeddings, and references the UUID in Qdrant.
        :param content: Binary content of the image file.
        """
        print("Processing image")
        # Store the image in MongoDB
        image_uuid = str(uuid.uuid4())
        image_document = {
            "_id": image_uuid,
            "student_email": self.student_email,
            "disciplina": self.disciplina,
            "image_data": content,
            "timestamp": datetime.now(timezone.utc)
        }
        self.image_collection.insert_one(image_document)
        print(f"Image stored in MongoDB with UUID: {image_uuid}")

        # Get the description of the image using VLM
        img_base64 = self.image_handler.encode_image_bytes(content)
        description = self.image_handler.image_summarize(img_base64)  # Sem await
        print(f"Image description: {description}")

        # Generate embedding of the description
        embedding = self.embeddings.embed_query(description)

        # Store the description and embedding in Qdrant
        metadata = {
            "type": "image",
            "description": description,
            "image_uuid": image_uuid,  # Reference to the image UUID in MongoDB
            "student_email": self.student_email,
            "disciplina": self.disciplina,
        }
        self.qdrant_handler.add_document(
            student_email=self.student_email,
            disciplina=self.disciplina,
            content=description,  # Store the description as content
            embedding=embedding,
            metadata=metadata,
        )
        print("Image description stored in Qdrant with reference to MongoDB.")

    def retrieve_image_and_description(self, image_uuid):
        """
        Recupera a imagem do MongoDB e sua descrição armazenada no Qdrant com base no UUID da imagem.

        :param image_uuid: O UUID da imagem armazenada.
        :return: Dicionário contendo os dados da imagem e a descrição da imagem.
        """
        try:
            # Recuperar a imagem do MongoDB pelo UUID
            image_data = self.image_collection.find_one({"_id": image_uuid})

            if not image_data:
                print(f"Imagem com UUID {image_uuid} não encontrada no MongoDB.")
                return {"error": "Image not found in MongoDB"}

            # Imagem encontrada
            print(f"Imagem com UUID {image_uuid} recuperada do MongoDB.")
            image_bytes = image_data.get("image_data")

            # Recuperar a descrição da imagem do Qdrant usando o UUID como chave de metadados
            query = f"SELECT description FROM qdrant WHERE image_uuid = '{image_uuid}'"
            results = self.qdrant_handler.similarity_search(
                query=query,
                student_email=self.student_email,
                disciplina=self.disciplina,
                k=1,  # Esperamos um único resultado, já que UUIDs são únicos
            )

            if not results:
                print(f"Descrição da imagem com UUID {image_uuid} não encontrada no Qdrant.")
                return {"error": "Image description not found in Qdrant"}

            # Descrição encontrada
            image_description = results[0].get("content")
            print(f"Descrição da imagem com UUID {image_uuid} recuperada do Qdrant.")

            return {
                "image_bytes": image_bytes,
                "description": image_description
            }

        except Exception as e:
            print(f"Erro ao recuperar a imagem ou descrição: {e}")
            return {"error": "Failed to retrieve image or description"}

    def _retrieve_context(self, query):
        """
        Retrieves relevant documents from the vector store based on the query.
        :param query: The user's question.
        :return: List of relevant documents' content.
        """
        print(f"Retrieving context for query: {query}")
        embedding = self.embeddings.embed_query(query)
        results = self.qdrant_handler.similarity_search(
            embedding,
            student_email=self.student_email,
            disciplina=self.disciplina,
            k=5,
        )
        print(f"Retrieved {len(results)} relevant documents")
        context = [result['content'] for result in results]
        return context

    async def _generate_response(self, user_input, context):
        """
        Generates the assistant's response using the LLM and the retrieved context.
        :param user_input: The user's input text.
        :param context: List of relevant documents.
        :return: Assistant's response.
        """
        print("Generating response")
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

        # Prepare the message history
        history_messages = self.chain_with_history.get_session_history(self.session_id).messages

        prompt = prompt_template.format(
            history=history_messages,
            input=user_input + "\n\nContexto:\n" + "\n".join(context),
        )
        print(f"Prompt prepared for LLM: {prompt}")

        response = await self.llm.agenerate([prompt])
        response_text = response.generations[0].text.strip()
        print(f"LLM response: {response_text}")

        return response_text

    def _save_message(self, role, content):
        """
        Saves a message to the chat history.
        :param role: 'user' or 'assistant'.
        :param content: Message content.
        """
        print(f"Saving message: role={role}, content={content}")
        if role == "user":
            message = BaseMessage(content=content, role="human")
        else:
            message = BaseMessage(content=content, role="ai")

        self.chain_with_history.get_session_history(self.session_id).add_message(message)



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