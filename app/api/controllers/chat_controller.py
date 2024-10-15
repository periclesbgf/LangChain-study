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
        print("Histórico de chat inicializado")

    def _setup_chain(self):
        # Prompt para reformular a pergunta para o estilo pirata
        contextualize_q_system_prompt = """You are given a chat history and the latest user question.
        Your task is to always reformulate the question as if it were being asked by a pirate. 
        Use pirate phrases, such as 'Arrr', 'Ahoy', 'Matey', and similar expressions.
        For every question, no matter the subject, you must change the tone to pirate-speak.

        For example:
        - Original: 'What is the weather today?'
        - Pirate: 'Arrr, what be the weather today, matey?'

        - Original: 'Who discovered America?'
        - Pirate: 'Arrr, who be discoverin' the lands across the sea, ye landlubber?'

        - Original: 'Where is the library?'
        - Pirate: 'Ahoy, where be the library, matey?'

        Now, reformulate the user's question in pirate speak."""
        
        # Primeira parte: contextualizar a pergunta
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("ai", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
        print("Contextualize prompt", contextualize_q_prompt)

        # Segunda parte: responder à pergunta reformulada
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a tutor who helps students develop critical thinking and solve problems on their own. 
                    The following is the reformulated version of the user question based on the chat history:""",
                ),
                ("ai", "{contextualized_question}"),  # Output da pergunta reformulada
                MessagesPlaceholder(variable_name="history"),  # Passa o histórico de chat
                ("human", "Now answer the reformulated question: {contextualized_question}"),  # IA responde à pergunta reformulada
            ]
        )
        print("Prompt", prompt)

        # Função para depuração: imprime o output da pergunta reformulada
        def debug_output(output):
            # Imprime a mensagem reformulada
            reformulated_question = output.messages[-1].content
            print(f"Reformulated question: {reformulated_question}")

            # Retorna os valores para continuar o processamento
            return {
                "contextualized_question": reformulated_question,  # Extrai a pergunta reformulada
                "history": output.messages  # Passa o histórico de mensagens
            }

        # Criar uma cadeia que processa a pergunta reformulada e passa ao próximo prompt
        chain = (
            contextualize_q_prompt
            | debug_output  # Função que imprime e retorna o output
            | prompt
            | self.llm  # Passa para o LLM responder
        )
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

    async def handle_user_message(self, user_input: str, files=None):
        print("Answering user message")

        # Prepare the session ID configuration
        config = {"configurable": {"session_id": self.session_id}}

        try:
            # Retrieve the chat history for the session from MongoDB
            chat_history = self.chain_with_history.get_session_history(self.session_id)

            # Prepare input for the chain (user input and history)
            inputs = {
                "input": user_input,
                "chat_history": chat_history  # Ensure this is properly formatted and passed
            }

            # Invoke the chain asynchronously with the user's input and session history
            result = await self.chain_with_history.ainvoke(inputs, config=config)
            print("Result:", result)
            response = result.content
            # Extract the AI response
            # if isinstance(result, AIMessage):
            #     response = result.content
            # else:
            #     response = str(result)  # Convert to string if necessary

            return response

        except Exception as e:
            print(f"Error handling message: {e}")
            return "An error occurred while processing your message."



    def _contextualize_question(self, question, chat_history):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        chain = contextualize_q_prompt | self.llm
        return chain

    def _does_need_context(self, user_input):
        # Verifica se dado uma pergunta, precisa de contexto a ser recuperado
        pass

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



META_PROMPT = """
Dada uma descrição de tarefa ou um prompt existente, produza um prompt de sistema detalhado para guiar um modelo de linguagem a completar a tarefa de maneira eficaz.

# Diretrizes

- Entenda a Tarefa: Compreenda o principal objetivo, metas, requisitos, restrições e a saída esperada.
- Alterações Mínimas: Se um prompt existente for fornecido, melhore-o apenas se for simples. Para prompts complexos, melhore a clareza e adicione elementos ausentes sem alterar a estrutura original.
- Raciocínio Antes das Conclusões**: Incentive etapas de raciocínio antes de chegar a conclusões. ATENÇÃO! Se o usuário fornecer exemplos onde o raciocínio ocorre depois, INVERTA a ordem! NUNCA COMECE EXEMPLOS COM CONCLUSÕES!
    - Ordem do Raciocínio: Identifique as partes de raciocínio do prompt e as partes de conclusão (campos específicos pelo nome). Para cada uma, determine a ORDEM em que isso é feito e se precisa ser invertido.
    - Conclusões, classificações ou resultados devem SEMPRE aparecer por último.
- Exemplos: Inclua exemplos de alta qualidade, se forem úteis, usando placeholders [entre colchetes] para elementos complexos.
   - Que tipos de exemplos podem precisar ser incluídos, quantos e se são complexos o suficiente para se beneficiar de placeholders.
- Clareza e Concisão: Use linguagem clara e específica. Evite instruções desnecessárias ou declarações genéricas.
- Formatação: Use recursos do markdown para legibilidade. NÃO USE ``` BLOCO DE CÓDIGO A MENOS QUE SEJA ESPECIFICAMENTE SOLICITADO.
- Preserve o Conteúdo do Usuário: Se a tarefa de entrada ou o prompt incluir diretrizes ou exemplos extensos, preserve-os inteiramente ou o mais próximo possível. Se forem vagos, considere dividir em subetapas. Mantenha quaisquer detalhes, diretrizes, exemplos, variáveis ou placeholders fornecidos pelo usuário.
- Constantes: Inclua constantes no prompt, pois não são suscetíveis a injeções de prompt. Tais como guias, rubricas e exemplos.
- Formato de Saída: Explique explicitamente o formato de saída mais apropriado, em detalhes. Isso deve incluir comprimento e sintaxe (por exemplo, frase curta, parágrafo, JSON, etc.)
    - Para tarefas que produzem dados bem definidos ou estruturados (classificação, JSON, etc.), dê preferência à saída em formato JSON.
    - O JSON nunca deve ser envolvido em blocos de código (```) a menos que explicitamente solicitado.

O prompt final que você gera deve seguir a estrutura abaixo. Não inclua comentários adicionais, apenas gere o prompt completo do sistema. ESPECIFICAMENTE, não inclua mensagens adicionais no início ou no fim do prompt. (por exemplo, sem "---")

[Instrução concisa descrevendo a tarefa - esta deve ser a primeira linha do prompt, sem cabeçalho de seção]

[Detalhes adicionais conforme necessário.]

[Seções opcionais com títulos ou listas para etapas detalhadas.]

# Etapas [opcional]

[opcional: um detalhamento das etapas necessárias para realizar a tarefa]

# Formato de Saída

[Especificamente, aponte como a saída deve ser formatada, seja o comprimento da resposta, estrutura, por exemplo, JSON, markdown, etc.]

# Exemplos [opcional]

[Opcional: 1-3 exemplos bem definidos com placeholders, se necessário. Marque claramente onde os exemplos começam e terminam e qual é a entrada e saída. Use placeholders conforme necessário.]
[Se os exemplos forem mais curtos do que o esperado para um exemplo real, faça uma referência com () explicando como exemplos reais devem ser mais longos / curtos / diferentes. E USE PLACEHOLDERS!]

# Notas [opcional]

[opcional: casos extremos, detalhes e uma área para repetir considerações importantes específicas]
""".strip()




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