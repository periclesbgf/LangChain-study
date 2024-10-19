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
from database.mongo_database_manager import MongoDatabaseManager, CustomMongoDBChatMessageHistory
from database.vector_db import TextSplitter, Embeddings, QdrantHandler
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, message_to_dict, messages_from_dict, AIMessage, HumanMessage

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
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from agent.prompt import AGENT_CHAT_PROMPT
from agent.agents import RetrievalAgent, ChatAgent
from langchain_core.output_parsers.json import JsonOutputParser
import fitz  # PyMuPDF library
import asyncio

from utils import (
    OPENAI_API_KEY,
    MONGO_DB_NAME,
    MONGO_URI,
    )

PROMPT_AGENTE_ORQUESTRADOR = """
Você é o Agente Orquestrador de um sistema de aprendizado que utiliza múltiplos agentes especializados. 
Sua função é analisar o histórico da conversa, o input do usuário e o plano de execução para transformar a pergunta em uma versão que possa ser respondida sem a necessidade de todo o histórico. 
Além disso, você deve determinar quais agentes serão ativados para responder à pergunta, seguindo a sequência pré-definida.

---

## Função do Agente Orquestrador
1. **Transformar a Pergunta**:
   - Analise o histórico da conversa e o input do usuário.
   - Reformule a pergunta para que ela possa ser respondida diretamente, sem a necessidade de todo o histórico anterior.
   - Verifique se a pergunta pode ser respondida pela LLM

2. **Decidir os Agentes Necessários**:
   - Identifique quais agentes serão ativados para responder à pergunta, seguindo a sequência:
     1. **Retrieval Agent (opcional)**: Ativado **se a LLM detectar que o estudante quer esclarecer dúvidas** sobre algum material enviado, como slides da aula ou atividades submetidas ou algo que ele quer que voce procure na internet. Ou se a LLM nao conseguir responder a pergunta.
        Função: Recuperar o material relevante para a pergunta do estudante e fazer busca na internet sobre materiais ou duvidas.
     2. **Agente Analista de Progresso**: Sempre ativado para garantir que a sessão esteja alinhada com o plano de execução e monitorar o aprendizado do estudante.
     3. **Agente de Chat**: Interage diretamente com o estudante e fornece a resposta final de forma clara e personalizada.

---

## Entrada
- **Plano de Execução**
- **Histórico da Conversa**
- **Input do Usuário**

---

## Saída (Formato JSON)
A saída deve ser um **JSON** contendo a pergunta reformulada e os agentes necessários para responder à pergunta.


  "pergunta_reformulada": "pergunta_reformulada",
  "agentes_necessarios": [

      "agente": "Retrieval Agent",
      "necessario": value

      "agente": "Agente Analista de Progresso",
      "necessario": true

      "agente": "Agente de Chat",
      "necessario": true

  ]


---

## Tarefas do Agente Orquestrador
1. **Analisar o histórico e o input**: Reformule a pergunta do estudante para que ela possa ser respondida de forma clara e objetiva.
2. **Determinar a sequência de agentes**: Ative os agentes necessários com base na natureza da pergunta.
3. **Verificar a necessidade do Retrieval Agent**: Ative-o se o estudante estiver perguntando sobre materiais que ele mesmo enviou.
4. **Gerar Saída JSON**: Produza uma saída organizada e clara no formato JSON, conforme especificado.

---

## Exemplo de Decisão do Orquestrador
- **Pergunta simples sobre conteúdo**:  
  - "O que é uma função recursiva?"  
  - **Agentes ativados**:  
    - **Agente Analista de Progresso**  
    - **Agente de Chat**

- **Pergunta sobre material enviado**:  
  - "No slide da aula 3, você pode explicar o exemplo do loop for?"  
  - **Agentes ativados**:  
    - **Retrieval Agent** para recuperar o slide da aula 3.  
    - **Agente Analista de Progresso** para garantir que o plano esteja sendo seguido.  
    - **Agente de Chat** para fornecer a explicação solicitada.

---

## Missão do Agente Orquestrador
Garanta que todas as perguntas sejam reformuladas corretamente e que a sequência adequada de agentes seja ativada para cada interação. 
Sua função é coordenar os agentes sem responder diretamente ao estudante, assegurando que cada passo esteja alinhado ao plano de execução e promovendo o progresso contínuo.


"""



class ChatController:
    def __init__(
        self,
        session_id: str,
        student_email: str,
        disciplina: str,
        qdrant_handler: QdrantHandler,
        image_handler: ImageHandler,
        retrieval_agent: RetrievalAgent,
        chat_agent: ChatAgent
    ):
        print("Initializing ChatController")
        self.session_id = session_id

        # Inicializa o LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        self.student_email = student_email
        self.disciplina = disciplina
        # Carrega o perfil e plano do estudante dos arquivos JSON
        self.perfil = self._carregar_json("/home/pericles/project/LangChain-study/app/resources/context_test.json")
        self.plano_execucao = self._carregar_json("/home/pericles/project/LangChain-study/app/resources/plano_acao.json")

        # Configura dependências adicionais
        self.qdrant_handler = qdrant_handler
        self.image_handler = image_handler
        self.text_splitter = TextSplitter()
        self.embeddings = Embeddings().get_embeddings()

        # Configuração das cadeias (chains)
        self.chain = self._setup_chain()
        self.chain_with_history = self.__setup_chat_history()

        # Inicializa o cliente MongoDB
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]
        self.image_collection = self.db["image_collection"]
        print("MongoDB client initialized")
        self.retrieval_agent = retrieval_agent
        self.chat_agent = chat_agent
        self.chat_history = CustomMongoDBChatMessageHistory(
            user_email=self.student_email,
            disciplina=self.disciplina,
            connection_string=MONGO_URI,
            session_id=session_id,
            database_name=MONGO_DB_NAME,
            collection_name="chat_history",
            session_id_key="session_id",
            history_key="history",
        )

    def _carregar_json(self, caminho_arquivo: str):
        """Carrega dados de um arquivo JSON."""
        caminho_absoluto = os.path.abspath(caminho_arquivo)
        try:
            with open(caminho_absoluto, 'r', encoding='utf-8') as arquivo:
                return json.load(arquivo)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao carregar {caminho_arquivo}: {e}")
            return {}

    def _setup_chain(self):
        print("Setting up chain")

        # Prompt principal usando o perfil e plano de ação
        main_prompt = ChatPromptTemplate.from_messages(
            [

                ("system", "Perfil do estudante: {perfil}"),
                ("system", "Plano de ação: {plano}"),
                MessagesPlaceholder(variable_name="history"),
                (
                    "system",
                    PROMPT_AGENTE_ORQUESTRADOR
                ),
                ("human", "{input}")  # Entrada do usuário
            ]
        )

        feedback_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Você é um analista de progresso de sessao de estudo, sua tarefa e analisar o historico de interacao entre uma LLM e um estudante.
                    Dado o historico de interacao, voce deve verificar se o estudante esta progredindo de acordo com o plano de acao.
                    Se nao estiver, voce deve fornecer feedback ao estudante e ao tutor sobre o que pode ser melhorado."""
                ),
                ("system", "Perfil do estudante: {perfil}"),
                ("system", "Plano de ação: {plano}"),
                MessagesPlaceholder(variable_name="history"),  # Placeholder para o histórico de mensagens
            ]
        )
        feedback_chain = (
            feedback_prompt
            | self.llm  # Chama o modelo de linguagem para processar o prompt
        )
        json_output_parser = JsonOutputParser()
        # Criação da cadeia (chain)
        chain = (
            main_prompt
            | self.llm
            | json_output_parser
        )
        return chain

    async def handle_user_message(self, user_input: Optional[str] = None, files=None):
        print("Handling user message")
        config = {"configurable": {"session_id": self.session_id}}

        try:
            if files:
                print("Processing uploaded files")
                self._process_files(files)
                return "Files processed and added to the database."

            if user_input:
                # Obter o histórico e preparar a entrada
                chat_history = self.chain_with_history.get_session_history(self.session_id).messages
                inputs = {
                    "input": user_input,
                    "chat_history": chat_history,
                    "perfil": self.perfil,
                    "plano": self.plano_execucao
                }

                # Salvar a mensagem do usuário no histórico
                self.chat_history.add_message(HumanMessage(content=user_input))

                # Invocar a cadeia do orquestrador
                result = await self.chain_with_history.ainvoke(inputs, config=config)
                print(f"Orchestrator Result: {result}")

                orchestrator_response = result if isinstance(result, dict) else json.loads(result)
                pergunta_reformulada = orchestrator_response.get("pergunta_reformulada")
                agentes_necessarios = orchestrator_response.get("agentes_necessarios", [])

                # Verificar se o RetrievalAgent é necessário
                retrieval_needed = any(
                    agente["agente"] == "Retrieval Agent" and agente["necessario"]
                    for agente in agentes_necessarios
                )

                if retrieval_needed:
                    print("Retrieval Agent activated")
                    retrieval_response = self.retrieval_agent.invoke(
                        query=pergunta_reformulada,
                        student_profile=self.perfil,
                        execution_plan=self.plano_execucao
                    )
                    print(f"Retrieval Agent Response: {retrieval_response}")

                    # Extrair o campo 'output' e salvar no histórico
                    output = retrieval_response.get('output', 'No relevant output found.')
                    self.chat_history.add_message(AIMessage(content=output))

                    return output

                # Salvar a pergunta reformulada como resposta do assistente
                self.chat_history.add_message(AIMessage(content=pergunta_reformulada))

                return pergunta_reformulada or "Pergunta não reformulada disponível."

            return "No input provided."

        except Exception as e:
            print(f"Error handling message: {e}")
            return "An error occurred while processing your message."

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

    def _process_files(self, files):
        """
        Processes the uploaded files: extracts text, images, and stores embeddings.
        :param files: List of files uploaded by the user.
        """
        print(f"Processing {len(files)} file(s)")
        for file in files:
            filename = file.filename
            print(f"Processing file: {filename}")
            content = file.read()
            if filename.lower().endswith(".pdf"):
                print(f"Processing PDF file: {filename}")
                self._process_pdf(content)
            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Processing image file: {filename}")
                self._process_image(content)
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

    @tool
    def retrieve_context(self, query):
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


