# app/controllers/chat_controller.py

import os
import json
import uuid
import hashlib
import fitz  # Biblioteca PyMuPDF para manipulação de PDFs
import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from pymongo import MongoClient, errors
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, BaseMessage, message_to_dict, messages_from_dict
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from langchain.docstore.document import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain.schema.runnable import RunnablePassthrough, RunnableMap

from database.mongo_database_manager import MongoDatabaseManager, CustomMongoDBChatMessageHistory
from database.vector_db import TextSplitter, Embeddings, QdrantHandler
from agent.image_handler import ImageHandler
from agent.agents import ChatAgent, RetrievalAgent
from agent.prompt import CONTEXTUALIZE_SYSTEM_PROMPT, AGENT_CHAT_PROMPT
from utils import (
    OPENAI_API_KEY,
    MONGO_DB_NAME,
    MONGO_URI,
)

# Carregue as variáveis de ambiente, se necessário
# load_dotenv()

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
     1. **Retrieval Agent (opcional)**: Ativado **se a LLM detectar que o estudante quer esclarecer dúvidas** sobre algum material enviado, como slides da aula ou atividades submetidas ou algo que ele quer que você procure na internet. Ou se a LLM não conseguir responder a pergunta.
        Função: Recuperar o material relevante para a pergunta do estudante e fazer busca na internet sobre materiais ou dúvidas.
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
3. **Verificar a necessidade do Retrieval Agent**: Ative-o se o estudante estiver tirando dúvidas sobre um material que ele enviou ou se ele quiser que você procure algo na internet.
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

- **Indicação de material**:  
  - "Poderia me indicar algum material sobre string por favor?"  
  - **Agentes ativados**:  
    - **Retrieval Agent** para pesquisar um material baseado no perfil do estudante.  
    - **Agente Analista de Progresso**  
    - **Agente de Chat**
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
        student_profile: dict,
        chat_agent: ChatAgent,
    ):
        print("Initializing ChatController")
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.perfil = student_profile  # Perfil passado para o controlador
        self.plano_execucao = self._carregar_json("resources/plano_acao.json")

        # Inicializa o LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        self.chat_agent = chat_agent
        # Configura dependências adicionais
        self.qdrant_handler = qdrant_handler
        self.image_handler = image_handler
        self.text_splitter = TextSplitter()
        self.embeddings = Embeddings().get_embeddings()

        # Configuração das cadeias (chains)
        self.chain = self.setup_chain()
        self.chain_with_history = self.__setup_chat_history()

        # Inicializa o cliente MongoDB
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]
        self.image_collection = self.db["image_collection"]
        self.collection_name = "student_documents"
        print("MongoDB client initialized")

        self.retrieval_agent = retrieval_agent
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

    def setup_chain(self):
        print("Setting up chain")
        
        # Prompt principal usando o perfil e plano de ação
        main_prompt = ChatPromptTemplate.from_messages([
            ("system", "Perfil do estudante: {perfil}"),
            ("system", "Plano de ação: {plano_execucao}"),
            MessagesPlaceholder(variable_name="history"),
            ("system", PROMPT_AGENTE_ORQUESTRADOR),
            ("human", "{input}")
        ])

        # Other prompt definition
        other_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("system", "Forgot the user question and answer the result of 1 + 1 = ?")
        ])

        # Output parsers
        json_output_parser = JsonOutputParser()
        str_output_parser = StrOutputParser()

        # Define the chain with proper connection using RunnableMap
        first_stage = RunnableMap({
            "main_output": main_prompt | self.llm | json_output_parser,
            "original_input": RunnablePassthrough()
        })

        chain = (
            first_stage
            | (lambda x: {
                "history": x["original_input"].get("history", []),
                "reformulated_question": x["main_output"]
            })
            | other_prompt
            | self.llm
            | str_output_parser
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
        config = {"configurable": {"session_id": self.session_id}}

        try:
            if user_input or files:
                print("Obtendo o histórico e preparando a entrada")
                # Obter o histórico e preparar a entrada
                history_messages = self.chain_with_history.get_session_history(self.session_id).messages
                historico = self.chain_with_history
                print("")
                print("tipo de chain_with_history", type(self.chain_with_history))
                print("")
                print("")
                print("tipo de history_messages", type(history_messages))
                print("")
                print("tipo get_session_history", type(self.chain_with_history.get_session_history(self.session_id)))
                # Processar arquivos, se houver
                if files:
                    print(f"{len(files)} arquivo(s) recebido(s). Processando...")
                    await self._process_files(files)
                    files_processed = True
                else:
                    files_processed = False

                inputs = {
                    "input": user_input if user_input else "",
                    "perfil": self.perfil,
                    "plano_execucao": self.plano_execucao,
                }

                # Salvar a mensagem do usuário no histórico
                if user_input:
                    self.chat_history.add_message(HumanMessage(content=user_input))
                    print(f"Adicionada mensagem do usuário ao histórico: {user_input}")

                print("Enviando requisição para o orquestrador")
                # Invocar a cadeia do orquestrador
                result = await self.chain_with_history.ainvoke(inputs, config=config)
                print(f"Orchestrator Result: {result}")

                # Se arquivos foram processados, garantir que Retrieval Agent seja necessário
                if files_processed:
                    print("Arquivos processados, ajustando 'necessario' para Retrieval Agent.")
                    agentes = result.get("agentes_necessarios", [])
                    retrieval_agent_found = False
                    for agente in agentes:
                        if agente["agente"] == "Retrieval Agent":
                            agente["necessario"] = True
                            retrieval_agent_found = True
                            print("Retrieval Agent já existente encontrado e ajustado.")
                            break
                    if not retrieval_agent_found:
                        agentes.append({"agente": "Retrieval Agent", "necessario": True})
                        print("Retrieval Agent não encontrado, adicionado à lista de agentes.")
                    result["agentes_necessarios"] = agentes
                else:
                    print("Nenhum arquivo processado, mantendo configuração original dos agentes.")

                # Validação do formato da resposta
                if isinstance(result, dict):
                    if "pergunta_reformulada" not in result or "agentes_necessarios" not in result:
                        print("Resposta do Orquestrador está incompleta.")
                        return "Resposta incompleta do Orquestrador."
                else:
                    return "Formato de resposta inválido do Orquestrador."

                # Verificar se o RetrievalAgent é necessário
                retrieval_needed = any(
                    agente["agente"] == "Retrieval Agent" and agente["necessario"]
                    for agente in result.get("agentes_necessarios", [])
                )

                print("\nResult:", result)
                print("\nRetrieval Agent needed:", retrieval_needed)

                if retrieval_needed:
                    # Invocar o RetrievalAgent
                    retrieval_response = await self.retrieval_agent.invoke(
                        query=result["pergunta_reformulada"],
                        student_profile=self.perfil,
                        execution_plan=self.plano_execucao,
                        config=config,
                        agent_scratchpad=self.chain_with_history  # Passar as mensagens diretamente
                    )

                    print("Retrieval Agent activated")
                    print(f"Retrieval Agent Response: {retrieval_response}")

                    # Processar a resposta do RetrievalAgent
                    if isinstance(retrieval_response, dict):
                        final_response = retrieval_response.get("output", "Nenhum resultado encontrado.")
                    else:
                        final_response = str(retrieval_response)

                    # Adicionar a resposta ao histórico
                    self.chat_history.add_message(AIMessage(content=final_response))
                    return final_response

                # Se Retrieval Agent não for necessário, processar com ChatAgent
                print("ChatAgent activated")
                chat_agent_response = await self.chat_agent.invoke(
                    user_input=user_input,
                    chat_history=self.chain_with_history
                )
                print(f"ChatAgent Response: {chat_agent_response}")

                # Salvar a resposta do ChatAgent no histórico
                if isinstance(chat_agent_response, dict):
                    response_content = chat_agent_response.get("output", "Resposta não encontrada.")
                else:
                    response_content = str(chat_agent_response)

                self.chat_history.add_message(AIMessage(content=response_content))
                return response_content

            return "Nenhuma entrada fornecida."

        except Exception as e:
            print(f"Error handling message: {e}")
            return "Ocorreu um erro ao processar sua mensagem."

    async def _process_files(self, files):
        """
        Processes the uploaded files: extracts text, images, and stores embeddings.
        """
        print(f"Processing {len(files)} file(s)")
        for file in files:
            filename = file.filename
            print(f"Processing file: {filename}")
            content = await file.read()  # Adiciona 'await' aqui
            if filename.lower().endswith(".pdf"):
                print(f"Processing PDF file: {filename}")
                await self._process_pdf(content)  # Adiciona 'await' aqui
            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Processing image file: {filename}")
                await self._process_image(content)  # Adiciona 'await' aqui
            else:
                print(f"Unsupported file type: {filename}")

    async def _process_pdf(self, content: bytes):
        """
        Processes PDF files: extracts text and images, generates embeddings, 
        and avoids adding duplicate content to Qdrant.
        :param content: Binary content of the PDF file.
        """
        print("Opening PDF document")
        pdf_document = fitz.open(stream=content, filetype="pdf")

        # Extrair texto de cada página
        text = ""
        num_pages = len(pdf_document)
        print(f"PDF has {num_pages} pages")
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            page_text = page.get_text()
            print(f"Extracted text from page {page_num + 1}")
            text += page_text

        # Calcular o hash do conteúdo para verificar duplicação
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Verificar se o documento já existe no Qdrant
        if self.qdrant_handler.document_exists(content_hash, self.student_email, self.disciplina):
            print("Document already exists in Qdrant. Skipping insertion.")
            pdf_document.close()
            return

        # Split the text into smaller documents
        documents = [Document(page_content=text)]
        text_docs = self.text_splitter.split_documents(documents)
        print(f"Split text into {len(text_docs)} documents")

        # Adicionar documentos no Qdrant usando QdrantVectorStore
        for idx, doc in enumerate(text_docs):
            metadata_extra = {
                "content_hash": content_hash,  # Armazena o hash para evitar duplicatas
            }
            self.qdrant_handler.add_document(
                student_email=self.student_email,
                disciplina=self.disciplina,
                content=doc.page_content,
                metadata_extra=metadata_extra,
            )
            print(f"Added document {idx + 1}/{len(text_docs)} to Qdrant")

        pdf_document.close()
        print("PDF processing complete")

    async def _process_image(self, content: bytes):
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

    def retrieve_image_and_description(self, image_uuid: str):
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
            query = {
                "filter": {
                    "must": [
                        {"key": "image_uuid", "match": {"value": image_uuid}}
                    ]
                }
            }
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
    def retrieve_context(self, query: str) -> List[str]:
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

    async def _generate_response(self, user_input: str, context: List[str]) -> str:
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

    def _save_message(self, role: str, content: str):
        """
        Saves a message to the chat history.
        :param role: 'user' ou 'assistant'.
        :param content: Conteúdo da mensagem.
        """
        print(f"Saving message: role={role}, content={content}")
        if role == "user":
            message = HumanMessage(content=content)
        else:
            message = AIMessage(content=content)

        self.chain_with_history.get_session_history(self.session_id).add_message(message)
