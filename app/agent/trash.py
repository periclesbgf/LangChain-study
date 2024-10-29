# # app/agent/agents.py

# from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool, initialize_agent, create_react_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.tools import tool
# from database.mongo_database_manager import CustomMongoDBChatMessageHistory
# from utils import OPENAI_API_KEY
# from database.vector_db import QdrantHandler, Embeddings
# from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
# from langchain.schema import BaseMessage, AIMessage, message_to_dict, messages_from_dict
# from pymongo import errors
# from datetime import datetime, timezone
# import json
# from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.tools import tool
# from langchain.tools import tool
# from youtubesearchpython import VideosSearch
# import wikipediaapi
# from serpapi import GoogleSearch
# from pydantic import BaseModel
# from langchain.agents.react.output_parser import ReActOutputParser
# from functools import partial
# from langchain.agents import (
#     Agent,
#     AgentExecutor,
#     AgentOutputParser,
#     BaseMultiActionAgent,
#     BaseSingleActionAgent,
#     LLMSingleActionAgent,
# )

# from utils import OPENAI_API_KEY

# class CustomOutputParser:
#     """Parser personalizado para lidar com saídas do agente."""
#     def parse(self, output: str):
#         if "Observation:" in output:
#             # Se a resposta contiver "Observation:", continue o fluxo
#             return {"action": "continue", "observation": output}
#         # Caso contrário, retorne a saída final
#         return {"final_output": output}

# RETRIEVAL_PROMPT = """
# ### Contexto:
# Você é o **Agente de Recuperação de Conteúdo**, especializado em fornecer respostas precisas e personalizadas com base nas necessidades e preferências do estudante. Seu objetivo é identificar e sugerir recursos educativos úteis e alinhados ao contexto fornecido, utilizando as ferramentas disponíveis.

# ### Suas Responsabilidades:
# 1. **Buscar Recursos Relevantes**:
#    - Consulte o banco vetorial e outros recursos para localizar os materiais mais adequados.
#    - Priorize sempre a recuperação de contexto relevante (retrieve_context) antes de recorrer a ferramentas externas, como Google ou Wikipedia.

# 2. **Fornecer Sugestões Personalizadas**:
#    - Adapte os recursos recomendados às necessidades específicas e ao perfil do estudante.
#    - Inclua vídeos, exemplos práticos e artigos que correspondam ao estilo de aprendizado e plano de execução do estudante.

# 3. **Apresentar Respostas de Forma Clara e Estruturada**:
#    - Organize as recomendações para que sejam fáceis de seguir e implementar.
#    - Se necessário, inclua orientações para ações práticas ou próximas etapas.

# ### Parâmetros de Entrada:
# - **Consulta do Usuário**: "{consulta_usuario}"
# - **Perfil do Estudante**: {perfil_do_estudante}
# - **Plano de Execução**: {plano_de_execucao}

# ### Fluxo de Execução:
# 1. **Análise Inicial**: Interprete a consulta do usuário e identifique quais recursos serão mais úteis.
# 2. **Seleção da Ferramenta**: Escolha a ferramenta adequada com base na consulta e nos recursos disponíveis.
# 3. **Pesquisa e Recuperação**: Aplique a ferramenta selecionada para encontrar a melhor resposta possível.
# 4. **Refinamento e Personalização**: Adapte as recomendações ao perfil e ao plano de execução do estudante.
# 5. **Resposta ao Usuário**: Formule uma resposta final clara, contendo os recursos encontrados e orientações específicas.

# ### Ferramentas Disponíveis:
# {tool_names}

# - Utilize as ferramentas fornecidas conforme necessário. 
# - Priorize **retrieve_context** para fornecer contexto relevante sempre que possível.

# ### Rascunho das Interações Realizadas:
# {agent_scratchpad}

# ### Lista Completa de Ferramentas:
# {tools}

# ---

# ### Nota Importante:
# - Sempre priorize a ferramenta `retrieve_context` para fornecer contexto educativo relevante antes de utilizar Google ou Wikipedia.
# - Garanta que todas as respostas sejam apresentadas de forma clara e direcionada ao objetivo do estudante.

# ---

# ### Exemplo de Fluxo:
# Question: {consulta_usuario}
# Thought: [Sua análise do que fazer]
# Action: [Nome da ação a ser realizada]
# Action Input: [Parâmetros da ação]
# Observation: [Resultado da ação]
# Thought: [Análise após observar o resultado]
# ... (Repita o ciclo, se necessário)
# Final Answer: [Resposta final ao usuário]

# - **Formato das Queries**: string  
# - **Siga Estritamente o Formato** para garantir consistência e clareza na resposta.
# """

# class RetrievalAgent:
#     """
#     Agente responsável por recuperar e sugerir recursos educacionais com base nas consultas do estudante.

#     Atributos:
#         model (ChatOpenAI): Modelo de linguagem usado para interações.
#         qdrant_handler: Handler para o banco vetorial Qdrant.
#         embeddings: Objeto para geração de embeddings de consultas.
#         agent (AgentExecutor): Agente que executa as tarefas definidas.
#         prompt (ChatPromptTemplate): Template do prompt para o agente.
#         output_parser (ReActOutputParser): Parser para verificar a saída do agente.
#         student_email (str): Email do estudante.
#         disciplina (str): Disciplina do estudante.
#     """

#     def __init__(self, qdrant_handler, embeddings, student_email: str, disciplina: str, model_name: str = "gpt-4o-mini"):
#         """
#         Inicializa o RetrievalAgent com os componentes necessários.
        
#         Args:
#             qdrant_handler: Handler para consultas ao banco vetorial Qdrant.
#             embeddings: Gerador de embeddings para consultas.
#             student_email (str): Email do estudante.
#             disciplina (str): Disciplina do estudante.
#             model_name (str): Nome do modelo de linguagem a ser utilizado.
#         """
#         self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
#         self.qdrant_handler = qdrant_handler
#         self.embeddings = embeddings
#         self.student_email = student_email
#         self.disciplina = disciplina

#         self.prompt = ChatPromptTemplate.from_template(RETRIEVAL_PROMPT)
#         self.output_parser = ReActOutputParser()

#         # Define wrapper functions para as ferramentas
#         def retrieve_context_tool(query: str) -> str:
#             return self.retrieve_context(query)
#         retrieve_context_tool.__name__ = "retrieve_context"

#         def search_youtube_tool(query: str) -> str:
#             return self.search_youtube(query)
#         search_youtube_tool.__name__ = "search_youtube"

#         def search_wikipedia_tool(query: str) -> str:
#             return self.search_wikipedia(query)
#         search_wikipedia_tool.__name__ = "search_wikipedia"

#         def search_google_tool(query: str) -> str:
#             return self.search_google(query)
#         search_google_tool.__name__ = "search_google"

#         # Cria a lista de ferramentas
#         tools = [
#             Tool(name="retrieve_context", func=retrieve_context_tool, description="Recupera contexto."),
#             Tool(name="search_youtube", func=search_youtube_tool, description="Busca vídeos no YouTube."),
#             Tool(name="search_wikipedia", func=search_wikipedia_tool, description="Pesquisa na Wikipedia."),
#             Tool(name="search_google", func=search_google_tool, description="Pesquisa no Google."),
#         ]

#         self.agent = create_react_agent(
#             llm=self.model,
#             tools=tools,
#             prompt=self.prompt,
#             output_parser=self.output_parser
#         )

#         self.agent_executor = AgentExecutor(
#             agent=self.agent,
#             tools=tools,
#             handle_parsing_errors=True,
#             verbose=True,
#             max_iterations=30,
#             return_intermediate_steps=True  # Retorna as etapas intermediárias para análise
#         )


#     def retrieve_context(self, query: str) -> str:
#         """
#         Recupera contexto relevante do banco vetorial Qdrant com base em uma consulta.
#         """
#         print(f"retrieve_context called with query: {query}")  # Debug

#         if not isinstance(query, str):
#             raise ValueError("O parâmetro 'query' deve ser uma string.")

#         embedding = self.embeddings.embed_query(query)
#         print(f"Generated embedding: {embedding[:5]}...")  # Debug - mostra parte do embedding

#         results = self.qdrant_handler.similarity_search(
#             embedding=embedding,
#             student_email=self.student_email,
#             disciplina=self.disciplina,
#             k=5,
#         )
#         print(f"Retrieved {len(results)} relevant documents")  # Debug

#         # Combine o conteúdo dos documentos recuperados
#         context = "\n".join([result['content'] for result in results])
#         return context or "Nenhum contexto relevante encontrado."

#     def search_youtube(self, query: str) -> str:
#         """
#         Busca o vídeo mais relevante no YouTube com base na consulta.
#         """
#         print(f"Searching YouTube for: {query}")  # Debug

#         from youtubesearchpython import VideosSearch
#         search = VideosSearch(query, limit=1)
#         result = search.result()
#         print(f"Search Result: {result}")  # Debug

#         if result['result']:
#             video = result['result'][0]
#             return f"Título: {video['title']}\nLink: {video['link']}"
#         return "Nenhum vídeo encontrado."

#     def search_wikipedia(self, query: str) -> str:
#         """
#         Realiza uma pesquisa na Wikipedia e retorna o resumo da página encontrada.
#         """
#         import wikipediaapi
#         wiki = wikipediaapi.Wikipedia('pt')
#         page = wiki.page(query)

#         if page.exists():
#             return f"Título: {page.title}\nResumo: {page.summary[:500]}"
#         return "Página não encontrada."

#     def search_google(self, query: str) -> str:
#         """
#         Realiza uma pesquisa no Google e retorna o primeiro resultado relevante.
#         """
#         from serpapi import GoogleSearch
#         search = GoogleSearch({"q": query, "api_key": "YOUR_SERPAPI_KEY"})
#         result = search.get_dict()
#         if "organic_results" in result and result["organic_results"]:
#             top_result = result["organic_results"][0]
#             return f"Título: {top_result['title']}\nLink: {top_result['link']}"
#         return "Nenhum resultado encontrado."

#     async def invoke(self, query: dict, student_profile, execution_plan):
#         try:
#             query_str = " ".join([f"{k}: {v}" for k, v in query.items()])
#             inputs = {
#                 "consulta_usuario": query_str,
#                 "perfil_do_estudante": student_profile,
#                 "plano_de_execucao": execution_plan,
#                 "input": query_str
#             }

#             print(f"Invoking agent with inputs: {inputs}")  # Debug

#             response = await self.agent_executor.ainvoke(inputs)
#             print(f"Raw response from agent: {response}")  # Debug

#             if not response:
#                 print("Agent did not return a valid response.")  # Debug
#                 return "Não foi possível gerar uma resposta."

#             return response.get("resultado", "Nenhum resultado encontrado.")
#         except Exception as e:
#             print(f"Error executing query: {e}")  # Debug
#             return f"Erro ao executar a consulta: {e}"

# PROMPT_AGENTE_ANALISE_PROGRESSO = """
# Você é o Agente de Análise de Progresso. Sua responsabilidade é avaliar o desempenho do estudante e fornecer feedback corretivo, se necessário.

# ### Responsabilidades:
# - **Avaliar o Progresso**: Verifique se o estudante está avançando conforme o plano de execução.
# - **Fornecer Feedback**: Identifique áreas de dificuldade e sugira melhorias.
# - **Ajustar o Plano**: Sinalize se o plano precisa ser revisado.

# ### Entrada:
# - **Histórico de Interações**: {historico_de_interacoes}
# - **Progresso Atual da Sessão**: {session_progress}
# - **Plano de Execução**: {plano_de_execucao}

# ### Tarefas:
# 1. **Analisar o Histórico**: Examine as interações para identificar padrões de dificuldade.
# 2. **Comparar com o Plano**: Verifique se o progresso está alinhado com os objetivos definidos.
# 3. **Fornecer Feedback**: Prepare um relatório com sugestões e observações.

# **Exemplo de Feedback**:
# "O estudante tem demonstrado dificuldade com conceitos fundamentais de álgebra linear. Recomendo focar em exercícios básicos antes de avançar para tópicos mais complexos."
# """

# class ProgressAnalysisAgent:
#     def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o"):
#         self.student_profile = student_profile
#         self.execution_plan = execution_plan
#         self.session_id = session_id
#         self.user_email = user_email
#         self.disciplina = disciplina
#         self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
#         self.history = CustomMongoDBChatMessageHistory(
#             user_email=self.user_email,
#             disciplina=self.disciplina,
#             connection_string=mongo_uri,
#             session_id=self.session_id,
#             database_name=database_name,
#             collection_name="chat_history",
#             session_id_key="session_id",
#             history_key="history",
#         )
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", PROMPT_AGENTE_ANALISE_PROGRESSO),
#                 MessagesPlaceholder(variable_name="history"),
#             ]
#         )
#         self.agent = create_tool_calling_agent(self.model, [self.analyze_progress], self.prompt)
#         self.agent_executor = AgentExecutor(agent=self.agent, tools=[self.analyze_progress])

#     @tool
#     def analyze_progress(self):
#         """Analisa o progresso do estudante."""
#         return "Analisando o progresso do estudante."

#     def invoke(self, session_progress: str):
#         # Obter o histórico de mensagens
#         history_messages = self.history.messages
#         # Formatar o prompt
#         formatted_prompt = self.prompt.format(
#             historico_de_interacoes="\n".join([msg.content for msg in history_messages]),
#             session_progress=session_progress,
#             plano_de_execucao=self.execution_plan
#         )
#         # Invocar o agente
#         response = self.agent_executor.invoke({"history": history_messages})
#         return response
    


# PROMPT_AGENTE_CONVERSASIONAL = """
# Você é o Agente de Pensamento Crítico, o tutor principal responsável por ensinar o estudante, se comunicar de forma eficaz e promover o desenvolvimento do pensamento crítico.

# ### Responsabilidades:
# - **Ensino do Conteúdo**: Apresente conceitos de forma clara e adaptada ao nível do estudante.
# - **Comunicação Eficaz**: Use exemplos personalizados e linguagem apropriada ao perfil do estudante.
# - **Desenvolvimento do Pensamento Crítico**: Incentive o estudante a refletir e encontrar respostas por conta própria.

# ### Entrada:
# - **Perfil do Estudante**: {perfil_do_estudante}
# - **Plano de Execução**: {plano_de_execucao}
# - **Histórico de Interações**: {historico_de_interacoes}

# ### Tarefas:
# 1. **Responda de forma personalizada**: Use o perfil e plano do estudante para adaptar sua resposta.
# 2. **Inicie perguntas reflexivas**: Ajude o estudante a desenvolver habilidades críticas e resolver problemas fazendo perguntas e instigando o pensamento c ritico.
# 3. **Verifique o alinhamento com o plano**: Certifique-se de que sua resposta está de acordo com o plano de execução.
# 4. **Forneça exemplos práticos**: Use exemplos e analogias para facilitar a compreensão do estudante (se aplicável).
# 5. **Incentive a participação ativa**: Encoraje o estudante a se envolver na conversa e fazer perguntas.
# 6. **Analise as instrucoes recebidas do Agente Analitico para tomar acoes**: 

# **Exemplo de Interação**:
# *Entrada*: "Não entendo como resolver essa equação diferencial."
# *Resposta*: "Vamos resolver isso juntos. O que você já sabe sobre integrais? Talvez possamos começar por aí."

# **Formato de Saída**:
# - Uma resposta clara e relevante para o estudante promovendo o pensamento critico.
# """

# class ChatAgent:
#     def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o-mini"):
#         self.student_profile = student_profile
#         self.execution_plan = execution_plan
#         self.session_id = session_id
#         self.user_email = user_email
#         self.disciplina = disciplina
#         self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
#         self.history = CustomMongoDBChatMessageHistory(
#             user_email=self.user_email,
#             disciplina=self.disciplina,
#             connection_string=mongo_uri,
#             session_id=self.session_id,
#             database_name=database_name,
#             collection_name="chat_history",
#             session_id_key="session_id",
#             history_key="history",
#         )
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("ai", "{instruction}"),
#                 ("human", "{input}"),
#                 ("system", PROMPT_AGENTE_CONVERSASIONAL),
#             ]
#         )
#         self.chain = self.build_chain()

#     def build_chain(self):

#         chain = (
#             self.prompt
#             | self.model

#         )
#         return chain

#     def invoke(self, user_input: str):
#         # Obter o histórico de mensagens
#         history_messages = self.history.messages
#         # Formatar o prompt
#         formatted_prompt = self.prompt.format(
#             perfil_do_estudante=self.student_profile,
#             plano_de_execucao=self.execution_plan,
#             historico_de_interacoes="\n".join([msg.content for msg in history_messages]),
#             input=user_input
#         )
#         # Invocar o agente
#         response = self.invoke({"input": user_input, "history": history_messages})
#         # Salvar a interação no histórico
#         self.history.add_message(BaseMessage(content=user_input, role="human"))
#         self.history.add_message(AIMessage(content=response, role="assistant"))
#         return response








from typing import Any, TypedDict, List, Dict, Optional, Union
from typing_extensions import TypeVar
from langgraph.graph import END, StateGraph, START, Graph
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import MessagesState
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.utils.function_calling import convert_to_openai_function
from utils import OPENAI_API_KEY
import json
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agent.tools import DatabaseUpdateTool
from database.vector_db import QdrantHandler
from dataclasses import dataclass
import base64

@dataclass
class ExecutionStep:
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[Dict]
    atividade: Dict
    progresso: int

class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_plan: str
    user_profile: dict
    extracted_context: str
    next_step: str | None
    iteration_count: int
    chat_history: List[BaseMessage]


class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str):
        print(f"[RETRIEVAL] Initializing RetrievalTools:")
        print(f"[RETRIEVAL] - Student: {student_email}")
        print(f"[RETRIEVAL] - Disciplina: {disciplina}")
        print(f"[RETRIEVAL] - Session: {session_id}")
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.image_collection = image_collection

    async def retrieve_context(
        self,
        query: str,
        use_global: bool = True,
        use_discipline: bool = True,
        use_session: bool = True,
        specific_file_id: Optional[str] = None,
        specific_metadata: Optional[dict] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Recupera contexto ou imagem baseado na query e filtros.
        Retorna string para contexto textual ou dicionário para imagens.
        """
        print(f"\n[RETRIEVAL] Buscando contexto para query: {query}")
        
        try:
            filter_results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                use_global=use_global,
                use_discipline=use_discipline,
                use_session=use_session,
                specific_file_id=specific_file_id,
                specific_metadata=specific_metadata
            )
            
            if filter_results:
                # Verifica se o resultado é uma descrição de imagem
                for doc in filter_results:
                    if doc.metadata.get("type") == "image":
                        image_uuid = doc.metadata.get("image_uuid")
                        if image_uuid:
                            return await self.retrieve_image_and_description(image_uuid)
                
                # Se não for imagem, retorna o contexto normal
                context = "\n".join([doc.page_content for doc in filter_results])
                print(f"[RETRIEVAL] Contexto extraído: {len(context)} caracteres")
                return context
                
            print("[RETRIEVAL] Nenhum contexto relevante encontrado")
            return "Nenhum contexto relevante encontrado."
            
        except Exception as e:
            print(f"[RETRIEVAL] Erro durante a recuperação: {str(e)}")
            return "Nenhum contexto relevante encontrado."

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        """
        Recupera a imagem e sua descrição de forma assíncrona.
        """
        try:
            print(f"[RETRIEVAL] Recuperando imagem com UUID: {image_uuid}")
            image_data = await self.image_collection.find_one({"_id": image_uuid})

            if not image_data:
                print(f"[RETRIEVAL] Imagem não encontrada: {image_uuid}")
                return {"type": "error", "message": "Imagem não encontrada"}

            # Garantir que temos os bytes da imagem
            image_bytes = image_data.get("image_data")
            if not image_bytes:
                print("[RETRIEVAL] Dados da imagem ausentes")
                return {"type": "error", "message": "Dados da imagem ausentes"}

            # Se os bytes já estiverem em formato binário, use-os diretamente
            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            # Se estiver em outro formato, converta para bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                print(f"[RETRIEVAL] Formato de imagem não suportado: {type(image_bytes)}")
                return {"type": "error", "message": "Formato de imagem não suportado"}
            
            # Busca a descrição da imagem
            results = self.qdrant_handler.similarity_search_with_filter(
                query="",
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                k=1,
                use_global=False,
                use_discipline=False,
                use_session=True,
                specific_metadata={"image_uuid": image_uuid, "type": "image"}
            )

            if not results:
                return {"type": "error", "message": "Descrição da imagem não encontrada"}

            print("[RETRIEVAL] Imagem e descrição recuperadas com sucesso")
            return {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content
            }

        except Exception as e:
            print(f"[RETRIEVAL] Erro ao recuperar imagem: {e}")
            import traceback
            traceback.print_exc()
            return {"type": "error", "message": str(e)}

def create_retrieval_node(tools: RetrievalTools):
    async def retrieve_context(state: AgentState) -> AgentState:
        print("\n[NODE:RETRIEVAL] Starting retrieval node execution")
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        print(f"[NODE:RETRIEVAL] Processing message: {latest_message.content}")

        result = await tools.retrieve_context(latest_message.content)
        print(f"[NODE:RETRIEVAL] Retrieved context: {result}")

        new_state = state.copy()
        new_state["extracted_context"] = result
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        print(f"[NODE:RETRIEVAL] Updated iteration count: {new_state['iteration_count']}")
        return new_state

    return retrieve_context

def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    print("\n[PLAN] Identifying current execution step")
    sorted_steps = sorted(plano_execucao, key=lambda x: x["progresso"])
    
    current_step = next(
        (step for step in sorted_steps if step["progresso"] < 100),
        sorted_steps[-1]
    )
    
    print(f"[PLAN] Selected step: {current_step['titulo']} (Progress: {current_step['progresso']}%)")
    return ExecutionStep(**current_step)

def create_answer_plan_node():
    PLANNING_PROMPT = """Você é um assistente educacional que cria planos de resposta adaptados ao perfil do aluno e ao momento atual do plano de execução.

    Perfil do Aluno:
    {user_profile}

    Etapa Atual do Plano:
    Título: {current_step_title}
    Descrição: {current_step_description}
    Progresso: {current_step_progress}%
    
    Pergunta do Aluno:
    {question}

    Histórico da Conversa:
    {chat_history}

    Baseado no estilo de aprendizagem do aluno ({learning_style}), crie um plano de resposta que:

    1. IDENTIFICAÇÃO DO CONTEXTO:
    - Identifique exatamente em qual parte do conteúdo a pergunta se encaixa
    - Avalie se a pergunta está alinhada com o momento atual do plano
    
    2. ESTRUTURA DE RESPOSTA:
    - Adapte a explicação ao estilo de aprendizagem do aluno
    - Divida a resposta em no máximo 3 partes
    - Para cada parte, defina um objetivo mensurável
    
    3. RECURSOS E ATIVIDADES:
    - Sugira recursos baseado no perfil do aluno (priorize o perfil de aprendizagem do aluno)
    - Selecione recursos específicos do plano que se aplicam
    - Sugira exercícios práticos adaptados ao perfil
    
    4. PRÓXIMOS PASSOS:
    - Defina claramente o que o aluno deve fazer após a explicação
    - Estabeleça indicadores de compreensão

    Forneça o plano de resposta no seguinte formato JSON:
    
        "contexto_identificado": "string",
        "alinhamento_plano": boolean,
        "estrutura_resposta": [
            "parte": "string", "objetivo": "string"
        ],
        "recursos_sugeridos": ["string"],
        "atividade_pratica": "string",
        "indicadores_compreensao": ["string"],
        "proxima_acao": "string"
    """

    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate_plan(state: AgentState) -> AgentState:
        print("\n[NODE:PLANNING] Starting plan generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:PLANNING] Processing question: {latest_question}")
        
        plano_execucao = json.loads(state["current_plan"])["plano_execucao"]
        current_step = identify_current_step(plano_execucao)
        
        chat_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in state["chat_history"][-3:]
        ])
        print("[NODE:PLANNING] Formatted chat history")

        print(f"[NODE:PLANNING] Generating response with learning style: {state['user_profile']['EstiloAprendizagem']}")
        response = model.invoke(prompt.format(
            user_profile=state["user_profile"],
            current_step_title=current_step.titulo,
            current_step_description=current_step.descricao,
            current_step_progress=current_step.progresso,
            question=latest_question,
            chat_history=chat_history,
            learning_style=state["user_profile"]["EstiloAprendizagem"]
        ))

        new_state = state.copy()
        new_state["current_plan"] = response.content
        print("[NODE:PLANNING] Plan generated successfully")
        print(f"[NODE:PLANNING] Generated plan: {response.content[:200]}...")
        
        return new_state

    return generate_plan

def create_teaching_node():
    # Prompt para verificar relevância do contexto
    RELEVANCE_PROMPT = """Você é um assistente que avalia a relevância do contexto recuperado para a pergunta do usuário.

    Pergunta do usuário: {question}
    
    Contexto recuperado: {context}
    
    Avalie se o contexto é realmente útil e relevante para responder à pergunta do usuário.
    Retorne APENAS "sim" ou "não", sem explicações adicionais.
    """
    
    TEACHING_PROMPT = """Você é um tutor personalizado que ajuda os alunos a entender conceitos através do pensamento crítico.
    
    Plano de Aprendizado:
    {learning_plan}
    
    Perfil do Aluno:
    {user_profile}
    
    Descrição da Imagem:
    {context}
    
    Histórico da Conversa:
    {chat_history}
    
    Pergunta:
    {question}
    
    Baseado na descrição da imagem fornecida, elabore uma explicação clara e didática sobre o conceito apresentado.
    
    Lembre-se: 
        - O usuário é LEIGO, então tome a liderança na explicação
        - Responda SEMPRE em português do Brasil de forma clara e objetiva
        - Foque em ajudar o aluno a entender o conceito usando a imagem como referência
        - Referencie elementos específicos da imagem na sua explicação
        - Forneça exemplos práticos relacionados ao conceito mostrado
    """
    
    TEXT_PROMPT = """Você é um tutor personalizado que ajuda os alunos através do pensamento crítico.
    Em vez de fornecer respostas diretas, guie-os através do processo de resolução de problemas.
    
    Plano de Aprendizado:
    {learning_plan}
    
    Perfil do Aluno:
    {user_profile}
    
    Contexto Atual:
    {context}
    
    Histórico da Conversa:
    {chat_history}
    
    Pergunta:
    {question}
    
    Lembre-se: 
        - O usuario é LEIGO, entao tome a liderança na explicação.
        - Responda SEMPRE em português do Brasil de forma clara e objetiva.
        - Evite respostas longas e complexas.
        - Foque em respostas que ajudem o aluno a entender o conceito.
        - Forneca exemplos e exercícios práticos sempre que possível.
    """
    
    relevance_prompt = ChatPromptTemplate.from_template(RELEVANCE_PROMPT)
    image_prompt = ChatPromptTemplate.from_template(TEACHING_PROMPT)
    text_prompt = ChatPromptTemplate.from_template(TEXT_PROMPT)
    model = ChatOpenAI(model="gpt-4o", temperature=0.5)
    
    def generate_teaching_response(state: AgentState) -> AgentState:
        print("\n[NODE:TEACHING] Starting teaching response generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:TEACHING] Processing question: {latest_question}")

        chat_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in state["chat_history"][-3:]
        ])
        
        context = state["extracted_context"]
        print(f"[NODE:TEACHING] Context type: {type(context)}")
        
        # Verifica se o contexto é uma imagem
        if isinstance(context, dict) and context.get("type") == "image":
            print("[NODE:TEACHING] Processing potential image response")
            
            # Verifica relevância da imagem/descrição
            relevance_check = model.invoke(relevance_prompt.format(
                question=latest_question,
                context=context["description"]
            ))
            
            is_relevant = relevance_check.content.lower().strip() == "sim"
            print(f"[NODE:TEACHING] Image relevance check: {is_relevant}")
            
            if is_relevant:
                try:
                    # Gera a explicação baseada na descrição da imagem
                    explanation = model.invoke(image_prompt.format(
                        learning_plan=state["current_plan"],
                        user_profile=state["user_profile"],
                        context=context["description"],
                        question=latest_question,
                        chat_history=chat_history
                    ))
                    
                    # Converte os bytes da imagem para base64
                    image_bytes = context.get("image_bytes")
                    if image_bytes:
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        response_content = {
                            "type": "image",
                            "content": explanation.content,
                            "image": f"data:image/jpeg;base64,{base64_image}"
                        }
                        print("[NODE:TEACHING] Image response processed successfully")
                        response = AIMessage(content=json.dumps(response_content))
                    else:
                        print("[NODE:TEACHING] Falling back to text response due to missing image bytes")
                        response = explanation  # Usa apenas a explicação sem a imagem
                        
                except Exception as e:
                    print(f"[NODE:TEACHING] Error processing image: {str(e)}")
                    # Cai para resposta em texto em caso de erro
                    response = model.invoke(text_prompt.format(
                        learning_plan=state["current_plan"],
                        user_profile=state["user_profile"],
                        context="",  # Contexto vazio para resposta genérica
                        question=latest_question,
                        chat_history=chat_history
                    ))
            else:
                # Se a imagem não for relevante, processa como texto normal
                print("[NODE:TEACHING] Image not relevant, processing as text response")
                response = model.invoke(text_prompt.format(
                    learning_plan=state["current_plan"],
                    user_profile=state["user_profile"],
                    context="",  # Contexto vazio para resposta genérica
                    question=latest_question,
                    chat_history=chat_history
                ))
        else:
            # Processamento normal de texto
            print(f"[NODE:TEACHING] Processing text response")
            response = model.invoke(text_prompt.format(
                learning_plan=state["current_plan"],
                user_profile=state["user_profile"],
                context=context if isinstance(context, str) else "",
                question=latest_question,
                chat_history=chat_history
            ))
        
        new_state = state.copy()
        new_state["messages"] = list(state["messages"]) + [response]
        new_state["chat_history"] = list(state["chat_history"]) + [
            HumanMessage(content=latest_question),
            response
        ]
        return new_state
    
    return generate_teaching_response

def should_continue(state: AgentState) -> str:
    MAX_ITERATIONS = 1
    current_iterations = state.get("iteration_count", 0)
    
    print(f"\n[WORKFLOW] Checking continuation - Current iterations: {current_iterations}")
    if current_iterations >= MAX_ITERATIONS:
        print("[WORKFLOW] Max iterations reached, ending workflow")
        return "end"
    
    print("[WORKFLOW] Continuing to next iteration")
    return "end"

class TutorWorkflow:
    def __init__(self, qdrant_handler, student_email: str, disciplina: str, session_id: str, image_collection):
        """
        Initializes TutorWorkflow with all required parameters.
        
        Args:
            qdrant_handler: QdrantHandler instance for vector search
            student_email (str): Student's email
            disciplina (str): Discipline ID
            session_id (str): Current session ID
            image_collection: MongoDB collection for images
        """
        print(f"\n[WORKFLOW] Initializing TutorWorkflow")
        print(f"[WORKFLOW] Parameters: student_email={student_email}, disciplina={disciplina}, session_id={session_id}")
        
        self.tools = RetrievalTools(
            qdrant_handler=qdrant_handler,
            student_email=student_email,
            disciplina=disciplina,
            session_id=session_id,
            image_collection=image_collection
        )
        self.workflow = self.create_workflow()
    
    def create_workflow(self) -> Graph:
        print("[WORKFLOW] Creating workflow graph")
        retrieval_node = create_retrieval_node(self.tools)
        planning_node = create_answer_plan_node()
        teaching_node = create_teaching_node()
        
        workflow = Graph()
        
        workflow.add_node("retrieve_context", retrieval_node)
        workflow.add_node("generate_plan", planning_node)
        workflow.add_node("teach", teaching_node)
        
        workflow.add_edge("generate_plan", "retrieve_context")
        workflow.add_edge("retrieve_context", "teach")
        workflow.add_conditional_edges(
            "teach",
            should_continue,
            {
                "continue": "generate_plan",
                "end": END
            }
        )
        
        workflow.set_entry_point("generate_plan")
        print("[WORKFLOW] Workflow graph created successfully")
        return workflow.compile()
    
    async def invoke(self, query: str, student_profile: dict, current_plan=None, chat_history=None) -> dict:
        print(f"\n[WORKFLOW] Starting workflow invocation")
        print(f"[WORKFLOW] Query: {query}")
        print(f"[WORKFLOW] Student profile: {student_profile.get('EstiloAprendizagem', 'Not found')}")
        
        if chat_history is None:
            chat_history = []
        elif not isinstance(chat_history, list):
            chat_history = list(chat_history)
            
        recent_history = chat_history[-10:]
        print(f"[WORKFLOW] Using {len(recent_history)} recent chat messages")
            
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            current_plan=current_plan if current_plan else "",
            user_profile=student_profile,
            extracted_context="",
            next_step=None,
            iteration_count=0,
            chat_history=recent_history
        )

        try:
            print("[WORKFLOW] Executing workflow")
            result = await self.workflow.ainvoke(initial_state)
            print("[WORKFLOW] Workflow execution completed successfully")
            
            return {
                "messages": result["messages"],
                "final_plan": result["current_plan"],
                "chat_history": result["chat_history"]
            }
        except Exception as e:
            print(f"[WORKFLOW] Error during workflow execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Erro na execução do workflow: {str(e)}",
                "messages": [AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")],
                "chat_history": recent_history
            }
        

























#WORKING CODE
from typing import TypedDict, List, Dict, Optional
from typing_extensions import TypeVar
from langgraph.graph import END, StateGraph, START, Graph
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import MessagesState
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.utils.function_calling import convert_to_openai_function
from utils import OPENAI_API_KEY
import json
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agent.tools import DatabaseUpdateTool
from database.vector_db import QdrantHandler
from dataclasses import dataclass


@dataclass
class ExecutionStep:
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[Dict]
    atividade: Dict
    progresso: int

class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_plan: str
    user_profile: dict
    extracted_context: str
    next_step: str | None
    iteration_count: int
    chat_history: List[BaseMessage]


class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str):
        print(f"[RETRIEVAL] Initializing RetrievalTools:")
        print(f"[RETRIEVAL] - Student: {student_email}")
        print(f"[RETRIEVAL] - Disciplina: {disciplina}")
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina

    def retrieve_context(
        self,
        query: str,
        session_id: str = '1',
        use_global: bool = True,
        use_discipline: bool = True,
        use_session: bool = True,
        specific_file_id: Optional[str] = None,
        specific_metadata: Optional[dict] = None
    ) -> str:
        """
        Recupera contexto usando a estrutura correta de filtros do Qdrant.
        """
        print(f"\n[RETRIEVAL] Buscando contexto para query: {query}")
        
        try:
            filter_results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=session_id,
                disciplina_id=self.disciplina,
                use_global=use_global,
                use_discipline=use_discipline,
                use_session=use_session,
                specific_file_id=specific_file_id,
                specific_metadata=specific_metadata
            )
            
            if filter_results:
                context = "\n".join([doc.page_content for doc in filter_results])
                print(f"[RETRIEVAL] Contexto extraído: {len(context)} caracteres")
                return context
                
            print("[RETRIEVAL] Nenhum contexto relevante encontrado")
            return "Nenhum contexto relevante encontrado."
            
        except Exception as e:
            print(f"[RETRIEVAL] Erro durante a recuperação: {str(e)}")
            return "Nenhum contexto relevante encontrado."

def create_retrieval_node(tools: RetrievalTools):
    def retrieve_context(state: AgentState) -> AgentState:
        print("\n[NODE:RETRIEVAL] Starting retrieval node execution")
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        print(f"[NODE:RETRIEVAL] Processing message: {latest_message.content}")

        result = tools.retrieve_context(latest_message.content)
        print(f"[NODE:RETRIEVAL] Retrieved context: {result}")

        new_state = state.copy()
        new_state["extracted_context"] = result
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        print(f"[NODE:RETRIEVAL] Updated iteration count: {new_state['iteration_count']}")
        return new_state

    return retrieve_context

def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    print("\n[PLAN] Identifying current execution step")
    sorted_steps = sorted(plano_execucao, key=lambda x: x["progresso"])
    
    current_step = next(
        (step for step in sorted_steps if step["progresso"] < 100),
        sorted_steps[-1]
    )
    
    print(f"[PLAN] Selected step: {current_step['titulo']} (Progress: {current_step['progresso']}%)")
    return ExecutionStep(**current_step)

def create_answer_plan_node():
    PLANNING_PROMPT = """Você é um assistente educacional que cria planos de resposta adaptados ao perfil do aluno e ao momento atual do plano de execução.

    Perfil do Aluno:
    {user_profile}

    Etapa Atual do Plano:
    Título: {current_step_title}
    Descrição: {current_step_description}
    Progresso: {current_step_progress}%
    
    Pergunta do Aluno:
    {question}

    Histórico da Conversa:
    {chat_history}

    Baseado no estilo de aprendizagem do aluno ({learning_style}), crie um plano de resposta que:

    1. IDENTIFICAÇÃO DO CONTEXTO:
    - Identifique exatamente em qual parte do conteúdo a pergunta se encaixa
    - Avalie se a pergunta está alinhada com o momento atual do plano
    
    2. ESTRUTURA DE RESPOSTA:
    - Adapte a explicação ao estilo de aprendizagem do aluno
    - Divida a resposta em no máximo 3 partes
    - Para cada parte, defina um objetivo mensurável
    
    3. RECURSOS E ATIVIDADES:
    - Sugira recursos baseado no perfil do aluno (priorize o perfil de aprendizagem do aluno)
    - Selecione recursos específicos do plano que se aplicam
    - Sugira exercícios práticos adaptados ao perfil
    
    4. PRÓXIMOS PASSOS:
    - Defina claramente o que o aluno deve fazer após a explicação
    - Estabeleça indicadores de compreensão

    Forneça o plano de resposta no seguinte formato JSON:
    
        "contexto_identificado": "string",
        "alinhamento_plano": boolean,
        "estrutura_resposta": [
            "parte": "string", "objetivo": "string"
        ],
        "recursos_sugeridos": ["string"],
        "atividade_pratica": "string",
        "indicadores_compreensao": ["string"],
        "proxima_acao": "string"
    """

    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate_plan(state: AgentState) -> AgentState:
        print("\n[NODE:PLANNING] Starting plan generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:PLANNING] Processing question: {latest_question}")
        
        plano_execucao = json.loads(state["current_plan"])["plano_execucao"]
        current_step = identify_current_step(plano_execucao)
        
        chat_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in state["chat_history"][-3:]
        ])
        print("[NODE:PLANNING] Formatted chat history")

        print(f"[NODE:PLANNING] Generating response with learning style: {state['user_profile']['EstiloAprendizagem']}")
        response = model.invoke(prompt.format(
            user_profile=state["user_profile"],
            current_step_title=current_step.titulo,
            current_step_description=current_step.descricao,
            current_step_progress=current_step.progresso,
            question=latest_question,
            chat_history=chat_history,
            learning_style=state["user_profile"]["EstiloAprendizagem"]
        ))

        new_state = state.copy()
        new_state["current_plan"] = response.content
        print("[NODE:PLANNING] Plan generated successfully")
        print(f"[NODE:PLANNING] Generated plan: {response.content[:200]}...")
        
        return new_state

    return generate_plan

def create_teaching_node():
    TEACHING_PROMPT = """Você é um tutor personalizado que ajuda os alunos através do pensamento crítico.
    Em vez de fornecer respostas diretas, guie-os através do processo de resolução de problemas.
    
    Plano de Aprendizado:
    {learning_plan}
    
    Perfil do Aluno:
    {user_profile}
    
    Contexto Atual:
    {context}
    
    Histórico da Conversa:
    {chat_history}
    
    Pergunta:
    {question}
    
    Lembre-se: 
        - O usuario é LEIGO, entao tome a liderança na explicação.
        - Responda SEMPRE em português do Brasil de forma clara e objetiva.
        - evite respostas longas e complexas.
        - Foque em respostas que ajudem o aluno a entender o conceito.
        - Forneca exemlos e exercícios práticos sempre que possível.
    
    Sua resposta:"""
    
    prompt = ChatPromptTemplate.from_template(TEACHING_PROMPT)
    model = ChatOpenAI(model="gpt-4o", temperature=0.5)
    
    def generate_teaching_response(state: AgentState) -> AgentState:
        print("\n[NODE:TEACHING] Starting teaching response generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:TEACHING] Processing question: {latest_question}")

        chat_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in state["chat_history"][-3:]
        ])
        
        print(f"[NODE:TEACHING] Context length: {len(state['extracted_context'])}")
        print(f"[NODE:TEACHING] Context preview: {state['extracted_context'][:200]}...")
        response = model.invoke(prompt.format(
            learning_plan=state["current_plan"],
            user_profile=state["user_profile"],
            context=state["extracted_context"],
            question=latest_question,
            chat_history=chat_history
        ))
        
        print("[NODE:TEACHING] Generated teaching response")
        print(f"[NODE:TEACHING] Response preview: {response.content[:200]}...")
        
        new_state = state.copy()
        new_state["messages"] = list(state["messages"]) + [AIMessage(content=response.content)]
        new_state["chat_history"] = list(state["chat_history"]) + [
            HumanMessage(content=latest_question),
            AIMessage(content=response.content)
        ]
        return new_state
    
    return generate_teaching_response

def should_continue(state: AgentState) -> str:
    MAX_ITERATIONS = 1
    current_iterations = state.get("iteration_count", 0)
    
    print(f"\n[WORKFLOW] Checking continuation - Current iterations: {current_iterations}")
    if current_iterations >= MAX_ITERATIONS:
        print("[WORKFLOW] Max iterations reached, ending workflow")
        return "end"
    
    print("[WORKFLOW] Continuing to next iteration")
    return "end"

class TutorWorkflow:
    def __init__(self, qdrant_handler, student_email, disciplina):
        print(f"\n[WORKFLOW] Initializing TutorWorkflow")
        print(f"[WORKFLOW] Parameters: student_email={student_email}, disciplina={disciplina}")
        self.tools = RetrievalTools(qdrant_handler, student_email, disciplina)
        self.workflow = self.create_workflow()
    
    def create_workflow(self) -> Graph:
        print("[WORKFLOW] Creating workflow graph")
        retrieval_node = create_retrieval_node(self.tools)
        planning_node = create_answer_plan_node()
        teaching_node = create_teaching_node()
        
        workflow = Graph()
        
        workflow.add_node("retrieve_context", retrieval_node)
        workflow.add_node("generate_plan", planning_node)
        workflow.add_node("teach", teaching_node)
        
        workflow.add_edge("generate_plan", "retrieve_context")
        workflow.add_edge("retrieve_context", "teach")
        workflow.add_conditional_edges(
            "teach",
            should_continue,
            {
                "continue": "generate_plan",
                "end": END
            }
        )
        
        workflow.set_entry_point("generate_plan")
        print("[WORKFLOW] Workflow graph created successfully")
        return workflow.compile()
    
    async def invoke(self, query: str, student_profile: dict, current_plan=None, chat_history=None) -> dict:
        print(f"\n[WORKFLOW] Starting workflow invocation")
        print(f"[WORKFLOW] Query: {query}")
        print(f"[WORKFLOW] Student profile: {student_profile.get('EstiloAprendizagem', 'Not found')}")
        
        if chat_history is None:
            chat_history = []
        elif not isinstance(chat_history, list):
            chat_history = list(chat_history)
            
        recent_history = chat_history[-10:]
        print(f"[WORKFLOW] Using {len(recent_history)} recent chat messages")
            
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            current_plan=current_plan if current_plan else "",
            user_profile=student_profile,
            extracted_context="",
            next_step=None,
            iteration_count=0,
            chat_history=recent_history
        )

        try:
            print("[WORKFLOW] Executing workflow")
            result = await self.workflow.ainvoke(initial_state)
            print("[WORKFLOW] Workflow execution completed successfully")
            
            return {
                "messages": result["messages"],
                "final_plan": result["current_plan"],
                "chat_history": result["chat_history"]
            }
        except Exception as e:
            print(f"[WORKFLOW] Error during workflow execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Erro na execução do workflow: {str(e)}",
                "messages": [AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")],
                "chat_history": recent_history
            }