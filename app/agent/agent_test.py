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
from database.vector_db import QdrantHandler, TextSplitter
from dataclasses import dataclass
import base64
import asyncio
from typing import Tuple, Dict, Any
from youtubesearchpython import VideosSearch
import wikipediaapi
from database.mongo_database_manager import MongoDatabaseManager
class UserProfile(BaseModel):
    Nome: str
    Email: str
    EstiloAprendizagem: Dict[str, str]
    Feedback: Optional[Dict[str, Any]] = None
    PreferenciaAprendizado: Optional[Dict[str, Any]] = None
    
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
    user_profile: Dict[str, Any]
    extracted_context: Dict[str, Any]
    next_step: str | None
    iteration_count: int
    chat_history: List[BaseMessage]
    needs_retrieval: bool
    evaluation_reason: str
    web_search_results: Dict[str, str]
    answer_type: str | None

# class StudyPlanUpdater(MongoDatabaseManager):
#     def __init__(self, db_name: str,state: AgentState, collection_name: str = "study_plans"):
#         super().__init__(db_name, collection_name)
#         self.state = state

#     def update_agent_state(self, state: AgentState):
#         self.state = state
#         self.update_one({"_id": "current_state"}, {"$set": state}, upsert=True)

#     def get_agent_state(self) -> AgentState:
#         return self.find_one({"_id": "current_state"}) or {}

# class StudentProfileUpdater(MongoDatabaseManager):
#     def __init__(self, db_name: str, state: AgentState, collection_name: str = "student_learn_preference"):
#         super().__init__(db_name, collection_name)
#         self.state = state

#     def update_agent_state(self, state: AgentState):
#         self.state = state
#         self.update_one({"_id": "current_state"}, {"$set": state}, upsert=True)

#     def get_agent_state(self) -> AgentState:
#         return self.find_one({"_id": "current_state"}) or {}

def filter_chat_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Filtra o histórico do chat para remover conteúdo de imagem e manter apenas texto.
    """
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage):
            try:
                # Verifica se é uma mensagem multimodal (JSON)
                content = json.loads(msg.content)
                if isinstance(content, dict):
                    if content.get("type") == "multimodal":
                        # Remove o campo 'image' e mantém apenas o texto
                        filtered_content = {
                            "type": "multimodal",
                            "content": content["content"]
                        }
                        filtered_messages.append(AIMessage(content=filtered_content["content"]))
                    else:
                        # Mensagem JSON sem imagem
                        filtered_messages.append(msg)
                else:
                    # Não é um objeto JSON válido
                    filtered_messages.append(msg)
            except json.JSONDecodeError:
                # Mensagem regular sem JSON
                filtered_messages.append(msg)
    return filtered_messages

def format_chat_history(messages: List[BaseMessage], max_messages: int = 3) -> str:
    """
    Formata o histórico do chat filtrado para uso em prompts.
    """
    # Primeiro filtra o histórico
    filtered_messages = filter_chat_history(messages[-max_messages:])

    # Então formata as mensagens filtradas
    formatted_history = []
    for msg in filtered_messages:
        role = 'Aluno' if isinstance(msg, HumanMessage) else 'Tutor'
        content = msg.content
        if isinstance(content, str):
            formatted_history.append(f"{role}: {content}")

    return "\n".join(formatted_history)

def create_question_evaluation_node():
    EVALUATION_PROMPT = """Você é um assistente especializado em avaliar se uma pergunta precisa de contexto adicional para ser respondida adequadamente.

    Histórico da Conversa:
    {chat_history}

    Pergunta Atual:
    {question}

    Analise se a pergunta:
    1. Requer informações específicas do material de estudo
    2. Pode ser respondida apenas com conhecimento geral
    3. É uma continuação direta do contexto já fornecido no histórico
    4. É uma pergunta de esclarecimento sobre algo já discutido

    Retorne apenas um JSON no formato:

        "needs_retrieval": boolean,
        "reason": "string"

    """

    prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def evaluate_question(state: AgentState) -> AgentState:
        print("\n[NODE:EVALUATION] Starting question evaluation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:EVALUATION] Evaluating question: {latest_question}")

        chat_history = format_chat_history(state["chat_history"])

        response = model.invoke(prompt.format(
            chat_history=chat_history,
            question=latest_question
        ))

        try:
            evaluation = json.loads(response.content)
            print(f"[NODE:EVALUATION] Evaluation result: {evaluation}")

            new_state = state.copy()
            new_state["needs_retrieval"] = evaluation["needs_retrieval"]
            new_state["evaluation_reason"] = evaluation["reason"]
            return new_state

        except json.JSONDecodeError as e:
            print(f"[NODE:EVALUATION] Error parsing evaluation: {e}")
            # Default to performing retrieval in case of error
            new_state = state.copy()
            new_state["needs_retrieval"] = True
            new_state["evaluation_reason"] = "Error in evaluation, defaulting to retrieval"
            return new_state

    return evaluate_question

class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
        print(f"[RETRIEVAL] Initializing RetrievalTools:")
        print(f"[RETRIEVAL] - Student: {student_email}")
        print(f"[RETRIEVAL] - Disciplina: {disciplina}")
        print(f"[RETRIEVAL] - Session: {session_id}")
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.image_collection = image_collection
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.state = state

        self.QUESTION_TRANSFORM_PROMPT = """
        Você é um especialista em transformar perguntas para melhorar a recuperação de contexto.

        Histórico da conversa: {chat_history}

        Pergunta original: {question}

        O usuário pode fazer perguntas que remetam a perguntas anteriores, então é importante analisar o histórico da conversa.

        Retorne apenas a pergunta reescrita, sem explicações adicionais.
        """

        self.RELEVANCE_ANALYSIS_PROMPT = """
        Analise a relevância dos contextos recuperados para a pergunta do usuário.

        Pergunta: {question}

        Contextos recuperados:
        Texto: {text_context}
        Imagem: {image_context}
        Tabela: {table_context}

        Para cada contexto, avalie a relevância em uma escala de 0 a 1 e explique brevemente por quê.
        Retorne um JSON no formato:
        {{
            "text": {{"score": 0.0, "reason": "string"}},
            "image": {{"score": 0.0, "reason": "string"}},
            "table": {{"score": 0.0, "reason": "string"}},
            "recommended_context": "text|image|table|combined"
        }}

        Mantenha o formato JSON exato e use apenas aspas duplas.
        """

    async def transform_question(self, question: str) -> str:
        """
        Transforma a pergunta para melhor recuperação de contexto baseado no tipo de busca.
        """
        # Usa a nova função de formatação do histórico
        formatted_history = format_chat_history(self.state["chat_history"], max_messages=4)

        print(f"[RETRIEVAL] Using chat history for question transformation:")
        print(formatted_history)

        prompt = ChatPromptTemplate.from_template(self.QUESTION_TRANSFORM_PROMPT)
        response = await self.model.ainvoke(prompt.format(
            chat_history=formatted_history,
            question=question,
        ))

        transformed_question = response.content.strip()
        print(f"[RETRIEVAL] Transformed question: {transformed_question}")
        return transformed_question

    async def parallel_context_retrieval(self, question: str) -> Dict[str, Any]:
        print(f"\n[RETRIEVAL] Starting parallel context retrieval for: {question}")

        text_question, image_question, table_question = await asyncio.gather(
            self.transform_question(question),
            self.transform_question(question),
            self.transform_question(question)
        )

        text_context, image_context, table_context = await asyncio.gather(
            self.retrieve_text_context(text_question),
            self.retrieve_image_context(image_question),
            self.retrieve_table_context(table_question)
        )

        relevance_analysis = await self.analyze_context_relevance(
            original_question=question,
            text_context=text_context,
            image_context=image_context,
            table_context=table_context
        )

        return {
            "text": text_context,
            "image": image_context,
            "table": table_context,
            "relevance_analysis": relevance_analysis
        }

    async def retrieve_text_context(self, query: str) -> str:
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "text"}
            )
            return "\n".join([doc.page_content for doc in results]) if results else ""
        except Exception as e:
            print(f"[RETRIEVAL] Error in text retrieval: {e}")
            return ""

    async def retrieve_image_context(self, query: str) -> Dict[str, Any]:
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "image"}
            )
            print("")
            print("--------------------------------------------------")
            print(f"[RETRIEVAL] Image search results: {results}")
            print("--------------------------------------------------")
            print("")
            if not results:
                return {"type": "image", "content": None, "description": ""}

            image_uuid = results[0].metadata.get("image_uuid")
            if not image_uuid:
                return {"type": "image", "content": None, "description": ""}

            return await self.retrieve_image_and_description(image_uuid)
        except Exception as e:
            print(f"[RETRIEVAL] Error in image retrieval: {e}")
            return {"type": "image", "content": None, "description": ""}

    async def retrieve_table_context(self, query: str) -> Dict[str, Any]:
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "table"}
            )

            if not results:
                return {"type": "table", "content": None}

            return {
                "type": "table",
                "content": results[0].page_content,
                "metadata": results[0].metadata
            }
        except Exception as e:
            print(f"[RETRIEVAL] Error in table retrieval: {e}")
            return {"type": "table", "content": None}

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

            image_bytes = image_data.get("image_data")
            if not image_bytes:
                print("[RETRIEVAL] Dados da imagem ausentes")
                return {"type": "error", "message": "Dados da imagem ausentes"}

            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                print(f"[RETRIEVAL] Formato de imagem não suportado: {type(image_bytes)}")
                return {"type": "error", "message": "Formato de imagem não suportado"}

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
            print(f"[RETRIEVAL] Resultados da busca de descrição: {results}")
            if not results:
                return {"type": "error", "message": "Descrição da imagem não encontrada"}
            print("[RETRIEVAL] Imagem e descrição recuperadas com sucesso")
            print(f"[RETRIEVAL] Descrição da imagem: {results[0].page_content}")
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

    async def analyze_context_relevance(
        self,
        original_question: str,
        text_context: str,
        image_context: Dict[str, Any],
        table_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if not any([text_context, image_context, table_context]):
                print("[RETRIEVAL] No context available for relevance analysis")
                return self._get_default_analysis()

            prompt = ChatPromptTemplate.from_template(self.RELEVANCE_ANALYSIS_PROMPT)

            image_description = ""
            if image_context and isinstance(image_context, dict):
                image_description = image_context.get("description", "")
            print(f"[RETRIEVAL] Image description: {image_description}")
            print(f"[RETRIEVAL] Image context: {image_context}")

            # Tratamento seguro para contexto de tabela
            table_content = ""
            if table_context and isinstance(table_context, dict):
                table_content = table_context.get("content", "")

            # Garantir que todos os contextos são strings antes de aplicar slice
            text_preview = str(text_context)[:500] + "..." if text_context and len(str(text_context)) > 500 else str(text_context or "")
            image_preview = str(image_description)[:500] + "..." if len(str(image_description)) > 500 else str(image_description)
            table_preview = str(table_content)[:500] + "..." if len(str(table_content)) > 500 else str(table_content)

            response = await self.model.ainvoke(prompt.format(
                question=original_question,
                text_context=text_preview,
                image_context=image_preview,
                table_context=table_preview
            ))

            try:
                cleaned_content = response.content.strip()
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()

                analysis = json.loads(cleaned_content)
                print(f"[RETRIEVAL] Relevance analysis: {analysis}")

                required_fields = ["text", "image", "table", "recommended_context"]
                if not all(field in analysis for field in required_fields):
                    raise ValueError("Missing required fields in analysis")

                return analysis

            except json.JSONDecodeError as e:
                print(f"[RETRIEVAL] Error parsing relevance analysis: {e}")
                print(f"[RETRIEVAL] Invalid JSON content: {cleaned_content}")
                return self._get_default_analysis()
            except ValueError as e:
                print(f"[RETRIEVAL] Validation error: {e}")
                return self._get_default_analysis()

        except Exception as e:
            print(f"[RETRIEVAL] Error in analyze_context_relevance: {e}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "text": {"score": 0, "reason": "Default fallback due to analysis error"},
            "image": {"score": 0, "reason": "Default fallback due to analysis error"},
            "table": {"score": 0, "reason": "Default fallback due to analysis error"},
            "recommended_context": "combined"
        }

def create_retrieval_node(tools: RetrievalTools):
    async def retrieve_context(state: AgentState) -> AgentState:
        print("\n[NODE:RETRIEVAL] Starting retrieval node execution")
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        print(f"[NODE:RETRIEVAL] Processing message: {latest_message.content}")

        tools.state = state
        context_results = await tools.parallel_context_retrieval(latest_message.content)

        new_state = state.copy()
        new_state["extracted_context"] = context_results
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        print(f"[NODE:RETRIEVAL] Updated iteration count: {new_state['iteration_count']}")
        return new_state

    return retrieve_context

def route_after_plan_generation():
    ROUTE_PROMPT = """Você é um assistente educacional que avalia o plano de resposta gerado e decide o próximo passo baseado na pergunta do aluno.

    Pergunta do Aluno:
    {question}

    Histórico da Conversa:
    {chat_history}

    Plano de Resposta Gerado:
    {plan}

    No Banco de dados (retrieval), ha varios materiais de estudo, como textos, imagens e tabelas, que podem ser relevantes para a pergunta do aluno.

    Analise a pergunta e determine o próximo passo:
    1. Se a pergunta pede explicitamente por informações da web (Wikipedia, YouTube, etc), use "websearch"
    2. Se a pergunta requer contexto do material de estudo ou exemplos, use "retrieval"
    3. Se a pergunta pode ser respondida diretamente, use "direct_answer"

    Retorne apenas um JSON com o seguinte formato, mantendo as aspas duplas:
        "analysis": "string explicando a análise",
        "next_step": "websearch|retrieval|direct_answer"
    """

    prompt = ChatPromptTemplate.from_template(ROUTE_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def route_after_plan(state: AgentState) -> AgentState:
        print("\n[ROUTING] Starting route after plan generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[ROUTING] Processing question: {latest_question}")

        chat_history = format_chat_history(state["chat_history"])
        current_plan = state.get("current_plan", "{}")

        try:
            response = model.invoke(prompt.format(
                question=latest_question,
                chat_history=chat_history,
                plan=current_plan
            ))

            # Processar a resposta
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            route = json.loads(cleaned_content)
            print(f"[ROUTING] Route analysis: {route}")

            if route.get("next_step") not in ["websearch", "retrieval", "direct_answer"]:
                raise ValueError(f"Invalid next_step value: {route.get('next_step')}")

            new_state = state.copy()
            new_state["next_step"] = route["next_step"]
            return new_state

        except Exception as e:
            print(f"[ROUTING] Error in routing: {str(e)}")
            # Em caso de erro, usamos uma lógica simples baseada em palavras-chave
            question_lower = latest_question.lower()
            if any(keyword in question_lower for keyword in ["youtube", "video", "wikipedia", "web"]):
                next_step = "websearch"
            else:
                next_step = "retrieval"

            print(f"[ROUTING] Fallback routing decision: {next_step}")
            new_state = state.copy()
            new_state["next_step"] = next_step
            return new_state

    return route_after_plan

def create_answer_plan_node():
    PLANNING_PROMPT = """Você é um assistente educacional que cria planos de resposta adaptados ao perfil do aluno e ao momento atual do plano de execução.

    Perfil do Aluno:
    Nome: {nome}
    Estilo de Aprendizagem:
    - Percepção: {percepcao}
    - Entrada: {entrada}
    - Processamento: {processamento}
    - Entendimento: {entendimento}

    Etapa Atual do Plano:
    Título: {titulo}
    Descrição: {descricao}
    Progresso: {progresso}%

    Pergunta do Aluno:
    {question}

    Histórico da Conversa:
    {chat_history}

    Baseado no estilo de aprendizagem do aluno, crie um plano de resposta que:

    1. IDENTIFICAÇÃO DO CONTEXTO:
    - Identifique exatamente em qual parte do conteúdo a pergunta se encaixa
    - Avalie se a pergunta está alinhada com o momento atual do plano

    2. ESTRUTURA DE RESPOSTA:
    - Adapte a explicação ao estilo de aprendizagem do aluno
    - Divida a resposta em no máximo 3 partes
    - Para cada parte, defina um objetivo mensurável

    3. RECURSOS E ATIVIDADES (OPCIONAL):
    - Sugira recursos baseado no perfil do aluno
    - Selecione recursos específicos do plano que se aplicam
    - Sugira exercícios práticos adaptados ao perfil

    4. PRÓXIMOS PASSOS:
    - Defina claramente o que o aluno deve fazer após a explicação
    - Estabeleça indicadores de compreensão
    - Estabeleca o nivel_resposta esperado, divida entre básico, intermediário e avançado

    Forneça o plano de resposta no seguinte formato JSON:
        "contexto_identificado": "string",
        "alinhamento_plano": boolean,
        "estrutura_resposta": [
            "parte": "string", "objetivo": "string"
        ],
        "recursos_sugeridos": ["string"],
        "atividade_pratica": "string",
        "indicadores_compreensao": ["string"],
        "nivel_resposta": "string",
        "proxima_acao": "string"
    """

    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate_plan(state: AgentState) -> AgentState:
        print("\n[NODE:PLANNING] Starting plan generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:PLANNING] Processing question: {latest_question}")

        try:
            # Processar o plano de execução
            plano_execucao = json.loads(state["current_plan"])
            current_step = identify_current_step(plano_execucao["plano_execucao"])
            print(f"[PLANNING] Current step: {current_step.titulo} ({current_step.progresso}%)")

            # Extrair perfil do usuário
            user_profile = state["user_profile"]
            estilos = user_profile.get("EstiloAprendizagem", {})

            # Formatar histórico do chat
            chat_history = format_chat_history(state["chat_history"])

            # Preparar parâmetros do prompt
            params = {
                "nome": user_profile.get("Nome", ""),
                "percepcao": estilos.get("Percepcao", ""),
                "entrada": estilos.get("Entrada", ""),
                "processamento": estilos.get("Processamento", ""),
                "entendimento": estilos.get("Entendimento", ""),
                "titulo": current_step.titulo,
                "descricao": current_step.descricao,
                "progresso": current_step.progresso,
                "question": latest_question,
                "chat_history": chat_history
            }

            # Gerar o plano
            response = model.invoke(prompt.format(**params))

            # Processar a resposta
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            plan = json.loads(cleaned_content)
            print(f"[PLANNING] Generated plan: {plan}")

            # Atualizar o estado
            new_state = state.copy()
            new_state["current_plan"] = json.dumps(plan)
            new_state["next_step"] = plan["proxima_acao"]
            return new_state

        except Exception as e:
            print(f"[PLANNING] Error in plan generation: {str(e)}")
            import traceback
            traceback.print_exc()

            default_plan = {
                "contexto_identificado": "Erro na geração do plano",
                "alinhamento_plano": True,
                "estrutura_resposta": [
                    {"parte": "Resposta básica", "objetivo": "Fornecer informação solicitada"}
                ],
                "recursos_sugeridos": [],
                "atividade_pratica": "N/A",
                "indicadores_compreensao": ["Compreensão básica do conceito"],
                "nivel_resposta": "básico",
                "proxima_acao": "retrieval"
            }

            new_state = state.copy()
            new_state["current_plan"] = json.dumps(default_plan)
            new_state["next_step"] = "retrieval"
            return new_state

    return generate_plan

def create_teaching_node():
    CONTEXT_TEACHING_PROMPT = """Você é um tutor personalizado que ajuda os alunos através do pensamento crítico.

    Plano de Aprendizado:
    {learning_plan}

    Perfil do Aluno:
    {user_profile}

    Fonte de Informação:
    {source_type}

    Contexto:
    {context}

    Histórico da Conversa:
    {chat_history}

    Pergunta:
    {question}

    Baseado nas informações disponíveis, crie uma explicação que:
    1. Se for resultado de busca web:
        - Apresente os links e recursos encontrados
        - Explique por que esses recursos são relevantes
        - Sugira como o aluno pode aproveitar melhor os materiais
    2. Se for contexto do material:
        - Primeiro explique o contexto do material
        - Integre os diferentes tipos de contexto de forma coerente
        - Priorize o tipo de contexto mais relevante
        - Adapte a explicação ao estilo de aprendizagem do aluno
    3. Se for resposta direta:
        - Foque no plano de resposta gerado
        - Use uma linguagem clara e objetiva
        - Estruture a explicação conforme o perfil do aluno

    Lembre-se:
    - Responda SEMPRE em português do Brasil de forma clara e objetiva
    - Evite respostas longas e complexas
    - Referencie elementos específicos dos contextos utilizados
    - Inclua os links quando disponíveis
    ATENÇÃO: VOCÊ RESPONDE DIRETAMENTE AO ALUNO, NÃO AO SISTEMA.
    """

    context_prompt = ChatPromptTemplate.from_template(CONTEXT_TEACHING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    def generate_teaching_response(state: AgentState) -> AgentState:
        print("\n[NODE:TEACHING] Starting teaching response generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:TEACHING] Processing question: {latest_question}")

        chat_history = format_chat_history(state["chat_history"])

        try:
            # Determinar tipo de resposta e contexto
            if state.get("next_step") == "direct_answer":
                print("[NODE:TEACHING] Processing direct answer")
                source_type = "Resposta direta"
                context = f"Plano de resposta: {state['current_plan']}"

            elif state.get("web_search_results"):
                print("[NODE:TEACHING] Processing web search results")
                source_type = "Resultados de busca web"
                web_results = state["web_search_results"]
                context = (
                    f"Wikipedia:\n{web_results.get('wikipedia', 'Não disponível')}\n\n"
                    f"YouTube:\n{web_results.get('youtube', 'Não disponível')}"
                )

            else:
                print("[NODE:TEACHING] Processing study material context")
                contexts = state["extracted_context"]
                source_type = "Material de estudo"
                context = (
                    f"Texto: {contexts.get('text', '')}\n"
                    f"Imagem: {contexts.get('image', {}).get('description', '')}\n"
                    f"Tabela: {contexts.get('table', {}).get('content', '')}"
                )

            explanation = model.invoke(context_prompt.format(
                learning_plan=state["current_plan"],
                user_profile=state["user_profile"],
                source_type=source_type,
                context=context[:1000] + "..." if len(context) > 1000 else context,
                question=latest_question,
                chat_history=chat_history
            ))

            # Check if we have an image to include
            image_content = None
            if state.get("extracted_context"):  # Only check for images in study material context
                image_context = state["extracted_context"].get("image", {})
                relevance = state["extracted_context"].get("relevance_analysis", {})
                if (image_context.get("type") == "image" and 
                    image_context.get("image_bytes") and 
                    relevance.get("image", {}).get("score", 0) > 0.3):
                    image_content = image_context.get("image_bytes")

            # Format response
            if image_content:
                base64_image = base64.b64encode(image_content).decode('utf-8')
                response_content = {
                    "type": "multimodal",
                    "content": explanation.content,
                    "image": f"data:image/jpeg;base64,{base64_image}"
                }
                response = AIMessage(content=json.dumps(response_content))
                history_message = AIMessage(content=explanation.content)
            else:
                response = explanation
                history_message = explanation

            # Update state
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                history_message
            ]
            return new_state

        except Exception as e:
            print(f"[NODE:TEACHING] Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
            error_message = "Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente."
            response = AIMessage(content=error_message)
            history_message = response

            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                history_message
            ]
            return new_state

    return generate_teaching_response

def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    print("\n[PLAN] Identifying current execution step")
    sorted_steps = sorted(plano_execucao, key=lambda x: x["progresso"])
    print(f"[PLAN] Sorted steps: {sorted_steps}")

    current_step = next(
        (step for step in sorted_steps if step["progresso"] < 100),
        sorted_steps[-1]
    )

    print(f"[PLAN] Selected step: {current_step['titulo']} (Progress: {current_step['progresso']}%)")
    return ExecutionStep(**current_step)

def should_continue(state: AgentState) -> str:
    MAX_ITERATIONS = 1
    current_iterations = state.get("iteration_count", 0)

    print(f"\n[WORKFLOW] Checking continuation - Current iterations: {current_iterations}")
    if current_iterations >= MAX_ITERATIONS:
        print("[WORKFLOW] Max iterations reached, ending workflow")
        return "end"

    print("[WORKFLOW] Continuing to next iteration")
    return "continue"

def route_after_evaluation(state: AgentState) -> str:
    needs_retrieval = state.get("needs_retrieval", True)
    print(f"\n[WORKFLOW] Routing after evaluation - Needs retrieval: {needs_retrieval}")
    return "retrieve_context" if needs_retrieval else "direct_answer"


class WebSearchTools:
    def __init__(self):
        print("[WEBSEARCH] Initializing WebSearchTools")

    def search_youtube(self, query: str) -> str:
        """
        Realiza uma pesquisa no YouTube e retorna o link do vídeo mais relevante.
        """
        try:
            print(f"[WEBSEARCH] Searching YouTube for: {query}")
            videos_search = VideosSearch(query, limit=3)  # Aumentamos para 3 resultados
            results = videos_search.result()

            if results['result']:
                videos_info = []
                for video in results['result'][:3]:  # Pegamos os 3 primeiros resultados
                    video_info = {
                        'title': video['title'],
                        'link': video['link'],
                        'channel': video.get('channel', {}).get('name', 'N/A'),
                        'duration': video.get('duration', 'N/A'),
                        'description': video.get('descriptionSnippet', [{'text': 'Sem descrição'}])[0]['text']
                    }
                    videos_info.append(video_info)
                
                # Formata a resposta com múltiplos vídeos
                response = "Vídeos encontrados:\n\n"
                for i, video in enumerate(videos_info, 1):
                    response += (
                        f"{i}. Título: {video['title']}\n"
                        f"   Link: {video['link']}\n"
                        f"   Canal: {video['channel']}\n"
                        f"   Duração: {video['duration']}\n"
                        f"   Descrição: {video['description']}\n\n"
                    )

                print(f"[WEBSEARCH] Found {len(videos_info)} videos")
                return response
            else:
                print("[WEBSEARCH] No videos found")
                return "Nenhum vídeo encontrado."
        except Exception as e:
            print(f"[WEBSEARCH] Error searching YouTube: {str(e)}")
            return "Ocorreu um erro ao buscar no YouTube."

    def search_wikipedia(self, query: str) -> str:
        """
        Realiza uma pesquisa no Wikipedia e retorna o resumo da página.
        """
        try:
            print(f"[WEBSEARCH] Searching Wikipedia for: {query}")
            wiki_wiki = wikipediaapi.Wikipedia(
                'TutorBot/1.0 (pericles.junior@cesar.school)',
                'pt'
            )
            page = wiki_wiki.page(query)

            if page.exists():
                summary = (
                    f"Título: {page.title}\n"
                    f"Resumo: {page.summary[:500]}...\n"
                    f"Link: {page.fullurl}"
                )
                print(f"[WEBSEARCH] Found Wikipedia article: {page.title}")
                return summary
            else:
                print("[WEBSEARCH] No Wikipedia article found")
                return "Página não encontrada na Wikipedia."
        except Exception as e:
            print(f"[WEBSEARCH] Error searching Wikipedia: {str(e)}")
            return "Ocorreu um erro ao buscar na Wikipedia."

def route_after_planning(state: AgentState) -> str:
    """
    Determina o próximo nó após o planejamento com base no plano gerado.
    """
    next_step = state.get("next_step", "retrieval")
    print(f"[ROUTING] Routing after planning: {next_step}")

    if next_step == "websearch":
        return "web_search"
    elif next_step == "retrieval":
        return "retrieve_context"
    else:
        return "direct_answer"

def create_websearch_node(web_tools: WebSearchTools):
    QUERY_OPTIMIZATION_PROMPT = """Você é um especialista em otimizar buscas no YouTube.

    Pergunta original do aluno: {question}
    Histórico da conversa: {chat_history}

    Seu objetivo é criar uma query otimizada para o YouTube que:
    1. Identifique os conceitos principais da pergunta
    2. Adicione termos relevantes para melhorar os resultados (como "tutorial", "explicação", "aula")
    3. Inclua termos técnicos apropriados
    4. Use uma linguagem natural e efetiva para buscas
    5. Mantenha o foco educacional

    Retorne apenas a query otimizada, sem explicações adicionais.
    """

    WEBSEARCH_PROMPT = """
    Você é um assistente especializado em integrar informações da web com o contexto educacional.

    A pergunta original do aluno foi: {question}
    A query otimizada usada foi: {optimized_query}

    Encontrei os seguintes recursos:

    {resources}

    Crie uma resposta educativa que:
    1. Apresente os recursos encontrados de forma organizada
    2. Destaque por que eles são relevantes para a pergunta
    3. Sugira como o aluno pode aproveitar melhor o conteúdo
    4. Inclua os links diretos para facilitar o acesso

    Mantenha os links originais na resposta.
    """

    query_prompt = ChatPromptTemplate.from_template(QUERY_OPTIMIZATION_PROMPT)
    response_prompt = ChatPromptTemplate.from_template(WEBSEARCH_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def web_search(state: AgentState) -> AgentState:
        print("\n[NODE:WEBSEARCH] Starting web search")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])

        try:
            # Otimizar a query
            print(f"[WEBSEARCH] Optimizing query: {latest_question}")
            optimized_query = model.invoke(query_prompt.format(
                question=latest_question,
                chat_history=chat_history
            )).content.strip()
            print(f"[WEBSEARCH] Optimized query: {optimized_query}")

            # Realizar buscas com a query otimizada
            wiki_result = web_tools.search_wikipedia(optimized_query)
            youtube_result = web_tools.search_youtube(optimized_query)

            # Formatar recursos para o prompt
            resources = (
                "YouTube:\n"
                f"{youtube_result}\n\n"
                "Wikipedia:\n"
                f"{wiki_result}"
            )

            # Gerar resposta
            response = model.invoke(response_prompt.format(
                question=latest_question,
                optimized_query=optimized_query,
                resources=resources
            ))

            # Estruturar contexto para o nó de teaching
            extracted_context = {
                "text": resources,
                "image": {"type": "image", "content": None, "description": ""},
                "table": {"type": "table", "content": None},
                "relevance_analysis": {
                    "text": {"score": 1.0, "reason": "Informação obtida da web"},
                    "image": {"score": 0.0, "reason": "Nenhuma imagem disponível"},
                    "table": {"score": 0.0, "reason": "Nenhuma tabela disponível"},
                    "recommended_context": "text"
                }
            }

            # Atualizar estado
            new_state = state.copy()
            new_state["web_search_results"] = {
                "wikipedia": wiki_result,
                "youtube": youtube_result,
                "optimized_query": optimized_query  # Guardamos a query otimizada para referência
            }
            new_state["extracted_context"] = extracted_context
            new_state["messages"] = list(state["messages"]) + [response]

            print("[WEBSEARCH] Search completed successfully")
            return new_state

        except Exception as e:
            print(f"[WEBSEARCH] Error during web search: {str(e)}")
            error_message = "Desculpe, encontrei um erro ao buscar os recursos. Por favor, tente novamente."
            
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [
                AIMessage(content=error_message)
            ]
            return new_state

    return web_search

class TutorWorkflow:
    def __init__(self, qdrant_handler, student_email: str, disciplina: str, session_id: str, image_collection):
        print(f"\n[WORKFLOW] Initializing TutorWorkflow")

        initial_state = AgentState(
            messages=[],
            current_plan="",
            user_profile={},
            extracted_context={},  # Alterado para dict
            next_step=None,
            iteration_count=0,
            chat_history=[],
            needs_retrieval=True,
            evaluation_reason="",
            web_search_results={}
        )

        self.tools = RetrievalTools(
            qdrant_handler=qdrant_handler,
            student_email=student_email,
            disciplina=disciplina,
            session_id=session_id,
            image_collection=image_collection,
            state=initial_state
        )

        self.web_tools = WebSearchTools()
        self.workflow = self.create_workflow()

    def create_workflow(self) -> Graph:
        print("[WORKFLOW] Creating workflow graph")

        planning_node = create_answer_plan_node()
        route_node = route_after_plan_generation()
        retrieval_node = create_retrieval_node(self.tools)
        websearch_node = create_websearch_node(self.web_tools)
        teaching_node = create_teaching_node()

        workflow = Graph()

        workflow.add_node("generate_plan", planning_node)
        workflow.add_node("route_after_plan", route_node)
        workflow.add_node("retrieve_context", retrieval_node)
        workflow.add_node("web_search", websearch_node)
        workflow.add_node("teach", teaching_node)
        
        workflow.add_edge("generate_plan", "route_after_plan")
        workflow.add_conditional_edges(
            "route_after_plan",
            route_after_planning,
            {
                "retrieve_context": "retrieve_context",
                "web_search": "web_search",
                "direct_answer": "teach"
            }
        )
        workflow.add_edge("retrieve_context", "teach")
        workflow.add_edge("web_search", "teach")
        workflow.add_edge("teach", END)
        # END

        workflow.set_entry_point("generate_plan")
        print("[WORKFLOW] Workflow graph created successfully")
        return workflow.compile()

    async def invoke(self, query: str, student_profile: dict, current_plan=None, chat_history=None) -> dict:
        print(f"\n[WORKFLOW] Starting workflow invocation")
        print(f"[WORKFLOW] Query: {query}")
        
        try:
            # Validar perfil do usuário
            validated_profile = student_profile
            print(f"[WORKFLOW] Student profile validated: {validated_profile.get('EstiloAprendizagem', 'Not found')}")

            if chat_history is None:
                chat_history = []
            elif not isinstance(chat_history, list):
                chat_history = list(chat_history)

            recent_history = chat_history[-10:]
            print(f"[WORKFLOW] Using {len(recent_history)} recent chat messages")

            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                current_plan=current_plan if current_plan else "",
                user_profile=validated_profile,
                extracted_context={},
                next_step=None,
                iteration_count=0,
                chat_history=recent_history,
                needs_retrieval=True,
                evaluation_reason="",
                web_search_results={}
            )

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