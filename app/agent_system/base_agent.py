import asyncio
import base64
import json
import time
from typing import Any, Dict, List, Optional, TypedDict

from dataclasses import dataclass
from datetime import datetime, timezone

from langgraph.graph import END, Graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Supondo que os módulos abaixo já são assíncronos ou foram adaptados:
from database.vector_db import QdrantHandler
from database.mongo_database_manager import MongoDatabaseManager
from youtubesearchpython import VideosSearch
import wikipediaapi

##############################################
# MODELOS DE DADOS E TIPOS
##############################################

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
    next_step: Optional[str]
    iteration_count: int
    chat_history: List[BaseMessage]
    needs_retrieval: bool
    evaluation_reason: str
    web_search_results: Dict[str, str]
    answer_type: Optional[str]
    current_progress: Optional[Dict[str, Any]]
    session_id: str

##############################################
# GERENCIAMENTO DE PROGRESSO (MongoDB)
##############################################

class StudyProgressManager(MongoDatabaseManager):
    def __init__(self, db_name: str = "study_plans"):
        super().__init__()
        self.collection_name = db_name

    async def sync_progress_state(self, session_id: str) -> bool:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                return False

            plano_execucao = plan.get("plano_execucao", [])
            modified = False
            for step in plano_execucao:
                original_progress = step.get("progresso", 0)
                corrected_progress = min(max(float(original_progress), 0), 100)
                if original_progress != corrected_progress:
                    step["progresso"] = corrected_progress
                    modified = True

            if modified:
                total_steps = len(plano_execucao)
                progresso_total = sum(step["progresso"] for step in plano_execucao) / total_steps
                await collection.update_one(
                    {"id_sessao": session_id},
                    {
                        "$set": {
                            "plano_execucao": plano_execucao,
                            "progresso_total": round(progresso_total, 2),
                            "updated_at": datetime.now(timezone.utc)
                        }
                    }
                )
            return True

        except Exception as e:
            return False

    async def get_study_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one(
                {"id_sessao": session_id},
                {"_id": 0, "plano_execucao": 1, "progresso_total": 1}
            )
            if not plan:
                return None

            if "plano_execucao" in plan:
                for step in plan["plano_execucao"]:
                    if "progresso" not in step:
                        step["progresso"] = 0
                    else:
                        step["progresso"] = min(max(float(step["progresso"]), 0), 100)
                total_steps = len(plan["plano_execucao"])
                if total_steps > 0:
                    progresso_total = sum(step["progresso"] for step in plan["plano_execucao"]) / total_steps
                    plan["progresso_total"] = round(progresso_total, 2)

            return {
                "plano_execucao": plan.get("plano_execucao", []),
                "progresso_total": plan.get("progresso_total", 0)
            }
        except Exception as e:
            return None

    async def update_step_progress(self, session_id: str, step_index: int, new_progress: int) -> bool:
        try:
            if not 0 <= new_progress <= 100:
                raise ValueError("Progresso deve estar entre 0 e 100")
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                return False

            plano_execucao = plan.get("plano_execucao", [])
            if step_index >= len(plano_execucao):
                raise ValueError(f"Índice de etapa inválido: {step_index}")

            plano_execucao[step_index]["progresso"] = new_progress
            total_steps = len(plano_execucao)
            progresso_total = sum(step["progresso"] for step in plano_execucao) / total_steps

            result = await collection.update_one(
                {"id_sessao": session_id},
                {
                    "$set": {
                        "plano_execucao": plano_execucao,
                        "progresso_total": round(progresso_total, 2),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            return False

    async def get_step_details(self, session_id: str, step_index: int) -> Optional[Dict[str, Any]]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan or "plano_execucao" not in plan:
                return None
            plano_execucao = plan["plano_execucao"]
            if step_index >= len(plano_execucao):
                return None
            return plano_execucao[step_index]
        except Exception as e:
            return None

    async def mark_step_completed(self, session_id: str, step_index: int) -> bool:
        return await self.update_step_progress(session_id, step_index, 100)

    async def get_next_incomplete_step(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan or "plano_execucao" not in plan:
                return None
            for index, step in enumerate(plan["plano_execucao"]):
                if step.get("progresso", 0) < 100:
                    return {"index": index, "step": step}
            return None
        except Exception as e:
            return None

    async def get_study_summary(self, session_id: str) -> Dict[str, Any]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                return {"error": "Plano não encontrado", "session_id": session_id}
            plano_execucao = plan.get("plano_execucao", [])
            total_steps = len(plano_execucao)
            completed_steps = sum(1 for step in plano_execucao if step.get("progresso", 0) == 100)
            return {
                "session_id": session_id,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "progress_percentage": plan.get("progresso_total", 0),
                "started_at": plan.get("created_at"),
                "last_updated": plan.get("updated_at"),
                "estimated_duration": plan.get("duracao_total", "60 minutos")
            }
        except Exception as e:
            return {"error": str(e), "session_id": session_id}

##############################################
# FUNÇÕES UTILITÁRIAS: FORMATAÇÃO DE HISTÓRICO
##############################################

def filter_chat_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage):
            try:
                content = json.loads(msg.content)
                if isinstance(content, dict) and content.get("type") == "multimodal":
                    filtered_messages.append(AIMessage(content=content["content"]))
                else:
                    filtered_messages.append(msg)
            except json.JSONDecodeError:
                filtered_messages.append(msg)
    return filtered_messages

def format_chat_history(messages: List[BaseMessage], max_messages: int = 3) -> str:
    filtered_messages = filter_chat_history(messages[-max_messages:])
    formatted_history = []
    for msg in filtered_messages:
        role = 'Aluno' if isinstance(msg, HumanMessage) else 'Tutor'
        if isinstance(msg.content, str):
            formatted_history.append(f"{role}: {msg.content}")
    return "\n".join(formatted_history)

##############################################
# NÓS E FUNÇÕES DO FLUXO DO AGENTE (WORKFLOW)
##############################################

# 1. Node: Avaliação da Pergunta
async def create_question_evaluation_node():
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

    async def evaluate_question(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])
        response = await model.ainvoke(prompt.format(
            chat_history=chat_history,
            question=latest_question
        ))
        try:
            evaluation = json.loads(response.content)
            new_state = state.copy()
            new_state["needs_retrieval"] = evaluation["needs_retrieval"]
            new_state["evaluation_reason"] = evaluation["reason"]
            return new_state
        except json.JSONDecodeError:
            new_state = state.copy()
            new_state["needs_retrieval"] = True
            new_state["evaluation_reason"] = "Error in evaluation, defaulting to retrieval"
            return new_state

    return evaluate_question

# 2. Ferramentas de Recuperação de Contexto
class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
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
Retorne um JSON no seguinte formato (sem usar chaves):
"texto": score 0.0, reason "string";
"imagem": score 0.0, reason "string";
"tabela": score 0.0, reason "string";
"recommended_context": "text|image|table|combined"

Mantenha o formato exato e use apenas aspas duplas.
"""

    async def transform_question(self, question: str) -> str:
        formatted_history = format_chat_history(self.state["chat_history"], max_messages=4)
        prompt = ChatPromptTemplate.from_template(self.QUESTION_TRANSFORM_PROMPT)
        response = await self.model.ainvoke(prompt.format(
            chat_history=formatted_history,
            question=question,
        ))
        return response.content.strip()

    async def parallel_context_retrieval(self, question: str) -> Dict[str, Any]:
        text_task = asyncio.create_task(self.transform_question(question))
        image_task = asyncio.create_task(self.transform_question(question))
        table_task = asyncio.create_task(self.transform_question(question))
        text_question, image_question, table_question = await asyncio.gather(text_task, image_task, table_task)

        text_context_task = asyncio.create_task(self.retrieve_text_context(text_question))
        image_context_task = asyncio.create_task(self.retrieve_image_context(image_question))
        table_context_task = asyncio.create_task(self.retrieve_table_context(table_question))
        text_context, image_context, table_context = await asyncio.gather(text_context_task, image_context_task, table_context_task)

        relevance_analysis = await self.analyze_context_relevance(
            original_question=question,
            text_context=text_context,
            image_context=image_context,
            table_context=table_context
        )
        print("Relevance analysis:")
        print(relevance_analysis)
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
            if not results:
                return {"type": "image", "content": None, "description": ""}
            image_uuid = results[0].metadata.get("image_uuid")
            if not image_uuid:
                return {"type": "image", "content": None, "description": ""}
            return await self.retrieve_image_and_description(image_uuid)
        except Exception as e:
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
            return {"type": "table", "content": None}

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        try:
            image_data = await self.image_collection.find_one({"_id": image_uuid})
            if not image_data:
                return {"type": "error", "message": "Imagem não encontrada"}
            image_bytes = image_data.get("image_data")
            if not image_bytes:
                return {"type": "error", "message": "Dados da imagem ausentes"}
            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
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
            if not results:
                return {"type": "error", "message": "Descrição da imagem não encontrada"}
            return {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"type": "error", "message": str(e)}

    async def analyze_context_relevance(self, original_question: str, text_context: str, image_context: Dict[str, Any], table_context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not any([text_context, image_context, table_context]):
                return self._get_default_analysis()

            prompt = ChatPromptTemplate.from_template(self.RELEVANCE_ANALYSIS_PROMPT)
            image_description = ""
            if image_context and isinstance(image_context, dict):
                image_description = image_context.get("description", "")
            table_content = ""
            if table_context and isinstance(table_context, dict):
                table_content = table_context.get("content", "")

            text_preview = str(text_context)[:500] + "..." if text_context and len(str(text_context)) > 500 else str(text_context or "")
            image_preview = str(image_description)[:500] + "..." if len(str(image_description)) > 500 else str(image_description)
            table_preview = str(table_content)[:500] + "..." if len(str(table_content)) > 500 else str(table_content)

            response = await self.model.ainvoke(prompt.format(
                question=original_question,
                text_context=text_preview,
                image_context=image_preview,
                table_context=table_preview
            ))
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            analysis = json.loads(cleaned_content)
            required_fields = ["text", "image", "table", "recommended_context"]
            if not all(field in analysis for field in required_fields):
                raise ValueError("Missing required fields in analysis")
            return analysis
        except Exception as e:
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "text": {"score": 0, "reason": "Default fallback due to analysis error"},
            "image": {"score": 0, "reason": "Default fallback due to analysis error"},
            "table": {"score": 0, "reason": "Default fallback due to analysis error"},
            "recommended_context": "combined"
        }

# 3. Node: Recuperação de Contexto
def create_retrieval_node(tools: RetrievalTools):
    async def retrieve_context(state: AgentState) -> AgentState:
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        tools.state = state
        context_results = await tools.parallel_context_retrieval(latest_message.content)
        new_state = state.copy()
        new_state["extracted_context"] = context_results
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        return new_state
    return retrieve_context

# 4. Node: Roteamento após Geração do Plano
def route_after_plan_generation():
    ROUTE_PROMPT = """
Você é um assistente educacional que avalia o plano de resposta gerado e decide o próximo passo baseado na pergunta do aluno.

Pergunta do Aluno:
{question}

Histórico da Conversa:
{chat_history}

Plano de Resposta Gerado:
{plan}

No Banco de dados (retrieval), há vários materiais de estudo, como textos, imagens e tabelas, que podem ser relevantes para a pergunta do aluno.

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

    async def route_after_plan(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])
        current_plan = state.get("current_plan", "{}")
        try:
            response = await model.ainvoke(prompt.format(
                question=latest_question,
                chat_history=chat_history,
                plan=current_plan
            ))
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            route = json.loads(cleaned_content)
            if route.get("next_step") not in ["websearch", "retrieval", "direct_answer"]:
                raise ValueError(f"Invalid next_step value: {route.get('next_step')}")
            new_state = state.copy()
            new_state["next_step"] = route["next_step"]
            return new_state
        except Exception as e:
            question_lower = latest_question.lower()
            next_step = "websearch" if any(keyword in question_lower for keyword in ["youtube", "video", "wikipedia", "web"]) else "retrieval"
            new_state = state.copy()
            new_state["next_step"] = next_step
            return new_state

    return route_after_plan

# 5. Node: Geração do Plano de Resposta
def create_answer_plan_node():
    PLANNING_PROMPT = """ROLE: Assistente Educacional Especializado em Personalização de Aprendizagem e Avaliação de Atividades

OBJETIVO: Criar um plano de resposta personalizado e/ou avaliar respostas de atividades do aluno, considerando seu perfil e progresso atual.

PERFIL DO ALUNO:
Nome: {nome}
Estilo de Aprendizagem:
- Percepção: {percepcao} (Sensorial/Intuitivo)
    → Sensorial: preferência por fatos, dados, exemplos práticos
    → Intuitivo: preferência por conceitos, teorias, significados
- Entrada: {entrada} (Visual/Verbal)
    → Visual: aprende melhor com imagens/videos
    → Verbal: aprende melhor com explicações escritas e faladas
- Processamento: {processamento} (Ativo/Reflexivo)
    → Ativo: aprende fazendo, experimenta, trabalha em grupo
    → Reflexivo: aprende pensando, analisa, trabalha sozinho
- Entendimento: {entendimento} (Sequencial/Global)
    → Sequencial: aprende em passos lineares, passo a passo
    → Global: aprende em saltos, visão do todo primeiro

CONTEXTO ATUAL:
Etapa do Plano:
- Título: {titulo}
- Descrição: {descricao}
- Progresso: {progresso}%

HISTÓRICO:
{chat_history}

ENTRADA DO ALUNO:
{question}

ANÁLISE INICIAL OBRIGATÓRIA:
[Instruções detalhadas...]

RETORNE JSON COM A ESTRUTURA:
"tipo_entrada": "nova_pergunta|resposta_atividade|duvida_atividade|resposta_parcial|pedido_ajuda",
"contexto_identificado": "string detalhada",
"alinhamento_plano": boolean,
"analise_resposta": "atividade_referencia: string, completude: 0-100, precisao: 0-100, pontos_fortes: [string], pontos_melhoria: [string], feedback: string",
"estrutura_resposta": "lista com cada item contendo: parte, objetivo, abordagem (string baseada no estilo de aprendizagem)",
"recursos_sugeridos": "lista com cada item contendo: tipo, formato, motivo",
"nova_atividade": "descricao: string, objetivo: string, criterios_conclusao: [string], nivel_dificuldade: string",
"indicadores_compreensao": "[string]",
"nivel_resposta": "basico|intermediario|avancado",
"proxima_acao": "string",
"revisao_necessaria": boolean,
"ajuste_plano": "necessario: boolean, motivo: string, sugestao: string"
"""
    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def generate_plan(state: AgentState) -> AgentState:
        try:
            latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
            try:
                plano_execucao = json.loads(state["current_plan"])
                current_step = identify_current_step(plano_execucao["plano_execucao"])
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError("Invalid execution plan format")

            user_profile = state.get("user_profile", {})
            if not user_profile:
                raise ValueError("User profile not found in state")
            estilos = user_profile.get("EstiloAprendizagem", {})
            for style in ["Percepcao", "Entrada", "Processamento", "Entendimento"]:
                if style not in estilos:
                    estilos[style] = "não especificado"

            chat_history = format_chat_history(state["chat_history"])
            activity_context = analyze_activity_history(state["chat_history"])

            params = {
                "nome": user_profile.get("Nome", "Estudante"),
                "percepcao": estilos.get("Percepcao", "não especificado"),
                "entrada": estilos.get("Entrada", "não especificado"),
                "processamento": estilos.get("Processamento", "não especificado"),
                "entendimento": estilos.get("Entendimento", "não especificado"),
                "titulo": current_step.titulo,
                "descricao": current_step.descricao,
                "progresso": current_step.progresso,
                "question": latest_question,
                "chat_history": chat_history
            }
            response = await model.ainvoke(prompt.format(**params))
            plan = process_model_response(response.content)
            print("Generated plan:")
            print(plan)
            if activity_context["is_activity_response"]:
                plan = adjust_plan_for_activity(plan, activity_context)
            validate_plan_structure(plan)
            new_state = state.copy()
            new_state["current_plan"] = json.dumps(plan)
            new_state["next_step"] = determine_next_step(plan)
            new_state["activity_context"] = activity_context
            return new_state

        except Exception as e:
            import traceback
            traceback.print_exc()
            return handle_plan_generation_error(state, str(e))

    def analyze_activity_history(chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            recent_messages = chat_history[-5:]
            is_activity_response = False
            last_activity = None
            for msg in reversed(recent_messages):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, str) and ("atividade_pratica" in content.lower() or "exercício" in content.lower()):
                        last_activity = content
                        break
            if recent_messages and isinstance(recent_messages[-1], HumanMessage):
                last_user_msg = recent_messages[-1].content.lower()
                if last_activity:
                    is_activity_response = True
            return {
                "is_activity_response": is_activity_response,
                "last_activity": last_activity,
                "activity_reference": last_activity
            }
        except Exception as e:
            return {"is_activity_response": False, "last_activity": None, "activity_reference": None}

    def process_model_response(content: str) -> Dict[str, Any]:
        try:
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            plan = json.loads(cleaned_content)
            return plan
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON in model response")

    def adjust_plan_for_activity(plan: Dict[str, Any], activity_context: Dict[str, Any]) -> Dict[str, Any]:
        if activity_context["is_activity_response"]:
            plan["tipo_entrada"] = "resposta_atividade"
            plan["analise_resposta"] = {
                "atividade_referencia": activity_context["activity_reference"],
                "completude": 0,
                "precisao": 0,
                "pontos_fortes": [],
                "pontos_melhoria": [],
                "feedback": "Análise pendente"
            }
        return plan

    def validate_plan_structure(plan: Dict[str, Any]) -> None:
        required_fields = ["tipo_entrada", "contexto_identificado", "alinhamento_plano", "estrutura_resposta", "nivel_resposta", "proxima_acao"]
        missing_fields = [field for field in required_fields if field not in plan]
        if missing_fields:
            raise ValueError(f"Missing required fields in plan: {missing_fields}")

    def determine_next_step(plan: Dict[str, Any]) -> str:
        if plan.get("tipo_entrada") == "resposta_atividade":
            return "direct_answer"
        return plan.get("proxima_acao", "retrieval")

    def handle_plan_generation_error(state: AgentState, error_msg: str) -> AgentState:
        default_plan = {
            "tipo_entrada": "erro",
            "contexto_identificado": f"Erro na geração do plano: {error_msg}",
            "alinhamento_plano": True,
            "estrutura_resposta": [{
                "parte": "Resposta básica",
                "objetivo": "Fornecer informação solicitada",
                "abordagem": "Direta e simples"
            }],
            "recursos_sugeridos": [],
            "nova_atividade": {
                "descricao": "N/A",
                "objetivo": "N/A",
                "criterios_conclusao": [],
                "nivel_dificuldade": "básico"
            },
            "indicadores_compreensao": ["Compreensão básica do conceito"],
            "nivel_resposta": "básico",
            "proxima_acao": "retrieval",
            "revisao_necessaria": False
        }
        new_state = state.copy()
        new_state["current_plan"] = json.dumps(default_plan)
        new_state["next_step"] = "retrieval"
        new_state["error"] = error_msg
        return new_state

    return generate_plan

# 6. Node: Geração da Resposta (Ensino/Tutoria)
def create_teaching_node():
    CONTEXT_BASED_PROMPT = """
ROLE: Tutor especializado em explicar materiais e contextos educacionais.
    
OBJETIVO: Explicar detalhadamente o contexto fornecido e guiar o aluno na compreensão do material.

Plano de Resposta:
{learning_plan}

Perfil do Aluno:
{user_profile}

Fonte de Informação:
{source_type}

Contexto Principal (OBRIGATÓRIO EXPLICAR):
{primary_context}

Contextos Secundários (EXPLICAR SE RELEVANTES):
{secondary_contexts}

Histórico da Conversa:
{chat_history}

Mensagem do aluno:
{question}

ESTRUTURA DA RESPOSTA:
- Se houver contexto principal, explique detalhadamente.
- Responda como um tutor educacional sempre orientando o aluno a chegar na resposta e dando próximos passos.

DIRETRIZES:
- NUNCA presuma conhecimento prévio sem explicar.
- NUNCA pule a explicação do contexto principal.
- Use linguagem clara e objetiva.
- Mantenha foco educacional.
- Incentive a busca da informação pelo aluno.
    """
    
    DIRECT_RESPONSE_PROMPT = """
ROLE: Tutor educacional

TASK: Guiar o aluno na compreensão e resolução de questões sem dar respostas diretas.

INSTRUCTIONS:
Leia atentamente o plano de resposta e a mensagem do aluno. Forneça orientações e dicas para ajudar o aluno a alcançar o objetivo do plano.
- Plano de Resposta:
{learning_plan}

- Perfil do Aluno:
{user_profile}

- Histórico da Conversa:
{chat_history}

- Mensagem do aluno:
{question}

ESTRUTURA DA RESPOSTA:
- Responda incentivando o raciocínio do aluno, sem dar a resposta final.
    """

    context_prompt = ChatPromptTemplate.from_template(CONTEXT_BASED_PROMPT)
    direct_prompt = ChatPromptTemplate.from_template(DIRECT_RESPONSE_PROMPT)
    model = ChatOpenAI(model="gpt-4o", temperature=0.2)

    async def generate_teaching_response(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])
        try:
            # Escolhe o prompt com base no valor de next_step
            if state.get("next_step") == "direct_answer":
                explanation = await model.ainvoke(
                    direct_prompt.format(
                        learning_plan=state["current_plan"],
                        user_profile=state["user_profile"],
                        question=latest_question,
                        chat_history=chat_history
                    )
                )
            else:
                if state.get("web_search_results"):
                    source_type = "Resultados de busca web"
                    web_results = state["web_search_results"]
                    primary_context = f"Wikipedia:\n{web_results.get('wikipedia', 'Não disponível')}"
                    secondary_contexts = f"YouTube:\n{web_results.get('youtube', 'Não disponível')}"
                else:
                    extracted = state.get("extracted_context", {})
                    relevance = extracted.get("relevance_analysis", {})
                    context_scores = {
                        "text": relevance.get("text", {}).get("score", 0),
                        "image": relevance.get("image", {}).get("score", 0),
                        "table": relevance.get("table", {}).get("score", 0)
                    }
                    sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
                    source_type = "Material de estudo"
                    primary_type = sorted_contexts[0][0] if sorted_contexts else "text"
                    if primary_type == "text":
                        primary_context = f"Texto: {extracted.get('text', '')}"
                    elif primary_type == "image":
                        primary_context = f"Imagem: {extracted.get('image', {}).get('description', '')}"
                    else:
                        primary_context = f"Tabela: {extracted.get('table', {}).get('content', '')}"
                    secondary_list = []
                    for ctx_type, score in sorted_contexts[1:]:
                        if score > 0.3:
                            if ctx_type == "text":
                                secondary_list.append(f"Texto Complementar: {extracted.get('text', '')}")
                            elif ctx_type == "image":
                                secondary_list.append(f"Descrição da Imagem: {extracted.get('image', {}).get('description', '')}")
                            elif ctx_type == "table":
                                secondary_list.append(f"Dados da Tabela: {extracted.get('table', {}).get('content', '')}")
                    secondary_contexts = "\n\n".join(secondary_list)
                explanation = await model.ainvoke(
                    context_prompt.format(
                        learning_plan=state["current_plan"],
                        user_profile=state["user_profile"],
                        source_type=source_type,
                        primary_context=primary_context,
                        secondary_contexts=secondary_contexts,
                        question=latest_question,
                        chat_history=chat_history
                    )
                )
            # Independente do ramo escolhido, verificar se há imagem a ser retornada
            image_content = None
            extracted = state.get("extracted_context", {})
            if extracted.get("image") and extracted["image"].get("type") == "image" and extracted["image"].get("image_bytes"):
                image_score = extracted.get("relevance_analysis", {}).get("image", {}).get("score", 0)
                if image_score > 0.3:
                    image_content = extracted["image"]["image_bytes"]
            print("Image content:", image_content)
            print("Explanation content:", explanation.content)

            # Monta a resposta multimodal se houver imagem
            if image_content:
                base64_image = base64.b64encode(image_content).decode('utf-8')
                response_content = {
                    "type": "multimodal",
                    "content": explanation.content,
                    "image": f"data:image/jpeg;base64,{base64_image}"
                }
                response = AIMessage(content=json.dumps(response_content))
                history_message = AIMessage(content=explanation.content)
                print("Multimodal response generated")
                print(response_content)
            else:
                response = explanation
                history_message = explanation

            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                history_message
            ]
            return new_state

        except Exception as e:
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

# 7. Node: Análise do Progresso
def create_progress_analyst_node(progress_manager: StudyProgressManager):
    ANALYSIS_PROMPT = """Você é um analista especializado em avaliar o progresso de aprendizado baseado em interações.

Histórico da Conversa:
{chat_history}

Plano de Execução Atual:
{current_plan}

Etapa Atual:
Título: {step_title}
Descrição: {step_description}
Progresso Atual: {current_progress}%

Analise a última interação e determine:
1. O nível de compreensão demonstrado pelo aluno
2. Se houve progresso efetivo no aprendizado
3. Quanto o progresso deve aumentar (0-100%)
4. Se a etapa atual deve ser considerada concluída
5. Retorne APENAS um JSON válido no seguinte formato EXATO:
"comprehension_level": "alto|medio|baixo",
"progress_made": true|false,
"progress_increment": number,
"step_completed": boolean,
"reasoning": "string"
"""
    prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def analyze_progress(state: AgentState) -> AgentState:
        try:
            messages = state['messages']
            chat_history = state['chat_history']
            current_plan = state['current_plan']
            session_id = state['session_id']
            study_progress = await progress_manager.get_study_progress(session_id)
            if not study_progress:
                return state

            plano_execucao = study_progress['plano_execucao']
            current_step = None
            step_index = 0
            for idx, step in enumerate(plano_execucao):
                if step['progresso'] < 100:
                    current_step = step
                    step_index = idx
                    break
            if not current_step:
                return state

            formatted_chat_history = format_chat_history(chat_history, max_messages=5)
            analysis_response = await model.ainvoke(prompt.format(
                chat_history=formatted_chat_history,
                current_plan=current_plan,
                step_title=current_step['titulo'],
                step_description=current_step['descricao'],
                current_progress=current_step['progresso']
            ))
            cleaned_content = analysis_response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            analysis = json.loads(cleaned_content)
            if analysis["progress_made"]:
                current_progress = current_step['progresso']
                new_progress = min(current_progress + analysis["progress_increment"], 100)
                if analysis["step_completed"]:
                    new_progress = 100
                update_success = await progress_manager.update_step_progress(session_id, step_index, new_progress)
                if update_success:
                    study_summary = await progress_manager.get_study_summary(session_id)
                    new_state = state.copy()
                    new_state.update({
                        'study_summary': study_summary,
                        'progress_analysis': analysis,
                        'current_progress': {
                            'plano_execucao': plano_execucao,
                            'step': current_step,
                            'step_index': step_index,
                            'progresso_total': study_summary['progress_percentage']
                        }
                    })
                    return new_state
                else:
                    return state
            return state

        except Exception as e:
            import traceback
            traceback.print_exc()
            return state

    return analyze_progress

# Função auxiliar para identificar a etapa atual
def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    if not plano_execucao:
        raise ValueError("Plano de execução vazio")
    for step in plano_execucao:
        if "progresso" not in step:
            step["progresso"] = 0
        else:
            step["progresso"] = min(max(float(step["progresso"]), 0), 100)
    for step in plano_execucao:
        if step["progresso"] < 100:
            return ExecutionStep(
                titulo=step["titulo"],
                duracao=step["duracao"],
                descricao=step["descricao"],
                conteudo=step["conteudo"],
                recursos=step["recursos"],
                atividade=step["atividade"],
                progresso=step["progresso"]
            )
    last_step = plano_execucao[-1]
    return ExecutionStep(
        titulo=last_step["titulo"],
        duracao=last_step["duracao"],
        descricao=last_step["descricao"],
        conteudo=last_step["conteudo"],
        recursos=last_step["recursos"],
        atividade=last_step["atividade"],
        progresso=last_step["progresso"]
    )

def should_continue(state: AgentState) -> str:
    MAX_ITERATIONS = 1
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"
    return "continue"

def route_after_evaluation(state: AgentState) -> str:
    return "retrieve_context" if state.get("needs_retrieval", True) else "direct_answer"

# 8. Ferramentas de Web Search
class WebSearchTools:
    def __init__(self):
        pass

    def search_youtube(self, query: str) -> str:
        try:
            videos_search = VideosSearch(query, limit=3)
            results = videos_search.result()
            if results['result']:
                videos_info = []
                for video in results['result'][:3]:
                    video_info = {
                        'title': video['title'],
                        'link': video['link'],
                        'channel': video.get('channel', {}).get('name', 'N/A'),
                        'duration': video.get('duration', 'N/A'),
                        'description': video.get('descriptionSnippet', [{'text': 'Sem descrição'}])[0]['text']
                    }
                    videos_info.append(video_info)
                response = "Vídeos encontrados:\n\n"
                for i, video in enumerate(videos_info, 1):
                    response += (
                        f"{i}. Título: {video['title']}\n"
                        f"   Link: {video['link']}\n"
                        f"   Canal: {video['channel']}\n"
                        f"   Duração: {video['duration']}\n"
                        f"   Descrição: {video['description']}\n\n"
                    )
                return response
            else:
                return "Nenhum vídeo encontrado."
        except Exception as e:
            return "Ocorreu um erro ao buscar no YouTube."

    def search_wikipedia(self, query: str) -> str:
        try:
            wiki_wiki = wikipediaapi.Wikipedia('TutorBot/1.0 (pericles.junior@cesar.school)', 'pt')
            page = wiki_wiki.page(query)
            if page.exists():
                summary = (
                    f"Título: {page.title}\n"
                    f"Resumo: {page.summary[:500]}...\n"
                    f"Link: {page.fullurl}"
                )
                return summary
            else:
                return "Página não encontrada na Wikipedia."
        except Exception as e:
            return "Ocorreu um erro ao buscar na Wikipedia."

def route_after_planning(state: AgentState) -> str:
    next_step = state.get("next_step", "retrieval")
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
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])
        try:
            optimized_query = (await model.ainvoke(query_prompt.format(
                question=latest_question,
                chat_history=chat_history
            ))).content.strip()
            wiki_result = web_tools.search_wikipedia(optimized_query)
            youtube_result = web_tools.search_youtube(optimized_query)
            resources = (
                "YouTube:\n" + f"{youtube_result}\n\n" +
                "Wikipedia:\n" + f"{wiki_result}"
            )
            response = await model.ainvoke(response_prompt.format(
                question=latest_question,
                optimized_query=optimized_query,
                resources=resources
            ))
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
            new_state = state.copy()
            new_state["web_search_results"] = {
                "wikipedia": wiki_result,
                "youtube": youtube_result,
                "optimized_query": optimized_query
            }
            new_state["extracted_context"] = extracted_context
            new_state["messages"] = list(state["messages"]) + [AIMessage(content=response.content)]
            return new_state

        except Exception as e:
            error_message = "Desculpe, encontrei um erro ao buscar os recursos. Por favor, tente novamente."
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [AIMessage(content=error_message)]
            return new_state

    return web_search

##############################################
# CLASSE: TutorWorkflow
##############################################

class TutorWorkflow:
    def __init__(self, qdrant_handler, student_email: str, disciplina: str, session_id: str, image_collection):
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.progress_manager = StudyProgressManager()
        initial_state = AgentState(
            messages=[],
            current_plan="",
            user_profile={},
            extracted_context={},
            next_step=None,
            iteration_count=0,
            chat_history=[],
            needs_retrieval=True,
            evaluation_reason="",
            web_search_results={},
            current_progress=None,
            session_id=session_id
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
        workflow = Graph()
        workflow.add_node("generate_plan", create_answer_plan_node())
        workflow.add_node("route_after_plan", route_after_plan_generation())
        workflow.add_node("retrieve_context", create_retrieval_node(self.tools))
        workflow.add_node("web_search", create_websearch_node(self.web_tools))
        workflow.add_node("teach", create_teaching_node())
        workflow.add_node("progress_analyst", create_progress_analyst_node(self.progress_manager))
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
        workflow.add_edge("teach", "progress_analyst")
        workflow.add_edge("progress_analyst", END)
        workflow.set_entry_point("generate_plan")
        return workflow.compile()

    async def handle_progress_update(self, session_id: str, step_index: int, new_progress: int) -> Optional[Dict[str, Any]]:
        try:
            success = await self.progress_manager.update_step_progress(session_id, step_index, new_progress)
            if success:
                return await self.progress_manager.get_study_summary(session_id)
            return None
        except Exception as e:
            return None

    def create_planning_node_with_progress(self):
        planning_node = create_answer_plan_node()
        progress_manager = self.progress_manager

        async def planning_with_progress(state: AgentState) -> AgentState:
            try:
                current_progress = await progress_manager.get_study_progress(self.session_id)
                if current_progress:
                    state["current_progress"] = current_progress
                new_state = await planning_node(state)
                return new_state
            except Exception as e:
                import traceback
                traceback.print_exc()
                return state

        return planning_with_progress

    def create_teaching_node_with_progress(self):
        teaching_node = create_teaching_node()

        async def teaching_with_progress(state: AgentState) -> AgentState:
            try:
                new_state = await teaching_node(state)
                return new_state
            except Exception as e:
                import traceback
                traceback.print_exc()
                return state

        return teaching_with_progress

    async def invoke(self, query: str, student_profile: dict, current_plan=None, chat_history=None) -> dict:
        start_time = time.time()
        try:
            validated_profile = student_profile
            await self.progress_manager.sync_progress_state(self.session_id)
            current_progress = await self.progress_manager.get_study_progress(self.session_id)
            if chat_history is None:
                chat_history = []
            elif not isinstance(chat_history, list):
                chat_history = list(chat_history)
            recent_history = chat_history[-10:]
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
                web_search_results={},
                current_progress=current_progress,
                session_id=self.session_id
            )
            result = await self.workflow.ainvoke(initial_state)
            study_summary = await self.progress_manager.get_study_summary(self.session_id)
            final_result = {
                "messages": result["messages"],
                "final_plan": result["current_plan"],
                "chat_history": result["chat_history"],
                "study_progress": study_summary
            }
            if "error" in result:
                final_result["error"] = result["error"]
            return final_result

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {
                "error": f"Erro na execução do workflow: {str(e)}",
                "messages": [AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")],
                "chat_history": recent_history
            }
            try:
                error_response["study_progress"] = await self.progress_manager.get_study_summary(self.session_id)
            except Exception as progress_error:
                print(f"[WORKFLOW] Error getting progress summary: {progress_error}")
            return error_response

        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[WORKFLOW] Workflow execution completed in {elapsed_time:.2f} seconds")
