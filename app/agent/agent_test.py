from typing import Any, TypedDict, List, Dict, Optional
from langgraph.graph import END, Graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import json
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from database.vector_db import QdrantHandler
from dataclasses import dataclass
import base64
import asyncio
from typing import Dict, Any
from youtubesearchpython import VideosSearch
import wikipediaapi
from database.mongo_database_manager import MongoDatabaseManager
import time
from datetime import datetime, timezone


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
    current_progress: Dict[str, Any]
    session_id: str


class StudyProgressManager(MongoDatabaseManager):
    def __init__(self, db_name: str = "study_plans"):
        """
        Inicializa o gerenciador de progresso de estudos.

        Args:
            db_name: Nome do banco de dados MongoDB
        """
        super().__init__()
        self.collection_name = db_name

    async def sync_progress_state(self, session_id: str) -> bool:
        """
        Sincroniza o estado do progresso, garantindo consistência entre o banco de dados e o estado da aplicação.
        """
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})

            if not plan:
                return False

            plano_execucao = plan.get("plano_execucao", [])
            modified = False

            # Validar e corrigir o progresso de cada etapa
            for step in plano_execucao:
                original_progress = step.get("progresso", 0)
                corrected_progress = min(max(float(original_progress), 0), 100)

                if original_progress != corrected_progress:
                    step["progresso"] = corrected_progress
                    modified = True

            if modified:
                # Recalcular e atualizar o progresso total
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
            #print(f"[PROGRESS] Erro na sincronização do progresso: {e}")
            return False

    async def get_study_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one(
                {"id_sessao": session_id},
                {"_id": 0, "plano_execucao": 1, "progresso_total": 1}
            )
            #print(f"[PROGRESS] (get_study_progress) Plano encontrado: {plan}")

            if not plan:
                #print(f"[PROGRESS] Plano não encontrado para sessão: {session_id}")
                return None

            # Validar e corrigir o progresso de cada etapa
            if "plano_execucao" in plan:
                for step in plan["plano_execucao"]:
                    if "progresso" not in step:
                        step["progresso"] = 0
                    else:
                        step["progresso"] = min(max(float(step["progresso"]), 0), 100)

                # Recalcular o progresso total
                total_steps = len(plan["plano_execucao"])
                if total_steps > 0:
                    progresso_total = sum(step["progresso"] for step in plan["plano_execucao"]) / total_steps
                    plan["progresso_total"] = round(progresso_total, 2)

            return {
                "plano_execucao": plan.get("plano_execucao", []),
                "progresso_total": plan.get("progresso_total", 0)
            }
        except Exception as e:
            #print(f"[PROGRESS] Erro ao recuperar progresso: {e}")
            return None

    async def update_step_progress(
        self,
        session_id: str,
        step_index: int,
        new_progress: int
    ) -> bool:
        """
        Atualiza o progresso de uma etapa específica do plano.

        Args:
            session_id: ID da sessão de estudo
            step_index: Índice da etapa no plano
            new_progress: Novo valor de progresso (0-100)

        Returns:
            bool indicando sucesso da operação
        """
        try:
            if not 0 <= new_progress <= 100:
                raise ValueError("Progresso deve estar entre 0 e 100")

            collection = self.db[self.collection_name]

            # Primeiro recupera o plano atual
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                #print(f"[PROGRESS] Plano não encontrado para sessão: {session_id}")
                return False

            # Atualiza o progresso da etapa específica
            plano_execucao = plan.get("plano_execucao", [])
            if step_index >= len(plano_execucao):
                raise ValueError(f"Índice de etapa inválido: {step_index}")

            plano_execucao[step_index]["progresso"] = new_progress

            # Calcula o progresso total
            total_steps = len(plano_execucao)
            progresso_total = sum(step["progresso"] for step in plano_execucao) / total_steps

            # Atualiza o documento
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
            #print(f"[PROGRESS] Erro ao atualizar progresso da etapa: {e}")
            return False

    async def get_step_details(
        self,
        session_id: str,
        step_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Recupera os detalhes de uma etapa específica do plano.

        Args:
            session_id: ID da sessão de estudo
            step_index: Índice da etapa no plano

        Returns:
            Dict contendo os detalhes da etapa ou None se não encontrado
        """
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
            #print(f"[PROGRESS] Erro ao recuperar detalhes da etapa: {e}")
            return None

    async def mark_step_completed(
        self,
        session_id: str,
        step_index: int
    ) -> bool:
        """
        Marca uma etapa como concluída (100% de progresso).

        Args:
            session_id: ID da sessão de estudo
            step_index: Índice da etapa no plano

        Returns:
            bool indicando sucesso da operação
        """
        return await self.update_step_progress(session_id, step_index, 100)

    async def get_next_incomplete_step(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Encontra a próxima etapa incompleta do plano.

        Args:
            session_id: ID da sessão de estudo

        Returns:
            Dict contendo os detalhes da próxima etapa incompleta ou None se todas estiverem completas
        """
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})

            if not plan or "plano_execucao" not in plan:
                return None

            for index, step in enumerate(plan["plano_execucao"]):
                if step.get("progresso", 0) < 100:
                    return {
                        "index": index,
                        "step": step
                    }

            return None  # Todas as etapas estão completas
        except Exception as e:
            #print(f"[PROGRESS] Erro ao buscar próxima etapa incompleta: {e}")
            return None

    async def get_study_summary(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Gera um resumo do progresso do plano de estudos.

        Args:
            session_id: ID da sessão de estudo

        Returns:
            Dict contendo o resumo do progresso
        """
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})

            if not plan:
                return {
                    "error": "Plano não encontrado",
                    "session_id": session_id
                }

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
            #print(f"[PROGRESS] Erro ao gerar resumo do estudo: {e}")
            return {
                "error": str(e),
                "session_id": session_id
            }

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
        #print("\n[NODE:EVALUATION] Starting question evaluation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        #print(f"[NODE:EVALUATION] Evaluating question: {latest_question}")

        chat_history = format_chat_history(state["chat_history"])

        response = model.invoke(prompt.format(
            chat_history=chat_history,
            question=latest_question
        ))

        try:
            evaluation = json.loads(response.content)
            #print(f"[NODE:EVALUATION] Evaluation result: {evaluation}")

            new_state = state.copy()
            new_state["needs_retrieval"] = evaluation["needs_retrieval"]
            new_state["evaluation_reason"] = evaluation["reason"]
            return new_state

        except json.JSONDecodeError as e:
            #print(f"[NODE:EVALUATION] Error parsing evaluation: {e}")
            # Default to performing retrieval in case of error
            new_state = state.copy()
            new_state["needs_retrieval"] = True
            new_state["evaluation_reason"] = "Error in evaluation, defaulting to retrieval"
            return new_state

    return evaluate_question

class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
        #print(f"[RETRIEVAL] Initializing RetrievalTools:")
        #print(f"[RETRIEVAL] - Student: {student_email}")
        #print(f"[RETRIEVAL] - Disciplina: {disciplina}")
        #print(f"[RETRIEVAL] - Session: {session_id}")
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

        ##print(f"[RETRIEVAL] Using chat history for question transformation:")
        #print(formatted_history)

        prompt = ChatPromptTemplate.from_template(self.QUESTION_TRANSFORM_PROMPT)
        response = await self.model.ainvoke(prompt.format(
            chat_history=formatted_history,
            question=question,
        ))

        transformed_question = response.content.strip()
        #print(f"[RETRIEVAL] Transformed question: {transformed_question}")
        return transformed_question

    async def parallel_context_retrieval(self, question: str) -> Dict[str, Any]:
        #print(f"\n[RETRIEVAL] Starting parallel context retrieval for: {question}")

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
            #print(f"[RETRIEVAL] Error in text retrieval: {e}")
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
            #print("")
            #print("--------------------------------------------------")
            #print(f"[RETRIEVAL] Image search results: {results}")
            #print("--------------------------------------------------")
            #print("")
            if not results:
                return {"type": "image", "content": None, "description": ""}

            image_uuid = results[0].metadata.get("image_uuid")
            if not image_uuid:
                return {"type": "image", "content": None, "description": ""}

            return await self.retrieve_image_and_description(image_uuid)
        except Exception as e:
            #print(f"[RETRIEVAL] Error in image retrieval: {e}")
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
            #print(f"[RETRIEVAL] Error in table retrieval: {e}")
            return {"type": "table", "content": None}

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        """
        Recupera a imagem e sua descrição de forma assíncrona.
        """
        try:
            #print(f"[RETRIEVAL] Recuperando imagem com UUID: {image_uuid}")
            image_data = await self.image_collection.find_one({"_id": image_uuid})
            if not image_data:
                #print(f"[RETRIEVAL] Imagem não encontrada: {image_uuid}")
                return {"type": "error", "message": "Imagem não encontrada"}

            image_bytes = image_data.get("image_data")
            if not image_bytes:
                #print("[RETRIEVAL] Dados da imagem ausentes")
                return {"type": "error", "message": "Dados da imagem ausentes"}

            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                #print(f"[RETRIEVAL] Formato de imagem não suportado: {type(image_bytes)}")
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
            #print(f"[RETRIEVAL] Resultados da busca de descrição: {results}")
            if not results:
                return {"type": "error", "message": "Descrição da imagem não encontrada"}
            #print("[RETRIEVAL] Imagem e descrição recuperadas com sucesso")
            #print(f"[RETRIEVAL] Descrição da imagem: {results[0].page_content}")
            return {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content
            }
        except Exception as e:
            #print(f"[RETRIEVAL] Erro ao recuperar imagem: {e}")
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
                #print("[RETRIEVAL] No context available for relevance analysis")
                return self._get_default_analysis()

            prompt = ChatPromptTemplate.from_template(self.RELEVANCE_ANALYSIS_PROMPT)

            image_description = ""
            if image_context and isinstance(image_context, dict):
                image_description = image_context.get("description", "")

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
                #print(f"[RETRIEVAL] Relevance analysis: {analysis}")

                required_fields = ["text", "image", "table", "recommended_context"]
                if not all(field in analysis for field in required_fields):
                    raise ValueError("Missing required fields in analysis")

                return analysis

            except json.JSONDecodeError as e:
                #print(f"[RETRIEVAL] Error parsing relevance analysis: {e}")
                #print(f"[RETRIEVAL] Invalid JSON content: {cleaned_content}")
                return self._get_default_analysis()
            except ValueError as e:
                #print(f"[RETRIEVAL] Validation error: {e}")
                return self._get_default_analysis()

        except Exception as e:
            #print(f"[RETRIEVAL] Error in analyze_context_relevance: {e}")
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
        #print("\n[NODE:RETRIEVAL] Starting retrieval node execution")
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        #print(f"[NODE:RETRIEVAL] Processing message: {latest_message.content}")

        tools.state = state
        context_results = await tools.parallel_context_retrieval(latest_message.content)

        new_state = state.copy()
        new_state["extracted_context"] = context_results
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        #print(f"[NODE:RETRIEVAL] Updated iteration count: {new_state['iteration_count']}")
        return new_state

    return retrieve_context

def route_after_plan_generation():
    ROUTE_PROMPT = """
    Você é um assistente educacional que avalia o plano de resposta gerado e decide o próximo passo baseado na pergunta do aluno.

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
        #print("\n[ROUTING] Starting route after plan generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        #print(f"[ROUTING] Processing question: {latest_question}")

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
            #print(f"[ROUTING] Route analysis: {route}")

            if route.get("next_step") not in ["websearch", "retrieval", "direct_answer"]:
                raise ValueError(f"Invalid next_step value: {route.get('next_step')}")

            new_state = state.copy()
            new_state["next_step"] = route["next_step"]
            return new_state

        except Exception as e:
            #print(f"[ROUTING] Error in routing: {str(e)}")
            # Em caso de erro, usamos uma lógica simples baseada em palavras-chave
            question_lower = latest_question.lower()
            if any(keyword in question_lower for keyword in ["youtube", "video", "wikipedia", "web"]):
                next_step = "websearch"
            else:
                next_step = "retrieval"

            #print(f"[ROUTING] Fallback routing decision: {next_step}")
            new_state = state.copy()
            new_state["next_step"] = next_step
            return new_state

    return route_after_plan

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

1. CLASSIFICAÇÃO DA ENTRADA [peso: crucial]
   DETERMINE O TIPO:
   - Pergunta nova
   - Resposta a atividade anterior
   - Dúvida sobre atividade
   - Resposta parcial/incompleta
   - Pedido de dica/ajuda

   Se for RESPOSTA A ATIVIDADE:
   - Identifique a atividade original no histórico
   - Compare com os critérios esperados
   - Avalie completude e precisão
   - Determine necessidade de correção ou complemento

2. CONTEXTUALIZAÇÃO [peso: crucial]
   - Identificar relação com conteúdo atual
   - Avaliar alinhamento com etapa do plano
   - OBRIGATÓRIO: Analisar histórico para encontrar atividade relacionada
   - OBRIGATÓRIO: Verificar critérios de avaliação anteriores
   - Métricas de Relevância:
     → Alta: Resposta direta a atividade ou pergunta relevante
     → Média: Relacionada a atividades anteriores
     → Baixa: Fora do contexto das atividades

3. AVALIAÇÃO DE RESPOSTA (quando aplicável)
   Critérios:
   A) Completude
      - Todos os pontos solicitados foram abordados
      - Profundidade adequada ao nível
      - Coerência com a pergunta/atividade

   B) Precisão
      - Correção técnica/conceitual
      - Adequação ao nível esperado
      - Uso apropriado de termos/conceitos

   C) Engajamento
      - Evidência de reflexão
      - Aplicação de conceitos
      - Criatividade/originalidade

4. PERSONALIZAÇÃO DA RESPOSTA [peso: crucial]
   [Manter toda a seção de personalização anterior]

5. FEEDBACK E PRÓXIMOS PASSOS
   Para Respostas de Atividades:
   - Fornecer feedback construtivo
   - Identificar pontos fortes
   - Sugerir melhorias específicas
   - Propor próximos desafios
   - Conectar com próximos conceitos

   Para Novas Perguntas:
   [Manter estrutura anterior de recursos e atividades]

ATENÇÃO ESPECIAL:
1. SEMPRE verifique se é resposta a uma atividade anterior
2. NUNCA ignore os critérios estabelecidos na atividade original
3. Mantenha feedback construtivo e motivador
4. Adapte próximos passos baseado no desempenho
5. Use linguagem apropriada ao nível do aluno

RESPOSTA OBRIGATÓRIA:
Retornar JSON com estrutura exata:

    "tipo_entrada": "nova_pergunta|resposta_atividade|duvida_atividade|resposta_parcial|pedido_ajuda",
    "contexto_identificado": "string detalhada",
    "alinhamento_plano": boolean,
    "analise_resposta": 
        "atividade_referencia": "string",
        "completude": 0-100,
        "precisao": 0-100,
        "pontos_fortes": ["string"],
        "pontos_melhoria": ["string"],
        "feedback": "string",
    "estrutura_resposta": [
        
            "parte": "string",
            "objetivo": "string",
            "abordagem": "string baseada no estilo de aprendizagem"
        
    ],
    "recursos_sugeridos": [
        
            "tipo": "string",
            "formato": "string",
            "motivo": "string"
        
    ],
    "nova_atividade":
        "descricao": "string",
        "objetivo": "string",
        "criterios_conclusao": ["string"],
        "nivel_dificuldade": "string",
    "indicadores_compreensao": ["string"],
    "nivel_resposta": "basico|intermediario|avancado",
    "proxima_acao": "string",
    "revisao_necessaria": boolean,
    "ajuste_plano":
        "necessario": boolean,
        "motivo": "string",
        "sugestao": "string"


MÉTRICAS DE QUALIDADE:
- Identificação: Precisão na classificação do tipo de entrada
- Contextualização: Conexão com atividades anteriores
- Avaliação: Qualidade do feedback e análise
- Personalização: Adaptação ao perfil do aluno
- Progressão: Contribuição para o avanço do aluno
- Motivação: Capacidade de manter o engajamento

REGRAS DE OURO:
1. SEMPRE identifique se é resposta a atividade antes de planejar
2. SEMPRE forneça feedback construtivo
3. NUNCA ignore o histórico de atividades
4. SEMPRE adapte o próximo passo ao desempenho atual
5. MANTENHA o foco no objetivo de aprendizagem"""

    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate_plan(state: AgentState) -> AgentState:
        #print("\n[NODE:PLANNING] Starting plan generation")
        
        try:
            # Extrair última mensagem
            latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
            #print(f"[NODE:PLANNING] Processing input: {latest_question}")

            # Processar o plano de execução atual
            try:
                # Verificar se o plano já é um dicionário ou uma string JSON
                current_plan = state["current_plan"]
                if isinstance(current_plan, dict):
                    plano_execucao = current_plan
                elif isinstance(current_plan, str) and current_plan.strip():
                    plano_execucao = json.loads(current_plan)
                else:
                    raise ValueError("Empty or invalid execution plan")
                    
                # Garantir que o plano tenha o formato esperado
                if "plano_execucao" not in plano_execucao:
                    raise KeyError("Missing 'plano_execucao' key in plan")
                    
                current_step = identify_current_step(plano_execucao["plano_execucao"])
                #print(f"[PLANNING] Current step: {current_step.titulo} ({current_step.progresso}%)")
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                #print(f"[PLANNING] Error processing execution plan: {e}")
                raise ValueError(f"Invalid execution plan format: {str(e)}")

            # Extrair e validar perfil do usuário
            user_profile = state.get("user_profile", {})
            if not user_profile:
                raise ValueError("User profile not found in state")

            estilos = user_profile.get("EstiloAprendizagem", {})
            required_styles = ["Percepcao", "Entrada", "Processamento", "Entendimento"]
            missing_styles = [style for style in required_styles if style not in estilos]
            if missing_styles:
                #print(f"[PLANNING] Warning: Missing learning styles: {missing_styles}")
                for style in missing_styles:
                    estilos[style] = "não especificado"

            # Preparar histórico do chat com contexto relevante
            chat_history = format_chat_history(state["chat_history"])
            
            # Analisar histórico para identificar atividades anteriores
            activity_context = analyze_activity_history(state["chat_history"])
            #print(f"[PLANNING] Activity context: {activity_context}")

            # Preparar parâmetros do prompt com informações adicionais
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

            # Gerar o plano usando o modelo
            #print("[PLANNING] Generating response plan")
            response = model.invoke(prompt.format(**params))

            #print(f"[PLANNING] Model response: {response.content}")

            # Processar e validar a resposta
            plan = process_model_response(response.content)
            #print(f"[PLANNING] Generated valid plan")

            # Ajustar o plano baseado no contexto de atividades
            if activity_context["is_activity_response"]:
                #print("[PLANNING] Adjusting plan for activity response")
                plan = adjust_plan_for_activity(plan, activity_context)

            # Validar o plano gerado
            validate_plan_structure(plan)

            # Atualizar o estado
            new_state = state.copy()
            new_state["current_plan"] = json.dumps(plan)
            new_state["next_step"] = determine_next_step(plan)
            new_state["activity_context"] = activity_context

            #print(f"[PLANNING] Plan generation completed successfully")
            return new_state

        except Exception as e:
            #print(f"[PLANNING] Error in plan generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return handle_plan_generation_error(state, str(e))

    def analyze_activity_history(chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """Analisa o histórico para identificar contexto de atividades."""
        try:
            # Filtrar últimas 5 mensagens para análise
            recent_messages = chat_history[-5:]
            is_activity_response = False
            last_activity = None
            activity_reference = None

            for msg in reversed(recent_messages):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, str):
                        # Verificar se é uma atividade proposta
                        if "atividade_pratica" in content.lower() or "exercício" in content.lower():
                            last_activity = content
                            break

            # Verificar se a última mensagem do usuário é uma resposta
            if recent_messages and isinstance(recent_messages[-1], HumanMessage):
                last_user_msg = recent_messages[-1].content.lower()
                if last_activity:
                    # Análise simples de similaridade ou referência à atividade
                    is_activity_response = True
                    activity_reference = last_activity

            return {
                "is_activity_response": is_activity_response,
                "last_activity": last_activity,
                "activity_reference": activity_reference
            }

        except Exception as e:
            #print(f"[PLANNING] Error analyzing activity history: {e}")
            return {
                "is_activity_response": False,
                "last_activity": None,
                "activity_reference": None
            }

    def process_model_response(content: str) -> Dict[str, Any]:
        """Processa e valida a resposta do modelo."""
        try:
            # Limpar o conteúdo
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            # Converter para JSON
            plan = json.loads(cleaned_content)
            #print(f"[PLANNING] Processed model response: {plan}")
            return plan

        except json.JSONDecodeError as e:
            #print(f"[PLANNING] Error parsing model response: {e}")
            raise ValueError("Invalid JSON in model response")

    def adjust_plan_for_activity(plan: Dict[str, Any], activity_context: Dict[str, Any]) -> Dict[str, Any]:
        """Ajusta o plano quando é uma resposta a atividade."""
        if activity_context["is_activity_response"]:
            plan["tipo_entrada"] = "resposta_atividade"
            plan["analise_resposta"] = {
                "atividade_referencia": activity_context["activity_reference"],
                "completude": 0,  # Será preenchido pela análise
                "precisao": 0,    # Será preenchido pela análise
                "pontos_fortes": [],
                "pontos_melhoria": [],
                "feedback": "Análise pendente"
            }
        return plan

    def validate_plan_structure(plan: Dict[str, Any]) -> None:
        """Valida a estrutura do plano gerado."""
        required_fields = [
            "tipo_entrada",
            "contexto_identificado",
            "alinhamento_plano",
            "estrutura_resposta",
            "nivel_resposta",
            "proxima_acao"
        ]

        missing_fields = [field for field in required_fields if field not in plan]
        if missing_fields:
            raise ValueError(f"Missing required fields in plan: {missing_fields}")

    def determine_next_step(plan: Dict[str, Any]) -> str:
        """Determina o próximo passo baseado no plano."""
        if plan.get("tipo_entrada") == "resposta_atividade":
            return "direct_answer"
        return plan.get("proxima_acao", "retrieval")

    def handle_plan_generation_error(state: AgentState, error_msg: str) -> AgentState:
        """Manipula erros na geração do plano."""
        #print(f"[PLANNING] Handling error: {error_msg}")

        default_plan = {
            "tipo_entrada": "erro",
            "contexto_identificado": f"Erro na geração do plano: {error_msg}",
            "alinhamento_plano": True,
            "estrutura_resposta": [
                {
                    "parte": "Resposta básica",
                    "objetivo": "Fornecer informação solicitada",
                    "abordagem": "Direta e simples"
                }
            ],
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
    - SE houver contexto principal, explique detalhadamente
    - Responda como um tutor educacional sempre orientando o aluno a chegar na resposta e dando proximos passos.

    DIRETRIZES:
    - NUNCA presuma conhecimento prévio sem explicar
    - NUNCA pule a explicação do contexto principal
    - Use linguagem clara e objetiva
    - Mantenha foco educacional

    ATENÇÃO: 
    - Você DEVE explicar o contexto fornecido antes de qualquer outra coisa
    - A resposta deve ser direta ao aluno
    - Mantenha a resposta educacional
    - Incentive a busca da informação pelo aluno
    """

    DIRECT_RESPONSE_PROMPT = """
    ROLE: Tutor educacional

    TASK: Guiar o aluno na compreensão e resolução de questões sem dar respostas diretas.

    INSTRUCTIONS: 
    Leita atentamente o plano de resposta e a mensagem do aluno. Forneça orientações e dicas para ajudar o aluno a alcancar o objetivo do plano de resposta
        - Plano de Resposta:
        {learning_plan}

        - Perfil do Aluno:
        {user_profile}

        - Histórico da Conversa:
        {chat_history}

        - Mensagem do aluno:
        {question}

    ESTRUTURA DA RESPOSTA:
    - Responda como um tutor educacional sempre orientando o aluno a chegar na resposta. Incentive o raciocínio e a busca ativa de soluções.
    - Siga o plano de resposta fornecido e adapte conforme necessário.

    DIRETRIZES:
    - Use linguagem amigavel e acessível
    - Foque em conceitos fundamentais
    - Adapte ao estilo de aprendizagem do aluno
    - Incentive o raciocínio do aluno

    ATENÇÃO:
    - Voce DEVE guiar o aluno sem dar respostas diretas
    - Voce responde diretamente ao aluno
    """

    context_prompt = ChatPromptTemplate.from_template(CONTEXT_BASED_PROMPT)
    direct_prompt = ChatPromptTemplate.from_template(DIRECT_RESPONSE_PROMPT)
    model = ChatOpenAI(model="gpt-4o", temperature=0.2)

    def generate_teaching_response(state: AgentState) -> AgentState:
        #print("\n[NODE:TEACHING] Starting teaching response generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])

        try:
            # Determinar se é resposta baseada em contexto ou direta
            if state.get("next_step") == "direct_answer":
                #print("[NODE:TEACHING] Using direct response prompt")
                #print(f"[NODE:TEACHING] Current plan: {state["current_plan"]}")
                explanation = model.invoke(direct_prompt.format(
                    learning_plan=state["current_plan"],
                    user_profile=state["user_profile"],
                    question=latest_question,
                    chat_history=chat_history
                ))
                image_content = None
                #print(f"[NODE:TEACHING] Direct response: {explanation.content}")

            else:
                #print("[NODE:TEACHING] Using context-based prompt")
                # Processar contextos para resposta baseada em contexto
                if state.get("web_search_results"):
                    source_type = "Resultados de busca web"
                    web_results = state["web_search_results"]
                    primary_context = f"Wikipedia:\n{web_results.get('wikipedia', 'Não disponível')}"
                    secondary_contexts = f"YouTube:\n{web_results.get('youtube', 'Não disponível')}"
                else:
                    contexts = state["extracted_context"]
                    relevance = contexts.get("relevance_analysis", {})

                    context_scores = {
                        "text": relevance.get("text", {}).get("score", 0),
                        "image": relevance.get("image", {}).get("score", 0),
                        "table": relevance.get("table", {}).get("score", 0)
                    }

                    sorted_contexts = sorted(
                        context_scores.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )

                    source_type = "Material de estudo"
                    primary_type = sorted_contexts[0][0]

                    if primary_type == "text":
                        primary_context = f"Texto: {contexts.get('text', '')}"
                    elif primary_type == "image":
                        primary_context = f"Imagem: {contexts.get('image', {}).get('description', '')}"
                    else:
                        primary_context = f"Tabela: {contexts.get('table', {}).get('content', '')}"

                    secondary_contexts_list = []
                    for context_type, score in sorted_contexts[1:]:
                        if score > 0.3:
                            if context_type == "text":
                                secondary_contexts_list.append(f"Texto Complementar: {contexts.get('text', '')}")
                            elif context_type == "image":
                                secondary_contexts_list.append(f"Descrição da Imagem: {contexts.get('image', {}).get('description', '')}")
                            elif context_type == "table":
                                secondary_contexts_list.append(f"Dados da Tabela: {contexts.get('table', {}).get('content', '')}")
                    
                    secondary_contexts = "\n\n".join(secondary_contexts_list)
                #print(f"[NODE:TEACHING] Current CONTEXT plan: {state["current_plan"]}")
                explanation = model.invoke(context_prompt.format(
                    learning_plan=state["current_plan"],
                    user_profile=state["user_profile"],
                    source_type=source_type,
                    primary_context=primary_context,
                    secondary_contexts=secondary_contexts,
                    question=latest_question,
                    chat_history=chat_history
                ))
                #print(f"[NODE:TEACHING] Context-based response: {explanation.content}")

                # Processar imagem se disponível e relevante
                image_content = None
                if (state.get("extracted_context") and 
                    state["extracted_context"].get("image", {}).get("type") == "image" and
                    state["extracted_context"].get("image", {}).get("image_bytes") and
                    context_scores.get("image", 0) > 0.3):
                    image_content = state["extracted_context"]["image"]["image_bytes"]

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
            #print(f"[NODE:TEACHING] Error generating response: {str(e)}")
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
    5. Avalie o progresso com base nas atividades e conceitos entregue pelo ALUNO. Voce pode considerar o campo contexto_identificado do plano de execução atual.

    IMPORTANTE: Retorne APENAS um JSON válido no seguinte formato EXATO:
        "comprehension_level": "alto|medio|baixo",
        "progress_made": true|false,
        "progress_increment": number,
        "step_completed": boolean,
        "reasoning": "string"
    """

    prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def analyze_progress(state: AgentState) -> AgentState:
        #print("\n[NODE:PROGRESS_ANALYST] Starting progress analysis")

        try:
            # Extrai informações necessárias do estado
            messages = state['messages']
            chat_history = state['chat_history']
            current_plan = state['current_plan']
            session_id = state['session_id']

            # Obtém informações atualizadas de progresso
            study_progress = await progress_manager.get_study_progress(session_id)
            if not study_progress:
                #print("[PROGRESS_ANALYST] No progress data found")
                return state

            # Identifica a etapa atual
            plano_execucao = study_progress['plano_execucao']
            current_step = None
            step_index = 0

            # Encontra a primeira etapa não concluída
            for idx, step in enumerate(plano_execucao):
                if step['progresso'] < 100:
                    current_step = step
                    step_index = idx
                    break

            if not current_step:
                #print("[PROGRESS_ANALYST] All steps completed")
                return state

            # Formata o histórico do chat para análise
            formatted_chat_history = format_chat_history(chat_history, max_messages=5)

            #print(f"[PROGRESS_ANALYST] current_plan: {current_plan}")

            # Obtém análise do modelo
            analysis_response = model.invoke(prompt.format(
                chat_history=formatted_chat_history,
                current_plan=current_plan,
                step_title=current_step['titulo'],
                step_description=current_step['descricao'],
                current_progress=current_step['progresso']
            ))

            # Processa a resposta
            cleaned_content = analysis_response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            analysis = json.loads(cleaned_content)
            #print(f"[PROGRESS_ANALYST] Analysis results: {analysis}")

            # Atualiza o progresso apenas se houve avanço
            if analysis["progress_made"]:
                current_progress = current_step['progresso']
                new_progress = min(
                    current_progress + analysis["progress_increment"], 
                    100
                )

                # Se a análise indica que a etapa foi concluída, força 100%
                if analysis["step_completed"]:
                    new_progress = 100

                #print(f"[PROGRESS_ANALYST] Updating progress - Current: {current_progress}%, New: {new_progress}%")

                # Atualiza o progresso no banco de dados
                update_success = await progress_manager.update_step_progress(
                    session_id,
                    step_index,
                    new_progress
                )

                if update_success:
                    # Obtém o resumo atualizado do estudo
                    study_summary = await progress_manager.get_study_summary(session_id)

                    # Atualiza o estado com as novas informações
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
                    #print("[PROGRESS_ANALYST] Failed to update progress")
                    return state

            #print("[PROGRESS_ANALYST] No progress update needed")
            return state

        except Exception as e:
            #print(f"[PROGRESS_ANALYST] Error in progress analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return state

    return analyze_progress


def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    #print("\n[PLAN] Identifying current execution step")

    if not plano_execucao:
        raise ValueError("[PLAN] Plano de execução vazio")

    # Adicionar validação do progresso
    for step in plano_execucao:
        if "progresso" not in step:
            step["progresso"] = 0
        else:
            # Garantir que o progresso é um número entre 0 e 100
            step["progresso"] = min(max(float(step["progresso"]), 0), 100)

    # Primeiro tenta encontrar uma etapa não concluída
    for step in plano_execucao:
        current_progress = step["progresso"]
        if current_progress < 100:
            #print(f"[PLAN] Found incomplete step: {step['titulo']} (Progress: {current_progress}%)")
            return ExecutionStep(
                titulo=step["titulo"],
                duracao=step["duracao"],
                descricao=step["descricao"],
                conteudo=step["conteudo"],
                recursos=step["recursos"],
                atividade=step["atividade"],
                progresso=current_progress
            )

    # Se todas as etapas estiverem concluídas, retorna a última etapa
    last_step = plano_execucao[-1]
    #print(f"[PLAN] All steps completed. Using last step: {last_step['titulo']} (Progress: {last_step['progresso']}%)")
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
    current_iterations = state.get("iteration_count", 0)

    #print(f"\n[WORKFLOW] Checking continuation - Current iterations: {current_iterations}")
    if current_iterations >= MAX_ITERATIONS:
        #print("[WORKFLOW] Max iterations reached, ending workflow")
        return "end"

    #print("[WORKFLOW] Continuing to next iteration")
    return "continue"

def route_after_evaluation(state: AgentState) -> str:
    needs_retrieval = state.get("needs_retrieval", True)
    #print(f"\n[WORKFLOW] Routing after evaluation - Needs retrieval: {needs_retrieval}")
    return "retrieve_context" if needs_retrieval else "direct_answer"


class WebSearchTools:
    def __init__(self):
        #print("[WEBSEARCH] Initializing WebSearchTools")
        pass

    def search_youtube(self, query: str) -> str:
        """
        Realiza uma pesquisa no YouTube e retorna o link do vídeo mais relevante.
        """
        try:
            #print(f"[WEBSEARCH] Searching YouTube for: {query}")
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

                #print(f"[WEBSEARCH] Found {len(videos_info)} videos")
                return response
            else:
                #print("[WEBSEARCH] No videos found")
                return "Nenhum vídeo encontrado."
        except Exception as e:
            #print(f"[WEBSEARCH] Error searching YouTube: {str(e)}")
            return "Ocorreu um erro ao buscar no YouTube."

    def search_wikipedia(self, query: str) -> str:
        """
        Realiza uma pesquisa no Wikipedia e retorna o resumo da página.
        """
        try:
            #print(f"[WEBSEARCH] Searching Wikipedia for: {query}")
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
                #print(f"[WEBSEARCH] Found Wikipedia article: {page.title}")
                return summary
            else:
                #print("[WEBSEARCH] No Wikipedia article found")
                return "Página não encontrada na Wikipedia."
        except Exception as e:
            #print(f"[WEBSEARCH] Error searching Wikipedia: {str(e)}")
            return "Ocorreu um erro ao buscar na Wikipedia."

def route_after_planning(state: AgentState) -> str:
    """
    Determina o próximo nó após o planejamento com base no plano gerado.
    """
    next_step = state.get("next_step", "retrieval")
    #print(f"[ROUTING] Routing after planning: {next_step}")

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
        #print("\n[NODE:WEBSEARCH] Starting web search")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])

        try:
            # Otimizar a query
            #print(f"[WEBSEARCH] Optimizing query: {latest_question}")
            optimized_query = model.invoke(query_prompt.format(
                question=latest_question,
                chat_history=chat_history
            )).content.strip()
            #print(f"[WEBSEARCH] Optimized query: {optimized_query}")

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

            #print("[WEBSEARCH] Search completed successfully")
            return new_state

        except Exception as e:
            #print(f"[WEBSEARCH] Error during web search: {str(e)}")
            error_message = "Desculpe, encontrei um erro ao buscar os recursos. Por favor, tente novamente."
            
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [
                AIMessage(content=error_message)
            ]
            return new_state

    return web_search

class TutorWorkflow:
    def __init__(
        self,
        qdrant_handler,
        student_email: str,
        disciplina: str,
        session_id: str,
        image_collection
    ):
        #print(f"\n[WORKFLOW] Initializing TutorWorkflow")

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

        # Adiciona nós sem gerenciamento de progresso redundante
        workflow.add_node("generate_plan", create_answer_plan_node())
        workflow.add_node("route_after_plan", route_after_plan_generation())
        workflow.add_node("retrieve_context", create_retrieval_node(self.tools))
        workflow.add_node("web_search", create_websearch_node(self.web_tools))
        workflow.add_node("teach", create_teaching_node())
        workflow.add_node("progress_analyst", create_progress_analyst_node(self.progress_manager))

        # Adiciona edges
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
        #print("[WORKFLOW] Workflow graph created successfully")
        return workflow.compile()

    async def handle_progress_update(
        self,
        session_id: str,
        step_index: int,
        new_progress: int
    ) -> Optional[Dict[str, Any]]:
        """
        Função utilitária para atualizar o progresso de forma segura.
        """
        try:
            success = await self.progress_manager.update_step_progress(
                session_id,
                step_index,
                new_progress
            )

            if success:
                return await self.progress_manager.get_study_summary(session_id)
            return None
        except Exception as e:
            #print(f"[PROGRESS] Error in handle_progress_update: {e}")
            return None

    def create_planning_node_with_progress(self):
        """Cria um nó de planejamento que apenas carrega o progresso atual."""
        planning_node = create_answer_plan_node()
        progress_manager = self.progress_manager

        async def planning_with_progress(state: AgentState) -> AgentState:
            try:
                # Apenas recupera o progresso atual antes do planejamento
                current_progress = await progress_manager.get_study_progress(self.session_id)

                # Atualiza o estado com o progresso atual sem modificá-lo
                if current_progress:
                    state["current_progress"] = current_progress
                
                # Executa o planejamento original
                new_state = planning_node(state)
                return new_state

            except Exception as e:
                #print(f"[PLANNING] Error in planning with progress: {e}")
                import traceback
                traceback.print_exc()
                return state

        return planning_with_progress

    def create_teaching_node_with_progress(self):
        """Cria um nó de ensino sem gerenciamento de progresso."""
        teaching_node = create_teaching_node()

        async def teaching_with_progress(state: AgentState) -> AgentState:
            try:
                # Executa apenas o ensino sem modificar progresso
                new_state = teaching_node(state)
                return new_state

            except Exception as e:
                #print(f"[PROGRESS] Error in teaching with progress: {e}")
                import traceback
                traceback.print_exc()
                return state

        return teaching_with_progress

    async def invoke(
        self, 
        query: str, 
        student_profile: dict, 
        current_plan=None, 
        chat_history=None
    ) -> dict:
        """
        Optimized workflow invocation with performance tracking and caching
        """
        start_time = time.time()
        query_hash = hash(f"{query}:{self.session_id}")
        cache_key = f"workflow:{query_hash}"
        
        # Use a simple in-memory cache for repeated identical queries
        if hasattr(self, '_result_cache') and cache_key in self._result_cache:
            return self._result_cache[cache_key]
        
        try:
            # Concurrent operations for preparation
            progress_task = asyncio.create_task(
                self.progress_manager.sync_progress_state(self.session_id)
            )
            current_progress_task = asyncio.create_task(
                self.progress_manager.get_study_progress(self.session_id)
            )
            
            # Normalize and slice chat history efficiently
            if chat_history is None:
                recent_history = []
            elif not isinstance(chat_history, list):
                recent_history = list(chat_history)[-10:]
            else:
                recent_history = chat_history[-10:]
                
            # Wait for concurrent tasks to complete
            await progress_task
            current_progress = await current_progress_task
            
            # Use faster immutable state initialization
            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                current_plan=current_plan if current_plan else "",
                user_profile=student_profile,
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

            # Execute workflow with optimized state
            result = await self.workflow.ainvoke(initial_state)
            
            # Get study summary concurrently with result processing
            study_summary_task = asyncio.create_task(
                self.progress_manager.get_study_summary(self.session_id)
            )
            
            # Process messages for efficiency
            messages = result.get("messages", [])
            
            # Get study summary
            study_summary = await study_summary_task
            
            # Build optimized result dictionary
            final_result = {
                "messages": messages,
                "final_plan": result.get("current_plan", ""),
                "chat_history": result.get("chat_history", recent_history),
                "study_progress": study_summary
            }

            # Add error information if present
            if "error" in result:
                final_result["error"] = result["error"]
                
            # Cache the result for future reuse
            if not hasattr(self, '_result_cache'):
                self._result_cache = {}
            self._result_cache[cache_key] = final_result
            
            # Clear old cache entries to prevent memory leaks
            if len(self._result_cache) > 100:  # Limit cache size
                # Remove oldest entries
                old_keys = list(self._result_cache.keys())[:-50]  # Keep the 50 newest
                for k in old_keys:
                    del self._result_cache[k]
                
            return final_result

        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Create compact error response
            error_response = {
                "error": str(e),
                "messages": [AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")],
                "chat_history": recent_history
            }

            # Try to get study progress even in error cases
            try:
                error_response["study_progress"] = await self.progress_manager.get_study_summary(self.session_id)
            except Exception:
                pass  # Ignore errors in error handling

            return error_response
        finally:
            elapsed_time = time.time() - start_time
            print(f"[WORKFLOW] Execution completed in {elapsed_time:.2f}s")