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
from logg import logger
from tavily import TavilyClient
from utils import TAVILY_API_KEY

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

def format_chat_history(messages: List[BaseMessage], max_messages: int = 5) -> str:
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

        Atenção: Foque em avaliar a relevância dos contextos para a pergunta original.
        Se não responder a pergunta, não é relevante.

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

        context_types = []
        if text_context:
            context_types.append("text")
        if image_context and image_context.get("description"):
            context_types.append("image")
        if table_context and table_context.get("content"):
            context_types.append("table")

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
            
            if results:
                logger.info(f"[RETRIEVAL] Retrieved {len(results)} text context documents.")

                content = "\n".join([doc.page_content for doc in results])
                return content
            else:
                logger.warning("[RETRIEVAL] No text context found.")
                return ""
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error in text retrieval: {e}")
            import traceback
            traceback.print_exc()
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

            logger.info(f"[RETRIEVAL] Image search results count: {len(results) if results else 0}")

            if not results:
                logger.info("[RETRIEVAL] No image results found")
                return {"type": "image", "content": None, "description": ""}

            image_result = results[0]
            logger.info(f"[RETRIEVAL] Found image with metadata: {image_result.metadata}")

            image_uuid = image_result.metadata.get("image_uuid")
            if not image_uuid:
                logger.warning("[RETRIEVAL] Image found but missing UUID in metadata")
                return {"type": "image", "content": None, "description": ""}

            result = await self.retrieve_image_and_description(image_uuid)

            if result.get("type") == "error":
                logger.error(f"[RETRIEVAL] Error retrieving image: {result.get('message')}")
                return {"type": "image", "content": None, "description": ""}

            desc_length = len(result.get("description", ""))
            logger.info(f"[RETRIEVAL] Successfully retrieved image, description length: {desc_length}")
            return result

        except Exception as e:
            logger.error(f"[RETRIEVAL] Error in image retrieval: {e}")
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
                logger.info("[RETRIEVAL] No table results found")
                return {"type": "table", "content": None}

            table_result = results[0]
            logger.info(f"[RETRIEVAL] Found table with length: {len(table_result.page_content)}")

            return {
                "type": "table",
                "content": table_result.page_content,
                "metadata": table_result.metadata
            }
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error in table retrieval: {e}")
            return {"type": "table", "content": None}

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        """
        Recupera a imagem e sua descrição de forma assíncrona.
        """
        try:
            image_data = await self.image_collection.find_one({"_id": image_uuid})

            if not image_data:
                logger.warning(f"[RETRIEVAL] Imagem não encontrada na coleção. UUID: {image_uuid}")
                return {"type": "error", "message": "Imagem não encontrada"}

            image_bytes = image_data.get("image_data")

            if not image_bytes:
                logger.warning("[RETRIEVAL] Dados da imagem ausentes no documento")
                return {"type": "error", "message": "Dados da imagem ausentes"}

            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                logger.warning(f"[RETRIEVAL] Formato de imagem não suportado: {type(image_bytes)}")
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

            logger.info(f"[RETRIEVAL] Descrição da imagem - resultados encontrados: {len(results) if results else 0}")

            if not results:
                logger.warning(f"[RETRIEVAL] Descrição da imagem não encontrada para UUID: {image_uuid}")
                return {"type": "error", "message": "Descrição da imagem não encontrada"}

            description = results[0].page_content
            logger.info(f"[RETRIEVAL] Imagem e descrição recuperadas com sucesso. Tamanho da descrição: {len(description)}")

            return {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": description
            }
        except Exception as e:
            logger.error(f"[RETRIEVAL] Erro ao recuperar imagem: {e}")
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

4. FEEDBACK E PRÓXIMOS PASSOS
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
6. Você deve rotear para o próximo passo após a geração do plano. Se você achar necessidade de fornecer um material como contexto,
 usar imagens ou tabelas, você deve fazer usar o campo "next_step": "retrieval".
7. Se achar necessário fazer uma busca na web, você deve usar o campo "next_step": "websearch". (Tente evitar isso, mas se necessário, faça)
8. Se o aluno pedir algum link ou vídeo, você deve usar o campo "next_step": "websearch".


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
    "proxima_acao": "string",
    "nivel_resposta": "basico|intermediario|avancado",
    "next_step": "websearch|retrieval|direct_answer"


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
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    def generate_plan(state: AgentState) -> AgentState:
        #print("\n[NODE:PLANNING] Starting plan generation")

        try:
            # Extrair última mensagem
            latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
            #print(f"[NODE:PLANNING] Processing input: {latest_question}")

            # Processar o plano de execução atual
            try:
                # Check if current_plan is already a dict or a JSON string
                if state["current_plan"] and isinstance(state["current_plan"], str):
                    plano_execucao = json.loads(state["current_plan"])
                elif state["current_plan"] and isinstance(state["current_plan"], dict):
                    plano_execucao = state["current_plan"]
                else:
                    # Handle the case when current_plan is empty or invalid
                    raise ValueError("Empty or invalid execution plan")

                current_step = identify_current_step(plano_execucao["plano_execucao"])
                #print(f"[PLANNING] Current step: {current_step.titulo} ({current_step.progresso}%)")
            except (json.JSONDecodeError, KeyError) as e:
                #print(f"[PLANNING] Error processing execution plan: {e}")
                raise ValueError("Invalid execution plan format")

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
            logger.error(f"[PLANNING] Error in plan generation: {str(e)}")
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
            logger.error(f"[PLANNING] Error analyzing activity history: {e}")
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
        # Verifica se há um valor explícito em next_step
        if "next_step" in plan and plan["next_step"] in ["websearch", "retrieval", "direct_answer"]:
            return plan["next_step"]

        # Caso não tenha ou seja inválido, usa a lógica anterior como fallback
        if plan.get("tipo_entrada") == "resposta_atividade":
            return "direct_answer"
        return "retrieval"

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

    Fonte do contexto:
    {source_type}

    Histórico da Conversa:
    {chat_history}

    Mensagem do aluno:
    {question}

    Contexto Principal (OBRIGATÓRIO EXPLICAR):
    {primary_context}

    Contextos Secundários (OBRIGATÓRIO EXPLICAR):
    {secondary_contexts}

    ESTRUTURA DA RESPOSTA:
    - Escolha uma maneira explicar o contexto recuperado.
    - Monte sua resposta utilizando o contexto principal e secundário.
    - Se o contexto for uma imagem, explique o que é e como funciona.
    - Responda como um tutor educacional sempre orientando o aluno a chegar na resposta. Incentive o raciocínio e a busca ativa de soluções.
    - Siga o plano de resposta fornecido e adapte conforme necessário.

    DIRETRIZES:
    - Use linguagem amigavel e acessível
    - Foque em conceitos fundamentais
    - Adapte ao estilo de aprendizagem do aluno
    - Incentive o raciocínio do aluno
    - Forneça curiosidades e dicas adicionais com o objetivo de conectar o assunto atual com outros assuntos (Se achar necessário)

    ATENÇÃO:
    - Você DEVE explicar o contexto fornecido utilizando a melhor abordagem educacional
    - Voce DEVE guiar o aluno sem dar respostas diretas
    - A resposta deve ser muito detalhada, abordando conceitos e passos necessários
    - Voce responde diretamente ao aluno
    - Você pode mostrar imagens ou tabelas, mas deve explicar o que são e como funcionam. Essas imagens estão presentes no contexto.
    """

    DIRECT_RESPONSE_PROMPT = """
    ROLE: Tutor educacional

    TASK: Guiar o aluno na compreensão e resolução de questões sem dar respostas diretas. As respostas devem ser detalhadas e educacionais.

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
    - Forneça curiosidades e dicas adicionais com o objetivo de conectar o assunto atual com outros assuntos (Se achar necessário)

    ATENÇÃO:
    - Voce DEVE guiar o aluno sem dar respostas diretas
    - A resposta deve ser muito detalhada, abordando conceitos e passos necessários
    - Voce responde diretamente ao aluno
    """

    context_prompt = ChatPromptTemplate.from_template(CONTEXT_BASED_PROMPT)
    direct_prompt = ChatPromptTemplate.from_template(DIRECT_RESPONSE_PROMPT)
    # Configuração do modelo com streaming ativado
    model = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)

    async def generate_teaching_response(state: AgentState):
        """Gera resposta em formato de streaming usando chunks"""
        import time
        start_time = time.time()

        #print("\n[NODE:TEACHING] Starting teaching response generation with streaming")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])

        # Primeiro chunk - indicando processamento
        yield {"type": "processing", "content": "Tudo pronto para responder..."}

        try:
            full_response = ""
            image_content = None

            # Determinar se é resposta baseada em contexto ou direta
            if state.get("next_step") == "direct_answer":
                #print("[NODE:TEACHING] Using direct response prompt with streaming")
                prompt_params = {
                    "learning_plan": state["current_plan"],
                    "user_profile": state["user_profile"],
                    "question": latest_question,
                    "chat_history": chat_history
                }
                print("[NODE:TEACHING] Prompt params:", prompt_params)
                stream = model.astream(direct_prompt.format(**prompt_params))

            else:
                #print("[NODE:TEACHING] Using context-based prompt with streaming")
                # Processar contextos para resposta baseada em contexto
                if state.get("web_search_results"):
                    source_type = "Resultados de busca web"
                    web_results = state["web_search_results"]
                    primary_context = f"Resultados da busca web Tavily:\n{web_results.get('tavily_results', 'Não disponível')}"
                    secondary_contexts = ""
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
                    primary_type = sorted_contexts[0][0] if sorted_contexts else "text"

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

                # Verificar se há imagem relevante antes de iniciar o streaming
                if (state.get("extracted_context") and 
                    state["extracted_context"].get("image", {}).get("type") == "image" and
                    state["extracted_context"].get("image", {}).get("image_bytes") and
                    context_scores.get("image", 0) > 0.3):
                    image_content = state["extracted_context"]["image"]["image_bytes"]

                prompt_params = {
                    "learning_plan": state["current_plan"],
                    "user_profile": state["user_profile"],
                    "source_type": source_type,
                    "primary_context": primary_context,
                    "secondary_contexts": secondary_contexts,
                    "question": latest_question,
                    "chat_history": chat_history
                }

                stream = model.astream(context_prompt.format(**prompt_params))

            # Se tiver imagem, enviar um chunk com a imagem primeiro
            if image_content:
                base64_image = base64.b64encode(image_content).decode('utf-8')
                yield {
                    "type": "image", 
                    "content": "", 
                    "image": f"data:image/jpeg;base64,{base64_image}"
                }

            # Processar os chunks do streaming
            async for chunk in stream:
                if chunk.content:
                    full_response += chunk.content
                    yield {"type": "chunk", "content": chunk.content}

            # Atualizar estado após o streaming completo
            if image_content:
                base64_image = base64.b64encode(image_content).decode('utf-8')
                response_content = {
                    "type": "multimodal",
                    "content": full_response,
                    "image": f"data:image/jpeg;base64,{base64_image}"
                }
                response = AIMessage(content=json.dumps(response_content))
            else:
                response = AIMessage(content=full_response)

            history_message = AIMessage(content=full_response)
            #print(f"[NODE:TEACHING] Full response: {response.content}")
            # Update state
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                history_message
            ]

            # Enviar mensagem de conclusão com o tempo de processamento
            processing_time = time.time() - start_time
            yield {"type": "complete", "content": f"Resposta completa em {processing_time:.2f}s"}

            # Atualizar estado após conclusão
            state.update(new_state)

        except Exception as e:
            logger.error(f"[NODE:TEACHING] Error generating streaming response: {str(e)}")

            yield {"type": "error", "content": f"Ocorreu um erro ao processar sua mensagem: {str(e)}"}

            error_message = "Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente."
            response = AIMessage(content=error_message)
            history_message = response

            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                history_message
            ]
            state.update(new_state)

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
            logger.error(f"[PROGRESS_ANALYST] Error in progress analysis: {str(e)}")
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
        logger.info("[WEBSEARCH] Inicializando WebSearchTools com Tavily")

        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def search_web(self, query: str) -> str:
        """
        Realiza uma pesquisa na web usando a API Tavily e retorna os resultados formatados.
        """
        try:
            logger.info(f"[WEBSEARCH] Tavily search: {query}")

            # Realizar a busca com Tavily
            search_results = self.client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_answer=True,
                include_images=False
            )

            # Extrair resultados
            results = search_results.get("results", [])
            answer = search_results.get("answer", "")

            if not results:
                logger.warning("[WEBSEARCH] Nenhum resultado encontrado pelo Tavily")
                return "Não foram encontrados resultados relevantes para sua pesquisa."

            # Formatar resultados
            formatted_results = f"Resumo Tavily: {answer}\n\nRecursos encontrados:\n\n"

            for i, result in enumerate(results, 1):
                title = result.get("title", "Sem título")
                url = result.get("url", "")
                content = result.get("content", "Sem conteúdo")[:300] + "..."

                formatted_results += (
                    f"{i}. {title}\n"
                    f"   URL: {url}\n"
                    f"   Resumo: {content}\n\n"
                )

            logger.info(f"[WEBSEARCH] Encontrados {len(results)} resultados pelo Tavily")
            return formatted_results

        except Exception as e:
            logger.error(f"[WEBSEARCH] Erro na busca Tavily: {str(e)}")
            return f"Ocorreu um erro ao realizar a busca: {str(e)}"

def route_after_planning(state: AgentState):
    """
    Determina o próximo nó após o planejamento com base no next_step definido no plano gerado.
    """
    #print("\n[ROUTING] Determining next node after planning")
    next_step = state.get("next_step", "retrieval")
    #print(f"[ROUTING] Routing after planning: {next_step}")
    #print()
    #print("---------------------------------------------")
    #print("next_step: ", next_step)
    #print("---------------------------------------------")
    #print()
    if next_step == "websearch":
        return "web_search"
    elif next_step == "retrieval":
        return "retrieve_context"
    else:
        return "direct_answer"

def create_websearch_node(web_tools: WebSearchTools):
    QUERY_GENERATOR_PROMPT = """Você é um especialista em gerar consultas de pesquisa eficazes.

    Plano de estudo:
    {plan}

    Pergunta do aluno:
    {question}

    TAREFA:
    Gere 4 consultas de pesquisa diferentes e eficazes baseadas no plano e na pergunta.

    DIRETRIZES:
    - Cada consulta deve ter no máximo 80 caracteres
    - Use termos técnicos precisos, evite termos genéricos
    - Inclua palavras-chave educacionais ("tutorial", "explicação", "conceito")
    - Varie entre consultas específicas e mais amplas

    Retorne EXATAMENTE neste formato:
    query1: [primeira consulta]
    query2: [segunda consulta]
    query3: [terceira consulta]
    query4: [quarta consulta]
    """

    query_prompt = ChatPromptTemplate.from_template(QUERY_GENERATOR_PROMPT)
    query_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    async def web_search(state: AgentState) -> AgentState:
        logger.info("[NODE:WEBSEARCH] Iniciando busca web com Tavily")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content

        try:
            # Extrair plano do estado
            current_plan = state.get("current_plan", "")

            # Gerar múltiplas consultas otimizadas usando GPT-4o-mini
            logger.info("[WEBSEARCH] Gerando consultas otimizadas com GPT-4o-mini")
            query_response = query_model.invoke(query_prompt.format(
                plan=current_plan,
                question=latest_question
            ))

            # Extrair as 4 consultas do formato especificado
            query_content = query_response.content.strip()
            logger.info(f"[WEBSEARCH] Resposta do gerador de consultas: {query_content}")

            # Processar as consultas geradas
            queries = []
            for line in query_content.split('\n'):
                if line.startswith('query') and ':' in line:
                    query = line.split(':', 1)[1].strip()
                    if query and len(query) > 0:
                        queries.append(query)

            # Garantir pelo menos uma consulta válida
            if not queries:
                logger.warning("[WEBSEARCH] Nenhuma consulta válida gerada, usando pergunta original")
                queries = [latest_question]

            logger.info(f"[WEBSEARCH] Consultas geradas: {queries}")

            # Buscar usando Tavily com a primeira consulta (mais específica)
            best_query = queries[0]
            logger.info(f"[WEBSEARCH] Consultando Tavily com query principal: {best_query}")
            search_results = web_tools.search_web(best_query)

            # Estruturar contexto para o nó de teaching
            extracted_context = {
                "text": search_results,
                "image": {"type": "image", "content": None, "description": ""},
                "table": {"type": "table", "content": None},
                "relevance_analysis": {
                    "text": {"score": 1.0, "reason": "Informação obtida da web via Tavily"},
                    "image": {"score": 0.0, "reason": "Nenhuma imagem disponível"},
                    "table": {"score": 0.0, "reason": "Nenhuma tabela disponível"},
                    "recommended_context": "text"
                }
            }

            # Atualizar estado
            new_state = state.copy()
            new_state["web_search_results"] = {
                "tavily_results": search_results,
                "original_query": latest_question,
                "generated_queries": queries,
                "selected_query": best_query
            }
            new_state["extracted_context"] = extracted_context

            logger.info("[WEBSEARCH] Busca web concluída com sucesso")
            return new_state

        except Exception as e:
            logger.error(f"[WEBSEARCH] Erro durante a busca web: {str(e)}")
            error_message = "Erro ao buscar informações na web."

            # Criar um estado de erro que será processado pelo teaching_node
            new_state = state.copy()
            new_state["web_search_results"] = {
                "error": error_message,
                "original_query": latest_question
            }
            new_state["extracted_context"] = {
                "text": f"Ocorreu um erro ao buscar informações: {str(e)}",
                "image": {"type": "image", "content": None, "description": ""},
                "table": {"type": "table", "content": None},
                "relevance_analysis": {
                    "text": {"score": 1.0, "reason": "Erro na busca web"},
                    "image": {"score": 0.0, "reason": ""},
                    "table": {"score": 0.0, "reason": ""},
                    "recommended_context": "text"
                }
            }
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

    def create_workflow(self):
        workflow = Graph()

        # Adiciona nós sem gerenciamento de progresso redundante
        workflow.add_node("generate_plan", create_answer_plan_node())
        workflow.add_node("retrieve_context", create_retrieval_node(self.tools))
        workflow.add_node("web_search", create_websearch_node(self.web_tools))
        # O nó de teaching agora retorna um gerador de chunks
        self.teaching_node = create_teaching_node()
        workflow.add_node("teach", self.teaching_node)
        workflow.add_node("progress_analyst", create_progress_analyst_node(self.progress_manager))

        # Adiciona edges diretamente do generate_plan para os próximos nós
        workflow.add_conditional_edges(
            "generate_plan",
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
                logger.error(f"[PLANNING] Error in planning with progress: {e}")
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
                logger.error(f"[PROGRESS] Error in teaching with progress: {e}")
                return state

        return teaching_with_progress

    async def invoke(
        self, 
        query: str, 
        student_profile: dict, 
        current_plan=None, 
        chat_history=None,
        stream=False
    ):
        """
        Invoca o workflow de tutoria.
        
        Parâmetros:
            query: Pergunta do usuário
            student_profile: Perfil do estudante
            current_plan: Plano atual (opcional)
            chat_history: Histórico da conversa (opcional)
            stream: Quando True, retorna chunks de resposta via streaming
            
        Retorno:
            Se stream=False: retorna o resultado completo como dict
            Se stream=True: retorna um gerador que produz chunks de resposta
        """
        start_time = time.time()
        logger.info(f"[TUTOR_INVOKE] Usuário={self.student_email} | sessão={self.session_id} | disciplina={self.disciplina}")
        # print(f"\n[WORKFLOW] Starting workflow invocation")
        # print(f"[WORKFLOW] Query: {query}")

        # Função interna para processamento de streaming
        async def stream_response(state):
            # Executa o fluxo até o nó de ensino
            try:
                # print(f"\n[WORKFLOW] Streaming response generation")
                # Vai do nó inicial até o nó de teaching
                interim_result = None
                logger.info(f"[TUTOR_STREAM_START] Usuário={self.student_email} | sessão={self.session_id}")

                plan_node = create_answer_plan_node()
                # print(f"\n[WORKFLOW] Generating plan")
                plan_state = plan_node(state)  # Chamada não-async
                # print(f"[WORKFLOW] Plan state: {plan_state}")
                # print(f"\n[WORKFLOW] Plan generated")
                next_step = route_after_planning(plan_state)
                # print(f"\n[WORKFLOW] Next step after planning: {next_step}")
                if next_step == "retrieve_context":
                    yield {"type": "processing", "content": "Buscando conteúdos interessantes..."}
                    # print(f"\n[WORKFLOW] Retrieval context")
                    logger.info(f"[TUTOR_RETRIEVAL] Usuário={self.student_email} | sessão={self.session_id} | Buscando contexto")
                    retrieve_node = create_retrieval_node(self.tools)
                    interim_result = await retrieve_node(plan_state)  # Este é async
                elif next_step == "web_search":
                    yield {"type": "processing", "content": "Pesquisando na web para encontrar as melhores informações..."}
                    logger.info(f"[TUTOR_WEBSEARCH] Usuário={self.student_email} | sessão={self.session_id} | Buscando na web com Tavily")
                    web_search_node = create_websearch_node(self.web_tools)
                    interim_result = await web_search_node(plan_state)  # Este é async
                else:
                    logger.info(f"[TUTOR_DIRECT] Usuário={self.student_email} | sessão={self.session_id} | Resposta direta")
                    interim_result = plan_state

                teaching_generator = self.teaching_node(interim_result)

                async for chunk in teaching_generator:
                    yield chunk

                try:
                    progress_node = create_progress_analyst_node(self.progress_manager)
                    progress_state = await progress_node(interim_result)
                    study_summary = await self.progress_manager.get_study_summary(self.session_id)

                    # Formate o resumo para serialização
                    serializable_summary = {}
                    for key, value in study_summary.items():
                        if isinstance(value, datetime):
                            serializable_summary[key] = value.isoformat()
                        else:
                            serializable_summary[key] = value

                    progress_pct = serializable_summary.get('progress_percentage', 0)
                    summary_content = f"Progresso atualizado: {progress_pct:.1f}%"
                    logger.info(f"[TUTOR_PROGRESS] Usuário={self.student_email} | sessão={self.session_id} | progresso={progress_pct:.1f}%")
                    yield {"type": "progress_update", "content": summary_content, "data": serializable_summary}
                except Exception as e:
                    logger.error(f"[TUTOR_ERROR] Usuário={self.student_email} | sessão={self.session_id} | erro={str(e)}")
                    raise e

            except Exception as e:
                logger.error(f"[TUTOR_STREAM_ERROR] Usuário={self.student_email} | sessão={self.session_id} | erro={str(e)}")
                yield {"type": "error", "content": f"Erro na execução do workflow: {str(e)}"}

        try:
            # Validar perfil do usuário
            validated_profile = student_profile
            #print(f"[WORKFLOW] Student profile validated: {validated_profile.get('EstiloAprendizagem', 'Not found')}")
            await self.progress_manager.sync_progress_state(self.session_id)
            # Recuperar progresso atual
            current_progress = await self.progress_manager.get_study_progress(self.session_id)
            ##print(f"[WORKFLOW] Current progress loaded: {current_progress}")

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

            # Se streaming estiver ativado, retorna o gerador de chunks
            if stream:
                return stream_response(initial_state)

            # Comportamento tradicional sem streaming
            #print("[WORKFLOW] Executing workflow")
            result = await self.workflow.astream(initial_state)
            #print("[WORKFLOW] Workflow execution completed successfully")

            # Recupera o resumo atualizado do estudo
            study_summary = await self.progress_manager.get_study_summary(self.session_id)

            # Prepara o resultado final
            final_result = {
                "messages": result["messages"],
                "final_plan": result["current_plan"],
                "chat_history": result["chat_history"],
                "study_progress": study_summary
            }

            # Adiciona informações de debug se necessário
            if "error" in result:
                final_result["error"] = result["error"]

            return final_result

        except Exception as e:
            logger.error(f"[WORKFLOW] Error during workflow execution: {str(e)}")

            if stream:
                # Se for streaming, convertemos a exceção em um gerador que retorna apenas um erro
                async def error_generator():
                    yield {"type": "error", "content": f"Erro na execução do workflow: {str(e)}"}
                return error_generator()

            # Sem streaming, retornamos um objeto de erro
            error_response = {
                "error": f"Erro na execução do workflow: {str(e)}",
                "messages": [AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")],
                "chat_history": recent_history
            }

            # Tenta adicionar o progresso mesmo em caso de erro
            try:
                error_response["study_progress"] = await self.progress_manager.get_study_summary(self.session_id)
            except Exception as progress_error:
                logger.error(f"[WORKFLOW] Error getting progress summary: {progress_error}")
                raise e

            return error_response
        finally:
            if not stream:  # Apenas registramos o tempo para execuções não-streaming
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"[TUTOR_COMPLETE] Usuário={self.student_email} | sessão={self.session_id} | tempo={elapsed_time:.2f}s")
                # print(f"[WORKFLOW] Workflow execution completed in {elapsed_time:.2f} seconds")