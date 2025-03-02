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
    """
    Versão aprimorada de RetrievalTools com caching, redução de consultas e
    reaproveitamento inteligente de contextos.
    """
    
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
        # Inicialização básica
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.image_collection = image_collection
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.state = state
        
        # Cache de consultas e resultados (TTL de 30 minutos para cada item)
        self._context_cache = {}
        self._cache_ttl = 1800  # 30 minutos em segundos
        
        # Contadores para monitoramento
        self._consultas_reutilizadas = 0
        self._consultas_realizadas = 0
        
        # Lista de termos educacionais para enriquecer prompts
        self._variability_terms = [
            "analise", "avalie", "compare", "explique", "defina", "discuta", 
            "contraste", "critique", "sintetize", "relacione", "aplique",
            "argumente", "classifique", "desenvolva", "exemplifique"
        ]
        
        # Prompts otimizados
        self.QUESTION_TRANSFORM_PROMPT = """
        Transforme esta pergunta para melhorar a recuperação de contexto educacional.
        Conecte a nova pergunta com o contexto da conversa anterior, mas mantenha-a específica.
        Torne-a mais precisa e focada nos termos técnicos relevantes.
        
        Histórico: {chat_history}
        Pergunta: {question}
        Termo de variabilidade: {variability_term}
        
        Retorne APENAS a pergunta reescrita, sem explicações.
        """

        self.RELEVANCE_ANALYSIS_PROMPT = """
        Analise a relevância e relevância dos contextos para a pergunta:
        
        Pergunta: {question}
        Contextos: 
        - Texto: {text_context}
        - Imagem: {image_context}
        - Tabela: {table_context}
        
        Retorne um JSON exato:
        {{
            "text": {{"score": X, "reason": "Y"}},
            "image": {{"score": X, "reason": "Y"}},
            "table": {{"score": X, "reason": "Y"}},
            "recommended_context": "text|image|table|combined"
        }}
        """
        
        # Inicializa um contador para rotação de termos de variabilidade
        self._variability_idx = 0

    def _get_next_variability_term(self) -> str:
        """Obtém o próximo termo de variabilidade em rotação"""
        term = self._variability_terms[self._variability_idx]
        self._variability_idx = (self._variability_idx + 1) % len(self._variability_terms)
        return term
    
    def _get_cache_key(self, query_type: str, query: str) -> str:
        """Gera uma chave de cache para consultas"""
        return f"{query_type}:{self.disciplina}:{hash(query)}"
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Verifica se o item do cache ainda é válido"""
        return (time.time() - timestamp) < self._cache_ttl

    def _should_use_vector_search(self, question: str) -> bool:
        """
        Decide inteligentemente se deve usar busca vetorial ou reutilizar contexto
        baseado no histórico de perguntas e padrões de consulta.
        """
        # Se é a primeira pergunta, sempre busca
        if not self.state["chat_history"]:
            return True
            
        # Analisa se a pergunta atual é similar à última pergunta
        last_human_msgs = [m for m in self.state["chat_history"] if isinstance(m, HumanMessage)]
        if not last_human_msgs:
            return True
            
        # Extrai a última pergunta
        last_question = last_human_msgs[-1].content.lower()
        current_question = question.lower()
        
        # Verifica se é uma pergunta de follow-up óbvia
        follow_up_indicators = ["explicar melhor", "poderia detalhar", "não entendi", 
                               "como assim", "por que", "mais sobre", "explique novamente",
                               "o que significa", "como funciona", "pode dar um exemplo"]
                               
        # Se é um follow-up claro, reutiliza o contexto
        if any(indicator in current_question for indicator in follow_up_indicators):
            return False
            
        # Compara similaridade de palavras-chave entre perguntas
        last_words = set(last_question.split())
        current_words = set(current_question.split())
        common_words = last_words.intersection(current_words)
        
        # Se há muitas palavras em comum (continuação do mesmo tópico)
        if len(common_words) >= 3 and len(common_words) / len(current_words) > 0.4:
            return False
            
        # Por padrão, faz busca vetorial
        return True

    async def transform_question(self, question: str) -> str:
        """Transforma a pergunta para melhorar recuperação de contexto"""
        # Otimiza o uso do histórico
        formatted_history = format_chat_history(self.state["chat_history"], max_messages=3)
        
        # Adiciona variabilidade controlada
        variability_term = self._get_next_variability_term()
        
        prompt = ChatPromptTemplate.from_template(self.QUESTION_TRANSFORM_PROMPT)
        response = await self.model.ainvoke(prompt.format(
            chat_history=formatted_history,
            question=question,
            variability_term=variability_term
        ))

        return response.content.strip()

    async def parallel_context_retrieval(self, question: str) -> Dict[str, Any]:
        """Recupera contextos de forma eficiente, com cache e evitando consultas desnecessárias"""
        # Decisão inteligente sobre uso de busca vetorial
        if not self._should_use_vector_search(question):
            # Tenta reutilizar contexto existente se for apropriado
            if "extracted_context" in self.state and self.state["extracted_context"]:
                self._consultas_reutilizadas += 1
                print(f"[OTIMIZAÇÃO] Reutilizando contexto existente. Total reutilizadas: {self._consultas_reutilizadas}")
                return self.state["extracted_context"]
        
        # Transformando a pergunta para busca
        transformed_question = await self.transform_question(question)
        
        # Execute as buscas em paralelo, mas com verificação de cache
        text_context_task = asyncio.create_task(self._get_text_context(transformed_question))
        image_context_task = asyncio.create_task(self._get_image_context(transformed_question))
        table_context_task = asyncio.create_task(self._get_table_context(transformed_question))
        
        # Aguardar todos os resultados
        text_context, image_context, table_context = await asyncio.gather(
            text_context_task, image_context_task, table_context_task
        )
        
        # Analisa a relevância dos contextos recuperados
        relevance_analysis = await self.analyze_context_relevance(
            original_question=question,
            text_context=text_context,
            image_context=image_context,
            table_context=table_context
        )
        
        # Construa o resultado final
        result = {
            "text": text_context,
            "image": image_context,
            "table": table_context,
            "relevance_analysis": relevance_analysis
        }
        
        # Registra o uso para análise
        self._consultas_realizadas += 1
        print(f"[OTIMIZAÇÃO] Consultas realizadas: {self._consultas_realizadas}, reutilizadas: {self._consultas_reutilizadas}")
        
        return result

    async def _get_text_context(self, query: str) -> str:
        """Obtém contexto de texto com cache e processamento em chunks"""
        cache_key = self._get_cache_key("text", query)
        
        # Verifica se existe no cache
        if cache_key in self._context_cache:
            entry = self._context_cache[cache_key]
            if self._is_cache_valid(entry["timestamp"]):
                return entry["data"]
        
        # Caso não esteja em cache ou expirado, busca novamente
        try:
            # Busca mais resultados para ter uma seleção melhor de chunks
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "text"},
                k=5  # Recupera mais resultados para seleção
            )
            
            if not results:
                return ""
                
            # Processa os resultados em chunks
            chunks = []
            for doc in results:
                # Extrai metadados relevantes
                source = doc.metadata.get("source", "")
                chunk_id = doc.metadata.get("chunk_id", "")
                
                # Divide o conteúdo em parágrafos para melhor processamento
                paragraphs = doc.page_content.split("\n")
                
                # Adiciona cabeçalho para cada chunk
                chunk_header = f"[Fonte: {source}]"
                if chunk_id:
                    chunk_header += f" [Parte: {chunk_id}]"
                
                # Seleciona apenas os parágrafos mais relevantes (máx 250 palavras)
                content = ""
                word_count = 0
                for para in paragraphs:
                    para_words = len(para.split())
                    if word_count + para_words <= 250:
                        content += para + "\n"
                        word_count += para_words
                    else:
                        # Adiciona um resumo se o parágrafo for muito longo
                        truncated_words = 250 - word_count
                        truncated_para = " ".join(para.split()[:truncated_words]) + "..."
                        content += truncated_para
                        break
                
                # Formata o chunk com seu cabeçalho
                formatted_chunk = f"{chunk_header}\n{content.strip()}"
                chunks.append(formatted_chunk)
            
            # Reranqueia os chunks por relevância e qualidade
            ranked_chunks = await self._rank_chunks(chunks, query)
            
            # Seleciona os chunks mais relevantes (top 3)
            selected_chunks = ranked_chunks[:3]
            
            # Junta os chunks com separadores claros
            text_context = "\n\n---\n\n".join(selected_chunks)
            
            # Armazena no cache
            self._context_cache[cache_key] = {
                "data": text_context,
                "timestamp": time.time()
            }
            
            return text_context
        except Exception as e:
            print(f"[RETRIEVAL] Erro na recuperação de texto: {e}")
            return ""
    
    async def _rank_chunks(self, chunks: List[str], query: str) -> List[str]:
        """Reordena os chunks por relevância e qualidade"""
        if not chunks:
            return []
            
        try:
            # Prompt para analisar relevância dos chunks
            CHUNK_RANKING_PROMPT = """
            Avalie a relevância de cada chunk de texto em relação à consulta do usuário.
            
            Consulta: {query}
            
            Chunks:
            {chunks}
            
            Para cada chunk, atribua uma pontuação de 0 a 10 baseada na:
            1. Relevância para a consulta (0-5 pontos)
            2. Qualidade da informação (0-3 pontos)
            3. Completude do conteúdo (0-2 pontos)
            
            Retorne apenas uma lista com os índices dos chunks em ordem de pontuação, do mais relevante ao menos relevante.
            Exemplo: [2, 0, 3, 1] (onde 2 é o índice do chunk mais relevante)
            """
            
            # Se tivermos menos de 3 chunks, não precisamos ranquear
            if len(chunks) <= 3:
                return chunks
                
            # Formata os chunks para análise
            formatted_chunks = ""
            for i, chunk in enumerate(chunks):
                formatted_chunks += f"Chunk {i}:\n{chunk}\n\n"
            
            prompt = ChatPromptTemplate.from_template(CHUNK_RANKING_PROMPT)
            
            # Usa um modelo mais simples para economizar
            ranking_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            
            # Obtém o ranking
            response = await ranking_model.ainvoke(prompt.format(
                query=query,
                chunks=formatted_chunks
            ))
            
            # Extrai o ranking da resposta
            response_text = response.content.strip()
            
            # Tenta encontrar uma lista na resposta
            import re
            match = re.search(r'\[([0-9, ]+)\]', response_text)
            
            if match:
                # Extrai e converte a lista de índices
                indices_str = match.group(1)
                try:
                    ranked_indices = [int(idx.strip()) for idx in indices_str.split(',')]
                    # Verifica se os índices são válidos
                    valid_indices = [idx for idx in ranked_indices if 0 <= idx < len(chunks)]
                    # Adiciona índices que não foram mencionados ao final
                    all_indices = set(range(len(chunks)))
                    missing_indices = list(all_indices - set(valid_indices))
                    complete_ranking = valid_indices + missing_indices
                    
                    # Retorna os chunks reordenados
                    return [chunks[idx] for idx in complete_ranking]
                except ValueError:
                    print("[RETRIEVAL] Erro ao processar índices de ranking")
            
            # Fallback para a ordem original
            return chunks
            
        except Exception as e:
            print(f"[RETRIEVAL] Erro ao classificar chunks: {e}")
            return chunks  # Retorna a ordem original em caso de erro

    async def _get_image_context(self, query: str) -> Dict[str, Any]:
        """Obtém contexto de imagem com cache"""
        cache_key = self._get_cache_key("image", query)
        
        # Verifica se existe no cache
        if cache_key in self._context_cache:
            entry = self._context_cache[cache_key]
            if self._is_cache_valid(entry["timestamp"]):
                return entry["data"]
        
        # Caso não esteja em cache ou expirado, busca novamente
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "image"},
                k=1  # Limita a 1 imagem para melhor desempenho
            )
            
            if not results:
                null_result = {"type": "image", "content": None, "description": ""}
                self._context_cache[cache_key] = {
                    "data": null_result,
                    "timestamp": time.time()
                }
                return null_result

            image_uuid = results[0].metadata.get("image_uuid")
            if not image_uuid:
                null_result = {"type": "image", "content": None, "description": ""}
                self._context_cache[cache_key] = {
                    "data": null_result,
                    "timestamp": time.time()
                }
                return null_result

            # Obtém a imagem e descrição
            image_result = await self.retrieve_image_and_description(image_uuid)
            
            # Armazena no cache
            self._context_cache[cache_key] = {
                "data": image_result,
                "timestamp": time.time()
            }
            
            return image_result
        except Exception as e:
            print(f"[RETRIEVAL] Erro na recuperação de imagem: {e}")
            return {"type": "image", "content": None, "description": ""}

    async def _get_table_context(self, query: str) -> Dict[str, Any]:
        """Obtém contexto de tabela com cache e processamento inteligente"""
        cache_key = self._get_cache_key("table", query)
        
        # Verifica se existe no cache
        if cache_key in self._context_cache:
            entry = self._context_cache[cache_key]
            if self._is_cache_valid(entry["timestamp"]):
                return entry["data"]
        
        # Caso não esteja em cache ou expirado, busca novamente
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "table"},
                k=2  # Busca até 2 tabelas
            )

            if not results:
                null_result = {"type": "table", "content": None}
                self._context_cache[cache_key] = {
                    "data": null_result,
                    "timestamp": time.time()
                }
                return null_result
            
            # Processa a tabela para exibição
            processed_table = await self._process_table_content(results[0].page_content, query)
            source = results[0].metadata.get("source", "Tabela")
            
            table_result = {
                "type": "table",
                "content": processed_table,
                "source": source,
                "metadata": results[0].metadata
            }
            
            # Armazena no cache
            self._context_cache[cache_key] = {
                "data": table_result,
                "timestamp": time.time()
            }
            
            return table_result
        except Exception as e:
            print(f"[RETRIEVAL] Erro na recuperação de tabela: {e}")
            return {"type": "table", "content": None}
            
    async def _process_table_content(self, table_content: str, query: str) -> str:
        """Processa o conteúdo da tabela para melhor usabilidade"""
        try:
            # Verifica se a tabela está em um formato reconhecível
            if not table_content or len(table_content) < 10:
                return table_content
                
            # Tenta identificar se é uma tabela CSV, Markdown ou outro formato
            lines = table_content.strip().split("\n")
            
            # Verifica se parece uma tabela markdown
            if len(lines) > 2 and "|" in lines[0] and "-+-" in lines[1]:
                # Parece ser uma tabela markdown
                return self._format_markdown_table(lines, query)
                
            # Verifica se parece uma tabela CSV
            if len(lines) > 1 and "," in lines[0]:
                # Parece ser uma tabela CSV
                return self._format_csv_table(lines, query)
                
            # Para outros formatos, tenta fazer uma formatação básica
            # Divide em linhas e trunca se for muito grande
            if len(lines) > 15:
                header = lines[:2]  # Mantém cabeçalho
                # Seleciona linhas mais relevantes com base na consulta
                relevant_lines = self._select_relevant_table_rows(lines[2:], query)
                # Limita a 10 linhas no total
                selected_lines = header + relevant_lines[:10]
                # Adiciona indicação se truncou
                if len(lines) > 12:
                    selected_lines.append("... (tabela truncada)")
                return "\n".join(selected_lines)
            
            # Se não for muito grande, retorna como está
            return table_content
                
        except Exception as e:
            print(f"[RETRIEVAL] Erro ao processar tabela: {e}")
            return table_content  # Retorna o conteúdo original em caso de erro
    
    def _format_markdown_table(self, lines: List[str], query: str) -> str:
        """Formata tabela markdown mantendo estrutura e selecionando linhas relevantes"""
        # Preserva cabeçalho e linha separadora
        header = lines[:2]
        
        # Se a tabela não for muito grande, usa toda
        if len(lines) <= 12:
            return "\n".join(lines)
            
        # Seleciona linhas mais relevantes
        content_lines = lines[2:]
        
        # Filtra linhas relevantes com base na consulta
        relevant_lines = self._select_relevant_table_rows(content_lines, query)
        
        # Limita a um número razoável de linhas
        selected_lines = relevant_lines[:10]
        
        # Formata a tabela final
        result_table = header + selected_lines
        
        # Indica se foi truncada
        if len(content_lines) > len(selected_lines):
            truncated_msg = f"| *Tabela truncada (mostrando {len(selected_lines)} de {len(content_lines)} linhas)* |"
            result_table.append(truncated_msg)
            
        return "\n".join(result_table)
    
    def _format_csv_table(self, lines: List[str], query: str) -> str:
        """Converte tabela CSV para formato markdown mais legível"""
        if not lines:
            return ""
            
        # Extrai cabeçalho
        header = lines[0].split(",")
        header = [col.strip() for col in header]
        
        # Cria linha separadora markdown
        separator = "|" + "|".join(["---" for _ in header]) + "|"
        
        # Formata cabeçalho como markdown
        header_md = "|" + "|".join(header) + "|"
        
        # Se a tabela não for muito grande, converte toda
        if len(lines) <= 12:
            # Converte linhas de conteúdo para markdown
            content_lines = []
            for line in lines[1:]:
                cols = line.split(",")
                cols = [col.strip() for col in cols]
                content_lines.append("|" + "|".join(cols) + "|")
                
            return "\n".join([header_md, separator] + content_lines)
        
        # Para tabelas grandes, seleciona linhas relevantes
        content_lines = lines[1:]
        
        # Filtra linhas relevantes
        relevant_lines = self._select_relevant_table_rows(content_lines, query)
        
        # Limita a um número razoável de linhas
        selected_lines = relevant_lines[:10]
        
        # Converte para markdown
        md_lines = []
        for line in selected_lines:
            cols = line.split(",")
            cols = [col.strip() for col in cols]
            md_lines.append("|" + "|".join(cols) + "|")
            
        # Adiciona nota sobre truncamento se necessário
        if len(content_lines) > len(selected_lines):
            truncated_msg = f"|*Tabela truncada (mostrando {len(selected_lines)} de {len(content_lines)} linhas)*|"
            md_lines.append(truncated_msg)
            
        return "\n".join([header_md, separator] + md_lines)
    
    def _select_relevant_table_rows(self, lines: List[str], query: str) -> List[str]:
        """Seleciona linhas da tabela mais relevantes para a consulta"""
        # Palavras-chave da consulta
        keywords = query.lower().split()
        
        # Filtra palavras muito comuns
        stopwords = {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "na", "se", "por", "como", "mas", "ao", "dos", "as", "ou", "seu", "sua", "quais", "qual", "quem", "quanto"}
        keywords = [word for word in keywords if word not in stopwords and len(word) > 2]
        
        # Pontua cada linha com base na relevância
        scored_lines = []
        for line in lines:
            line_lower = line.lower()
            
            # Calcula pontuação baseada em quantas palavras-chave aparecem
            score = sum(1 for keyword in keywords if keyword in line_lower)
            
            # Adiciona a linha com sua pontuação
            scored_lines.append((score, line))
        
        # Ordena por pontuação, maior primeiro
        scored_lines.sort(reverse=True)
        
        # Retorna as linhas mais relevantes
        return [line for _, line in scored_lines]

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        """Recupera imagem e descrição de forma assíncrona"""
        # Verifica o cache de imagem específica
        cache_key = f"img:{image_uuid}"
        if cache_key in self._context_cache:
            entry = self._context_cache[cache_key]
            if self._is_cache_valid(entry["timestamp"]):
                return entry["data"]
        
        try:
            # Busca no MongoDB
            image_data = await self.image_collection.find_one({"_id": image_uuid})
            if not image_data:
                return {"type": "error", "message": "Imagem não encontrada"}

            image_bytes = image_data.get("image_data")
            if not image_bytes:
                return {"type": "error", "message": "Dados da imagem ausentes"}

            # Processamento de bytes
            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                return {"type": "error", "message": "Formato de imagem não suportado"}

            # Busca metadados no vetor
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
                
            result = {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content
            }
            
            # Armazena no cache
            self._context_cache[cache_key] = {
                "data": result,
                "timestamp": time.time()
            }
            
            return result
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
        """Analisa a relevância dos contextos com processamento otimizado"""
        # Verifica se há algum contexto para analisar
        if not any([text_context, image_context.get("description") if isinstance(image_context, dict) else False, 
                  table_context.get("content") if isinstance(table_context, dict) else False]):
            return self._get_default_analysis()

        # Prepara os trechos de contexto para análise
        image_description = ""
        if image_context and isinstance(image_context, dict):
            image_description = image_context.get("description", "")

        table_content = ""
        if table_context and isinstance(table_context, dict):
            table_content = table_context.get("content", "")

        # Trunca os contextos para economizar tokens
        text_preview = str(text_context)[:300] + "..." if text_context and len(str(text_context)) > 300 else str(text_context or "")
        image_preview = str(image_description)[:300] + "..." if len(str(image_description)) > 300 else str(image_description)
        table_preview = str(table_content)[:300] + "..." if len(str(table_content)) > 300 else str(table_content)

        # Gera a análise de relevância
        try:
            prompt = ChatPromptTemplate.from_template(self.RELEVANCE_ANALYSIS_PROMPT)
            response = await self.model.ainvoke(prompt.format(
                question=original_question,
                text_context=text_preview,
                image_context=image_preview,
                table_context=table_preview
            ))

            # Processa o resultado
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            analysis = json.loads(cleaned_content)
            
            # Valida os campos
            required_fields = ["text", "image", "table", "recommended_context"]
            if not all(field in analysis for field in required_fields):
                raise ValueError("Missing required fields in analysis")

            return analysis

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"[RETRIEVAL] Erro na análise de relevância: {str(e)}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Análise padrão quando não há contexto ou ocorre erro"""
        return {
            "text": {"score": 0.5, "reason": "Contexto padrão utilizado"},
            "image": {"score": 0.2, "reason": "Sem imagem relevante"},
            "table": {"score": 0.1, "reason": "Sem tabela relevante"},
            "recommended_context": "text"
        }
        
    def clear_cache(self):
        """Limpa o cache de contexto"""
        self._context_cache = {}

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
    """
    Cria um nó de ensino melhorado com variabilidade nas respostas
    e personalização baseada no perfil do aluno.
    """
    
    # Cada estilo de aprendizagem terá um template diferente
    PROMPT_TEMPLATES = {
        "Sequencial": """
        ROLE: Tutor Educacional - Abordagem Sequencial
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Sequencial, use estas técnicas:
        - Apresente conceitos em passos lógicos e ordenados
        - Forneça explicações detalhadas passo a passo
        - Destaque a progressão linear do conhecimento
        - Construa sobre conceitos já apresentados
        - Use listas numeradas e sequências claras
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Inicie com uma breve conexão ao histórico recente
        2. Apresente uma sequência clara de passos ou conceitos
        3. Explique cada ponto separadamente e em ordem lógica
        4. Faça conexões explícitas entre os conceitos
        5. Conclua com os próximos passos lógicos para aprofundamento
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """,
        
        "Global": """
        ROLE: Tutor Educacional - Abordagem Global
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Global, use estas técnicas:
        - Comece com o panorama geral e a visão completa
        - Relacione o tópico com o contexto mais amplo
        - Faça conexões entre diferentes áreas do conhecimento
        - Use analogias e metáforas
        - Destaque padrões e relações entre conceitos
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Inicie com uma visão geral do tema e sua relevância
        2. Conecte o tópico com outros conceitos já conhecidos
        3. Apresente o quadro completo antes dos detalhes
        4. Destaque como as partes se encaixam no todo
        5. Conclua mostrando como este conhecimento se integra ao campo maior
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """,
        
        "Visual": """
        ROLE: Tutor Educacional - Abordagem Visual
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Visual, use estas técnicas:
        - Descreva imagens mentais e representações visuais
        - Use linguagem rica em descrições visuais
        - Sugira diagramas, gráficos e mapas conceituais
        - Organize informações espacialmente
        - Utilize metáforas visuais para explicar conceitos
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Crie uma imagem mental clara do conceito
        2. Descreva relações espaciais entre elementos
        3. Sugira como o aluno poderia visualizar o conceito
        4. Utilize descrições ricas em detalhes visuais
        5. Recomende recursos visuais complementares quando possível
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """,
        
        "Verbal": """
        ROLE: Tutor Educacional - Abordagem Verbal
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Verbal, use estas técnicas:
        - Utilize explicações detalhadas e precisas em texto
        - Defina termos com cuidado e precisão
        - Ofereça explicações claras e bem estruturadas
        - Use narrativas e explicações passo a passo
        - Forneça definições formais quando apropriado
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Comece com definições claras dos termos-chave
        2. Desenvolva explicações bem estruturadas
        3. Use vocabulário preciso e técnico quando necessário
        4. Forneça exemplos verbais e narrativos
        5. Conclua com um resumo verbal conciso dos pontos principais
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """,
        
        "Ativo": """
        ROLE: Tutor Educacional - Abordagem Ativa
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Ativa, use estas técnicas:
        - Proponha exercícios práticos e aplicações
        - Incentive a experimentação e tentativa-erro
        - Faça perguntas que estimulem o pensamento
        - Sugira atividades práticas e projetos
        - Encoraje a aplicação imediata dos conceitos
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Comece com uma questão desafiadora sobre o tema
        2. Apresente o conteúdo intercalado com perguntas reflexivas
        3. Sugira pelo menos um exercício prático ou atividade
        4. Proponha um pequeno desafio que aplique o conceito
        5. Incentive o aluno a experimentar e compartilhar resultados
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """,
        
        "Reflexivo": """
        ROLE: Tutor Educacional - Abordagem Reflexiva
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Reflexiva, use estas técnicas:
        - Ofereça tempo e espaço para reflexão
        - Apresente diferentes perspectivas sobre o tema
        - Estimule o pensamento crítico e a análise
        - Forneça material para consideração aprofundada
        - Encoraje a conexão com conhecimentos prévios
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Apresente o conceito de forma completa primeiro
        2. Ofereça múltiplas perspectivas ou interpretações
        3. Levante questões reflexivas sobre as implicações
        4. Convide o aluno a considerar e analisar o conteúdo
        5. Conclua com perguntas abertas que estimulem reflexão contínua
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """,
        
        "Sensorial": """
        ROLE: Tutor Educacional - Abordagem Sensorial
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Sensorial, use estas técnicas:
        - Forneça exemplos concretos e práticos
        - Concentre-se em fatos observáveis e dados
        - Use exemplos do mundo real e aplicações práticas
        - Apresente procedimentos e métodos específicos
        - Inclua detalhes e passos bem definidos
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Comece com um exemplo concreto ou caso prático
        2. Forneça dados específicos e observáveis
        3. Demonstre aplicações práticas do conceito
        4. Use linguagem direta e precisa
        5. Conclua com sugestões de aplicação no mundo real
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """,
        
        "Intuitivo": """
        ROLE: Tutor Educacional - Abordagem Intuitiva
        
        PERFIL DO ALUNO:
        {user_profile}
        
        SUA ABORDAGEM:
        Para este aluno com preferência Intuitiva, use estas técnicas:
        - Foque em conceitos abstratos e teóricos
        - Explore significados e interpretações
        - Discuta possibilidades e inovações
        - Apresente padrões e tendências gerais
        - Conecte o tema com teoria e princípios gerais
        
        CONTEXTO:
        - Plano: {learning_plan}
        - Fonte: {source_type}
        - Contexto principal: {primary_context}
        - Contextos secundários: {secondary_contexts}
        - Histórico da conversa: {chat_history}
        - Pergunta atual: {question}
        
        FORMATO DA RESPOSTA:
        1. Inicie com um princípio ou teoria geral
        2. Explore significados subjacentes e implicações
        3. Faça conexões com conceitos abstratos relacionados
        4. Discuta potenciais inovações ou aplicações futuras
        5. Conclua com considerações teóricas mais amplas
        
        MANTENHA O FOCO EDUCACIONAL, GUIANDO O ALUNO SEM DAR RESPOSTAS DIRETAS.
        """
    }
    
    # Prompt dedicado para respostas diretas (sem contexto)
    DIRECT_PROMPT = """
    ROLE: Tutor Educacional Personalizado
    
    PERFIL DO ALUNO E SUAS PREFERÊNCIAS:
    {user_profile}
    
    PLANO DE RESPOSTA E CONTEXTO:
    {learning_plan}
    
    HISTÓRICO DA CONVERSA:
    {chat_history}
    
    PERGUNTA ATUAL:
    {question}
    
    SUA MISSÃO:
    1. Criar uma resposta educativa personalizada alinhada com as preferências de aprendizagem do aluno
    2. Guiar sem dar respostas diretas, incentivando o desenvolvimento do raciocínio próprio
    3. Variar seu estilo de comunicação para evitar respostas previsíveis e repetitivas 
    4. Adaptar seu tom e abordagem para otimizar o engajamento
    
    DIRETRIZES DE RESPOSTA:
    - SEMPRE personalize seu estilo conforme as preferências do aluno (Sensorial/Intuitivo, Visual/Verbal, Ativo/Reflexivo, Sequencial/Global)
    - NUNCA dê a resposta completa, mas guie com ferramentas apropriadas ao estilo do aluno
    - Use vocabulário progressivamente mais complexo e técnico conforme o aluno avança
    - Evite repetir padrões de resposta usados anteriormente
    
    IMPORTANTE: Sua resposta deve ser diferente de interações anteriores, mesmo para perguntas similares. 
    Inove constantemente na forma de apresentar o conteúdo.
    """
    
    # Cache para tracking de variabilidade
    response_patterns = {}
    
    # Inicializa modelos com diferentes temperaturas para variabilidade
    models = {
        "precise": ChatOpenAI(model="gpt-4o", temperature=0.1),
        "creative": ChatOpenAI(model="gpt-4o", temperature=0.4),
        "balanced": ChatOpenAI(model="gpt-4o", temperature=0.2)
    }

    def generate_teaching_response(state: AgentState) -> AgentState:
        """Gera resposta de ensino personalizada"""
        # Obtém dados necessários
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"], max_messages=5)
        
        try:
            # Identifica modelo principal
            session_id = state.get("session_id", "default")
            request_count = len([m for m in state.get("chat_history", []) if isinstance(m, HumanMessage)])
            
            # Rotação de modelos para garantir variabilidade nas respostas
            if request_count % 3 == 0:
                model_key = "creative"  # Maior variabilidade a cada 3 perguntas
            elif request_count % 3 == 1:
                model_key = "balanced"  # Balanceado para maioria das respostas
            else:
                model_key = "precise"   # Mais preciso periodicamente
            
            model = models[model_key]
            
            # Determina o tipo de resposta (direta ou baseada em contexto)
            if state.get("next_step") == "direct_answer":
                # Resposta direta (sem retrieval)
                prompt = ChatPromptTemplate.from_template(DIRECT_PROMPT)
                explanation = model.invoke(prompt.format(
                    learning_plan=state["current_plan"],
                    user_profile=state["user_profile"],
                    question=latest_question,
                    chat_history=chat_history
                ))
                image_content = None
                
            else:
                # Resposta baseada em contexto recuperado
                # Processa os contextos disponíveis
                if state.get("web_search_results"):
                    # Formatação para dados de busca web
                    source_type = "Resultados de busca web"
                    web_results = state["web_search_results"]
                    primary_context = f"Wikipedia:\n{web_results.get('wikipedia', 'Não disponível')}"
                    secondary_contexts = f"YouTube:\n{web_results.get('youtube', 'Não disponível')}"
                    
                else:
                    # Formatação para dados de retrieval de contexto
                    contexts = state["extracted_context"]
                    relevance = contexts.get("relevance_analysis", {})
                    
                    # Analisa scores de contexto
                    context_scores = {
                        "text": relevance.get("text", {}).get("score", 0),
                        "image": relevance.get("image", {}).get("score", 0),
                        "table": relevance.get("table", {}).get("score", 0)
                    }
                    
                    # Ordena contextos por relevância
                    sorted_contexts = sorted(
                        context_scores.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    # Define tipo fonte e contexto principal
                    source_type = "Material de estudo"
                    primary_type = sorted_contexts[0][0] if sorted_contexts else "text"
                    
                    # Formata contexto primário
                    if primary_type == "text":
                        primary_context = f"Texto: {contexts.get('text', '')}"
                    elif primary_type == "image":
                        primary_context = f"Descrição da Imagem: {contexts.get('image', {}).get('description', '')}"
                    else:
                        primary_context = f"Tabela: {contexts.get('table', {}).get('content', '')}"
                    
                    # Prepara contextos secundários
                    secondary_contexts_list = []
                    for context_type, score in sorted_contexts[1:]:
                        if score > 0.3:  # Filtra apenas contextos relevantes
                            if context_type == "text":
                                secondary_contexts_list.append(f"Texto: {contexts.get('text', '')}")
                            elif context_type == "image":
                                secondary_contexts_list.append(f"Descrição da Imagem: {contexts.get('image', {}).get('description', '')}")
                            elif context_type == "table":
                                secondary_contexts_list.append(f"Tabela: {contexts.get('table', {}).get('content', '')}")
                    
                    secondary_contexts = "\n\n".join(secondary_contexts_list)
                
                # Determina o estilo de aprendizagem predominante para personalizar o prompt
                learning_style = determine_predominant_style(state["user_profile"])
                prompt_template = PROMPT_TEMPLATES.get(learning_style, PROMPT_TEMPLATES["Verbal"])
                
                # Cria o prompt personalizado
                prompt = ChatPromptTemplate.from_template(prompt_template)
                
                # Invoca o modelo
                explanation = model.invoke(prompt.format(
                    learning_plan=state["current_plan"],
                    user_profile=state["user_profile"],
                    source_type=source_type,
                    primary_context=primary_context,
                    secondary_contexts=secondary_contexts,
                    question=latest_question,
                    chat_history=chat_history
                ))
                
                # Processa imagem se disponível e relevante
                image_content = None
                if (state.get("extracted_context") and 
                    state["extracted_context"].get("image", {}).get("type") == "image" and
                    state["extracted_context"].get("image", {}).get("image_bytes") and
                    context_scores.get("image", 0) > 0.3):
                    image_content = state["extracted_context"]["image"]["image_bytes"]
            
            # Rastreia padrões de resposta para análise
            track_response_pattern(session_id, latest_question, explanation.content)
            
            # Formata a resposta final
            if image_content:
                # Formata resposta com imagem (multimodal)
                base64_image = base64.b64encode(image_content).decode('utf-8')
                response_content = {
                    "type": "multimodal",
                    "content": explanation.content,
                    "image": f"data:image/jpeg;base64,{base64_image}"
                }
                response = AIMessage(content=json.dumps(response_content))
                history_message = AIMessage(content=explanation.content)
            else:
                # Resposta apenas texto
                response = explanation
                history_message = explanation
            
            # Atualiza o estado do agente
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
            
            # Resposta de erro
            error_message = "Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente."
            response = AIMessage(content=error_message)
            
            # Atualiza estado com erro
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                response
            ]
            
            return new_state

    def determine_predominant_style(user_profile: Dict[str, Any]) -> str:
        """Determina o estilo de aprendizagem predominante do aluno"""
        try:
            # Tenta encontrar o estilo de aprendizagem nos dados do perfil
            learning_styles = user_profile.get("EstiloAprendizagem", {})
            
            # Se não houver dados de estilo, use verbal como padrão
            if not learning_styles:
                return "Verbal"
                
            # Mapeie os estilos para o formato esperado
            style_map = {
                "Entrada": {"Visual": "Visual", "Verbal": "Verbal"},
                "Processamento": {"Ativo": "Ativo", "Reflexivo": "Reflexivo"},
                "Entendimento": {"Sequencial": "Sequencial", "Global": "Global"},
                "Percepcao": {"Sensorial": "Sensorial", "Intuitivo": "Intuitivo"}
            }
            
            # Obtém os valores de cada dimensão
            for dimension, styles in style_map.items():
                if dimension in learning_styles:
                    value = learning_styles[dimension]
                    if value in styles.values():
                        return value
            
            # Fallback para Verbal se nenhum estilo for identificado
            return "Verbal"
            
        except Exception as e:
            print(f"Error determining learning style: {e}")
            return "Verbal"  # Estilo padrão em caso de erro
    
    def track_response_pattern(session_id: str, question: str, response: str) -> None:
        """Rastreia padrões nas respostas para evitar repetições"""
        if session_id not in response_patterns:
            response_patterns[session_id] = []
            
        # Armazena apenas um resumo da resposta para análise
        response_summary = response[:100] + "..." if len(response) > 100 else response
        response_patterns[session_id].append({
            "timestamp": time.time(),
            "question": question,
            "response_summary": response_summary
        })
        
        # Limita o histórico para economia de memória
        if len(response_patterns[session_id]) > 10:
            response_patterns[session_id] = response_patterns[session_id][-10:]

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