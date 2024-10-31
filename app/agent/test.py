from typing import Any, TypedDict, List, Dict, Optional, Union
from typing_extensions import TypeVar
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from datetime import datetime

# Prompts constants
ANSWER_PLAN_PROMPT = """Você é um planejador de respostas educacionais especializado em criar planos adaptados ao perfil do aluno.

Contexto Atual:
{current_context}

Perfil do Aluno:
{student_profile}

Estilo de Aprendizagem:
- Percepção: {learning_style_perception}
- Entrada: {learning_style_input}
- Processamento: {learning_style_processing}
- Entendimento: {learning_style_understanding}

Estado Atual do Plano:
- Etapa: {current_step_title}
- Progresso: {current_step_progress}%
- Descrição: {current_step_description}

Pergunta do Aluno:
{question}

Com base nas informações acima, crie um plano de resposta detalhado que:

1. ANÁLISE DE CONTEXTO E ALINHAMENTO:
- Identifique o contexto específico da pergunta
- Avalie o alinhamento com a etapa atual
- Determine pré-requisitos necessários

2. OBJETIVOS DE APRENDIZADO:
- Defina objetivos específicos e mensuráveis
- Alinhe com o estilo de aprendizagem do aluno
- Estabeleça critérios de sucesso

3. ESTRATÉGIA DE RESPOSTA:
- Estruture a explicação em etapas claras
- Adapte ao nível de conhecimento atual
- Inclua pontos de verificação de compreensão

4. RECURSOS E ATIVIDADES:
- Selecione recursos alinhados ao perfil
- Proponha exercícios práticos relevantes
- Sugira materiais complementares

5. MÉTRICAS DE AVALIAÇÃO:
- Defina indicadores de compreensão
- Estabeleça pontos de verificação
- Proponha método de feedback

Forneça o plano no seguinte formato JSON:
{
    "contexto": {
        "area_conhecimento": string,
        "nivel_complexidade": string,
        "pre_requisitos": [string]
    },
    "objetivos": {
        "primario": string,
        "secundarios": [string],
        "criterios_sucesso": [string]
    },
    "estrategia": {
        "abordagem": string,
        "etapas": [
            {
                "titulo": string,
                "descricao": string,
                "duracao_estimada": string
            }
        ],
        "pontos_verificacao": [string]
    },
    "recursos": {
        "materiais": [string],
        "exercicios": [
            {
                "tipo": string,
                "descricao": string,
                "nivel": string
            }
        ],
        "complementares": [string]
    },
    "avaliacao": {
        "indicadores": [string],
        "checkpoints": [string],
        "feedback": {
            "tipo": string,
            "criterios": [string]
        }
    }
}
"""

RETRIEVAL_ANALYSIS_PROMPT = """Você é um especialista em análise de relevância de conteúdo educacional.

Pergunta do Aluno:
{question}

Contexto Recuperado:
{retrieved_context}

Perfil do Aluno:
{student_profile}

Avalie a relevância e adequação do conteúdo considerando:

1. RELEVÂNCIA DO CONTEÚDO:
- Alinhamento com a pergunta
- Cobertura do tópico
- Precisão da informação

2. ADEQUAÇÃO AO PERFIL:
- Nível de complexidade
- Estilo de aprendizagem
- Conhecimento prévio necessário

Retorne sua análise no seguinte formato JSON:
{
    "relevancia": {
        "score": float (0-1),
        "justificativa": string,
        "aspectos_cobertos": [string],
        "lacunas": [string]
    },
    "adequacao": {
        "complexidade_adequada": boolean,
        "alinhamento_estilo": boolean,
        "pre_requisitos_atendidos": boolean,
        "ajustes_necessarios": [string]
    },
    "recomendacao": {
        "usar_conteudo": boolean,
        "modificacoes_sugeridas": [string],
        "busca_adicional_necessaria": boolean
    }
}
"""

TEACHING_RESPONSE_PROMPT = """Você é um tutor educacional especializado em explicações adaptativas.

Plano de Resposta:
{response_plan}

Perfil do Aluno:
{student_profile}

Contexto da Resposta:
{context}

Histórico da Conversa:
{chat_history}

Pergunta:
{question}

Elabore uma resposta que:

1. INTRODUÇÃO:
- Conecte com conhecimento prévio
- Estabeleça objetivos claros
- Desperte interesse

2. DESENVOLVIMENTO:
- Explique progressivamente
- Use analogias apropriadas
- Forneça exemplos práticos

3. VERIFICAÇÃO:
- Faça perguntas de compreensão
- Proponha aplicações práticas
- Esclareça possíveis dúvidas

4. CONCLUSÃO:
- Resuma pontos principais
- Conecte com próximos passos
- Forneça recursos adicionais

Lembre-se de:
- Manter linguagem adequada ao nível
- Usar exemplos relevantes ao contexto
- Verificar compreensão regularmente
- Adaptar ao estilo de aprendizagem
- Manter engajamento ativo

Retorne sua resposta em formato que mantenha a estrutura clara e facilite a compreensão."""

PROGRESS_ANALYSIS_PROMPT = """Você é um analista especializado em avaliação de progresso educacional.

Interação Atual:
{current_interaction}

Histórico de Progresso:
{progress_history}

Métricas de Aprendizado:
{learning_metrics}

Analise o progresso considerando:

1. COMPREENSÃO:
- Nível de entendimento demonstrado
- Qualidade das respostas
- Aplicação dos conceitos

2. ENGAJAMENTO:
- Participação ativa
- Qualidade das perguntas
- Interesse demonstrado

3. EVOLUÇÃO:
- Progresso em relação aos objetivos
- Velocidade de aprendizado
- Superação de dificuldades

Retorne sua análise no seguinte formato JSON:
{
    "compreensao": {
        "nivel": string,
        "evidencias": [string],
        "areas_melhoria": [string]
    },
    "engajamento": {
        "nivel": string,
        "indicadores": [string],
        "sugestoes_melhoria": [string]
    },
    "evolucao": {
        "progresso_percentual": float,
        "marcos_alcancados": [string],
        "proximos_passos": [string]
    },
    "recomendacoes": {
        "ajustes_necessarios": [string],
        "intervencoes_sugeridas": [string],
        "recursos_recomendados": [string]
    }
}
"""

# Basic Types and Models
@dataclass
class ExecutionStep:
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[Dict]
    atividade: Dict
    progresso: int

class LearningMetrics(BaseModel):
    time_spent: int
    completed_activities: int
    correct_answers: int
    engagement_level: float
    comprehension_rate: float

class SessionProgress(BaseModel):
    current_step: str
    completed_steps: List[str]
    step_metrics: Dict[str, Any]
    overall_progress: float

class EngagementData(BaseModel):
    interaction_count: int
    average_response_time: float
    session_duration: int
    focus_periods: List[Dict[str, Any]]

class PerformanceMetrics(BaseModel):
    accuracy: float
    completion_rate: float
    learning_velocity: float
    retention_rate: float

class GraphState(TypedDict):
    messages: List[BaseMessage]
    current_plan: str
    user_profile: dict
    extracted_context: str
    next_step: str | None
    iteration_count: int
    chat_history: List[BaseMessage]
    learning_metrics: Dict[str, Any]
    session_progress: Dict[str, Any]
    engagement_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    current_node: str
    error_count: int

class WorkflowConfig(BaseModel):
    vector_store: Any
    db_handler: Any
    memory_handler: Any
    model_config: Dict[str, Any]
    student_email: str
    disciplina: str
    session_id: str


from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Any, Dict, List, Optional
import json
import asyncio
from datetime import datetime

class AnswerPlanGenerator:
    def __init__(self, model_config: Dict[str, Any]):
        self.model = ChatOpenAI(
            model=model_config.get("model_name", "gpt-4"),
            temperature=model_config.get("temperature", 0)
        )
        self.prompt = ChatPromptTemplate.from_template(ANSWER_PLAN_PROMPT)

    async def generate(self, state: GraphState) -> GraphState:
        try:
            print("\n[PLAN] Iniciando geração do plano de resposta")
            
            # Extrai última mensagem
            latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
            
            # Extrai informações do estado atual
            current_step = self._extract_current_step(state["current_plan"])
            learning_style = state["user_profile"].get("EstiloAprendizagem", {})
            
            # Gera o plano
            response = await self.model.ainvoke(
                self.prompt.format(
                    current_context=state["extracted_context"],
                    student_profile=state["user_profile"],
                    learning_style_perception=learning_style.get("Percepcao", ""),
                    learning_style_input=learning_style.get("Entrada", ""),
                    learning_style_processing=learning_style.get("Processamento", ""),
                    learning_style_understanding=learning_style.get("Entendimento", ""),
                    current_step_title=current_step.get("titulo", ""),
                    current_step_progress=current_step.get("progresso", 0),
                    current_step_description=current_step.get("descricao", ""),
                    question=latest_message.content
                )
            )
            
            # Atualiza estado
            new_state = state.copy()
            new_state["current_plan"] = json.loads(response.content)
            new_state["current_node"] = "plan_generation"
            
            print("[PLAN] Plano de resposta gerado com sucesso")
            return new_state
            
        except Exception as e:
            print(f"[PLAN] Erro na geração do plano: {str(e)}")
            raise e

    def _extract_current_step(self, plan_str: str) -> Dict[str, Any]:
        if not plan_str:
            return {}
            
        try:
            plan = json.loads(plan_str) if isinstance(plan_str, str) else plan_str
            steps = plan.get("plano_execucao", [])
            return next(
                (step for step in steps if step.get("progresso", 100) < 100),
                steps[-1] if steps else {}
            )
        except:
            return {}

class RetrievalAgent:
    def __init__(self, vector_store: Any, model_config: Dict[str, Any]):
        self.vector_store = vector_store
        self.model = ChatOpenAI(
            model=model_config.get("model_name", "gpt-4o-mini"),
            temperature=model_config.get("temperature", 0)
        )
        self.prompt = ChatPromptTemplate.from_template(RETRIEVAL_ANALYSIS_PROMPT)
        self.cache = {}

    async def retrieve(self, state: GraphState) -> GraphState:
        try:
            print("\n[RETRIEVAL] Iniciando recuperação de contexto")
            
            # Extrai última mensagem
            latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
            query = latest_message.content
            
            # Verifica cache
            cache_key = self._generate_cache_key(query, state["user_profile"])
            if cache_key in self.cache:
                print("[RETRIEVAL] Utilizando resultado em cache")
                new_state = state.copy()
                new_state["extracted_context"] = self.cache[cache_key]
                new_state["current_node"] = "context_retrieval"
                return new_state

            # Realiza busca vetorial
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=5,
                filter=self._build_search_filter(state)
            )

            if not results:
                print("[RETRIEVAL] Nenhum resultado encontrado")
                return self._handle_no_results(state)

            # Analisa relevância
            analysis = await self._analyze_relevance(
                query=query,
                context=results,
                profile=state["user_profile"]
            )

            # Processa resultados
            if analysis["recomendacao"]["usar_conteudo"]:
                context = self._process_results(results, analysis)
                self.cache[cache_key] = context
            else:
                return await self._handle_irrelevant_results(state, analysis)

            # Atualiza estado
            new_state = state.copy()
            new_state["extracted_context"] = context
            new_state["current_node"] = "context_retrieval"
            
            print("[RETRIEVAL] Contexto recuperado com sucesso")
            return new_state

        except Exception as e:
            print(f"[RETRIEVAL] Erro na recuperação: {str(e)}")
            raise e

    def _generate_cache_key(self, query: str, profile: Dict) -> str:
        # Implementa geração de chave de cache
        profile_key = json.dumps(profile, sort_keys=True)
        return f"{query}_{hash(profile_key)}"

    def _build_search_filter(self, state: GraphState) -> Dict[str, Any]:
        # Implementa construção de filtro de busca
        return {
            "discipline": state.get("current_plan", {}).get("disciplina", ""),
            "difficulty_level": state["user_profile"].get("nivel", "intermediario")
        }

    async def _analyze_relevance(
        self,
        query: str,
        context: List[Any],
        profile: Dict
    ) -> Dict[str, Any]:
        # Analisa relevância do conteúdo
        context_text = "\n".join([doc.page_content for doc in context])
        
        response = await self.model.ainvoke(
            self.prompt.format(
                question=query,
                retrieved_context=context_text,
                student_profile=profile
            )
        )
        
        return json.loads(response.content)

    def _process_results(
        self,
        results: List[Any],
        analysis: Dict[str, Any]
    ) -> str:
        # Processa e formata resultados
        processed = []
        for doc in results:
            processed.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance": analysis["relevancia"]["score"]
            })
        
        return json.dumps(processed)

    def _handle_no_results(self, state: GraphState) -> GraphState:
        # Manipula caso de nenhum resultado
        new_state = state.copy()
        new_state["extracted_context"] = ""
        new_state["current_node"] = "context_retrieval"
        return new_state

    async def _handle_irrelevant_results(
        self,
        state: GraphState,
        analysis: Dict[str, Any]
    ) -> GraphState:
        # Manipula resultados irrelevantes
        new_state = state.copy()
        new_state["extracted_context"] = json.dumps({
            "error": "conteudo_irrelevante",
            "analysis": analysis,
            "recommendations": analysis["recomendacao"]["modificacoes_sugeridas"]
        })
        new_state["current_node"] = "context_retrieval"
        return new_state

class ChatAgent:
    def __init__(self, model_config: Dict[str, Any]):
        self.model = ChatOpenAI(
            model=model_config.get("model_name", "gpt-4o-mini"),
            temperature=model_config.get("temperature", 0.5)
        )
        self.prompt = ChatPromptTemplate.from_template(TEACHING_RESPONSE_PROMPT)

    async def respond(self, state: GraphState) -> GraphState:
        try:
            print("\n[CHAT] Iniciando geração de resposta")
            
            # Extrai última mensagem e histórico recente
            latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
            recent_history = state["chat_history"][-5:]
            
            # Gera resposta
            response = await self.model.ainvoke(
                self.prompt.format(
                    response_plan=state["current_plan"],
                    student_profile=state["user_profile"],
                    context=state["extracted_context"],
                    chat_history=recent_history,
                    question=latest_message.content
                )
            )
            
            # Adiciona verificação de compreensão
            comprehension_check = await self._generate_comprehension_check(
                response.content,
                state["user_profile"]
            )
            
            # Atualiza estado
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [
                AIMessage(content=response.content),
                AIMessage(content=comprehension_check)
            ]
            new_state["chat_history"] = list(state["chat_history"]) + [
                latest_message,
                AIMessage(content=response.content)
            ]
            new_state["current_node"] = "chat_response"
            
            print("[CHAT] Resposta gerada com sucesso")
            return new_state
            
        except Exception as e:
            print(f"[CHAT] Erro na geração de resposta: {str(e)}")
            raise e

    async def _generate_comprehension_check(
        self,
        response: str,
        profile: Dict[str, Any]
    ) -> str:
        # Implementa geração de verificação de compreensão
        check_prompt = """
        Gere uma pergunta de verificação de compreensão sobre:
        
        Resposta dada: {response}
        
        Nível do aluno: {student_level}
        Estilo de aprendizagem: {learning_style}
        
        A pergunta deve:
        1. Verificar entendimento do conceito principal
        2. Ser adequada ao nível do aluno
        3. Promover reflexão
        """
        
        check_response = await self.model.ainvoke(
            ChatPromptTemplate.from_template(check_prompt).format(
                response=response,
                student_level=profile.get("nivel", "intermediario"),
                learning_style=profile.get("EstiloAprendizagem", {})
            )
        )
        
        return check_response.content

class ProgressAnalysisAgent:
    def __init__(self, db_handler: Any, model_config: Dict[str, Any]):
        self.db = db_handler
        self.model = ChatOpenAI(
            model=model_config.get("model_name", "gpt-4"),
            temperature=model_config.get("temperature", 0)
        )
        self.prompt = ChatPromptTemplate.from_template(PROGRESS_ANALYSIS_PROMPT)

    async def analyze(self, state: GraphState) -> GraphState:
        try:
            print("\n[PROGRESS] Iniciando análise de progresso")
            
            # Extrai métricas atuais
            current_metrics = self._extract_interaction_metrics(state)
            
            # Analisa progresso
            analysis = await self._analyze_progress(
                current_metrics=current_metrics,
                state=state
            )
            
            # Atualiza métricas
            updated_metrics = self._update_metrics(
                current_state=state,
                new_analysis=analysis
            )
            
            # Persiste dados
            await self._persist_metrics(
                student_id=state["user_profile"].get("email", ""),
                metrics=updated_metrics
            )
            
            # Atualiza estado
            new_state = state.copy()
            new_state.update({
                "learning_metrics": updated_metrics["learning"],
                "session_progress": updated_metrics["progress"],
                "engagement_data": updated_metrics["engagement"],
                "current_node": "progress_analysis"
            })
            
            print("[PROGRESS] Análise de progresso concluída")
            return new_state
            
        except Exception as e:
            print(f"[PROGRESS] Erro na análise de progresso: {str(e)}")
            raise e

    def _extract_interaction_metrics(self, state: GraphState) -> Dict[str, Any]:
        """Extrai métricas da interação atual."""
        latest_messages = [m for m in state["messages"][-2:]]
        
        # Análise básica da interação
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "interaction_time": 0,  # Será calculado se disponível
            "message_count": len(latest_messages),
            "response_length": sum(len(m.content) for m in latest_messages),
            "interaction_type": "qa_pair"
        }
        
        # Adiciona métricas de qualidade se disponível
        if len(latest_messages) >= 2:
            metrics.update(self._analyze_interaction_quality(latest_messages))
            
        return metrics

    def _analyze_interaction_quality(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Analisa qualidade da interação."""
        # Implementa análise de qualidade básica
        return {
            "complexity_level": self._estimate_complexity(messages),
            "engagement_signals": self._detect_engagement_signals(messages),
            "comprehension_indicators": self._detect_comprehension_signals(messages)
        }

    async def _analyze_progress(
        self,
        current_metrics: Dict[str, Any],
        state: GraphState
    ) -> Dict[str, Any]:
        """Realiza análise completa do progresso."""
        response = await self.model.ainvoke(
            self.prompt.format(
                current_interaction=current_metrics,
                progress_history=state["session_progress"],
                learning_metrics=state["learning_metrics"]
            )
        )
        
        return json.loads(response.content)

    def _update_metrics(
        self,
        current_state: GraphState,
        new_analysis: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Atualiza todas as métricas baseado na nova análise."""
        return {
            "learning": self._update_learning_metrics(
                current_state["learning_metrics"],
                new_analysis
            ),
            "progress": self._update_progress_metrics(
                current_state["session_progress"],
                new_analysis
            ),
            "engagement": self._update_engagement_metrics(
                current_state["engagement_data"],
                new_analysis
            )
        }

    async def _persist_metrics(self, student_id: str, metrics: Dict[str, Any]) -> None:
        """Persiste métricas no banco de dados."""
        try:
            await self.db.update_one(
                collection="student_metrics",
                query={"student_id": student_id},
                update={
                    "$set": {
                        "last_updated": datetime.utcnow(),
                        "current_metrics": metrics
                    },
                    "$push": {
                        "metrics_history": {
                            "timestamp": datetime.utcnow(),
                            "metrics": metrics
                        }
                    }
                },
                upsert=True
            )
        except Exception as e:
            print(f"[PROGRESS] Erro ao persistir métricas: {str(e)}")
            raise e

class PatternAnalysisAgent:
    def __init__(self, memory_handler: Any, model_config: Dict[str, Any]):
        self.memory = memory_handler
        self.model = ChatOpenAI(
            model=model_config.get("model_name", "gpt-4o-mini"),
            temperature=model_config.get("temperature", 0)
        )

    async def analyze(self, state: GraphState) -> GraphState:
        try:
            print("\n[PATTERN] Iniciando análise de padrões")
            
            # Realiza análises
            sentiment_analysis = await self._analyze_sentiment(state)
            learning_patterns = await self._analyze_learning_patterns(state)
            error_patterns = await self._analyze_error_patterns(state)
            
            # Consolida resultados
            analysis_results = {
                "sentiment": sentiment_analysis,
                "learning_patterns": learning_patterns,
                "error_patterns": error_patterns,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Atualiza memória de longo prazo
            await self._update_memory(
                student_id=state["user_profile"].get("email", ""),
                analysis=analysis_results
            )
            
            # Atualiza estado
            new_state = state.copy()
            new_state["analysis_results"] = analysis_results
            new_state["current_node"] = "pattern_analysis"
            
            print("[PATTERN] Análise de padrões concluída")
            return new_state
            
        except Exception as e:
            print(f"[PATTERN] Erro na análise de padrões: {str(e)}")
            raise e

    async def _analyze_sentiment(self, state: GraphState) -> Dict[str, Any]:
        """Analisa sentimento das interações recentes."""
        sentiment_prompt = """
        Analise o sentimento e estado emocional do aluno nas últimas interações:
        
        Interações: {interactions}
        
        Retorne um JSON com:
        {
            "sentiment": string (positivo, neutro, negativo),
            "confidence": float,
            "emotional_state": string,
            "engagement_indicators": [string]
        }
        """
        
        recent_interactions = "\n".join([
            m.content for m in state["chat_history"][-5:]
            if isinstance(m, HumanMessage)
        ])
        
        response = await self.model.ainvoke(
            ChatPromptTemplate.from_template(sentiment_prompt).format(
                interactions=recent_interactions
            )
        )
        
        return json.loads(response.content)

    async def _analyze_learning_patterns(self, state: GraphState) -> Dict[str, Any]:
        """Analisa padrões de aprendizado."""
        learning_prompt = """
        Analise os padrões de aprendizado baseado nas métricas e interações:
        
        Métricas: {metrics}
        Histórico: {history}
        
        Retorne um JSON com padrões identificados.
        """
        
        response = await self.model.ainvoke(
            ChatPromptTemplate.from_template(learning_prompt).format(
                metrics=state["learning_metrics"],
                history=state["chat_history"][-10:]
            )
        )
        
        return json.loads(response.content)

    async def _analyze_error_patterns(self, state: GraphState) -> Dict[str, Any]:
        """Analisa padrões de erro e dificuldades."""
        error_prompt = """
        Analise os padrões de erro e dificuldades:
        
        Histórico: {history}
        Progresso: {progress}
        
        Retorne um JSON com padrões de erro e recomendações.
        """
        
        response = await self.model.ainvoke(
            ChatPromptTemplate.from_template(error_prompt).format(
                history=state["chat_history"][-10:],
                progress=state["session_progress"]
            )
        )
        
        return json.loads(response.content)

    async def _update_memory(self, student_id: str, analysis: Dict[str, Any]) -> None:
        """Atualiza a memória de longo prazo com os resultados da análise."""
        try:
            await self.memory.update_one(
                collection="student_memory",
                query={"student_id": student_id},
                update={
                    "$push": {
                        "analysis_history": {
                            "timestamp": datetime.utcnow(),
                            "analysis": analysis
                        }
                    }
                },
                upsert=True
            )
        except Exception as e:
            print(f"[PATTERN] Erro ao atualizar memória: {str(e)}")
            raise e
        
from langgraph.graph import StateGraph, Graph
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import END, StateGraph, START
import asyncio
from typing import Dict, List, Any, Optional

class TutorWorkflow:
    def __init__(self, config: WorkflowConfig):
        """
        Inicializa o workflow do tutor com todos os componentes necessários.
        
        Args:
            config: Configuração contendo todas as dependências necessárias
        """
        print("\n[WORKFLOW] Inicializando TutorWorkflow")
        
        # Inicializa agentes
        self.answer_plan_generator = AnswerPlanGenerator(config.model_config)
        self.retrieval_agent = RetrievalAgent(config.vector_store, config.model_config)
        self.chat_agent = ChatAgent(config.model_config)
        self.progress_agent = ProgressAnalysisAgent(config.db_handler, config.model_config)
        self.pattern_agent = PatternAnalysisAgent(config.memory_handler, config.model_config)
        
        # Configura workflow
        self.workflow = self._build_workflow()
        self.config = config

    def _build_workflow(self) -> Graph:
        """Constrói o grafo do workflow."""
        print("[WORKFLOW] Construindo grafo do workflow")
        
        # Cria o grafo
        workflow = StateGraph(GraphState)
        
        # Adiciona nós
        workflow.add_node("plan_generation", self._create_plan_node())
        workflow.add_node("context_retrieval", self._create_retrieval_node())
        workflow.add_node("chat_response", self._create_chat_node())
        workflow.add_node("progress_analysis", self._create_progress_node())
        workflow.add_node("pattern_analysis", self._create_pattern_node())
        
        # Define ponto de entrada
        workflow.set_entry_point("plan_generation")
        
        # Adiciona arestas normais
        workflow.add_edge("plan_generation", "context_retrieval")
        workflow.add_edge("context_retrieval", "chat_response")
        workflow.add_edge("chat_response", "progress_analysis")
        workflow.add_edge("progress_analysis", "pattern_analysis")
        
        # Adiciona condicionais para continuação
        workflow.add_conditional_edges(
            "pattern_analysis",
            self._should_continue,
            {
                "continue": "plan_generation",
                "end": END
            }
        )
        
        # Adiciona handlers de erro para cada nó
        for node in ["plan_generation", "context_retrieval", "chat_response", 
                    "progress_analysis", "pattern_analysis"]:
            workflow.add_exception_handler(node, self._handle_error)
        
        print("[WORKFLOW] Grafo do workflow construído com sucesso")
        return workflow.compile()

    def _create_plan_node(self):
        """Cria nó de geração de plano."""
        async def plan_node(state: GraphState) -> GraphState:
            print("\n[NODE:PLAN] Executando nó de planejamento")
            return await self.answer_plan_generator.generate(state)
        return plan_node

    def _create_retrieval_node(self):
        """Cria nó de recuperação de contexto."""
        async def retrieval_node(state: GraphState) -> GraphState:
            print("\n[NODE:RETRIEVAL] Executando nó de recuperação")
            return await self.retrieval_agent.retrieve(state)
        return retrieval_node

    def _create_chat_node(self):
        """Cria nó de resposta do chat."""
        async def chat_node(state: GraphState) -> GraphState:
            print("\n[NODE:CHAT] Executando nó de chat")
            return await self.chat_agent.respond(state)
        return chat_node

    def _create_progress_node(self):
        """Cria nó de análise de progresso."""
        async def progress_node(state: GraphState) -> GraphState:
            print("\n[NODE:PROGRESS] Executando nó de progresso")
            return await self.progress_agent.analyze(state)
        return progress_node

    def _create_pattern_node(self):
        """Cria nó de análise de padrões."""
        async def pattern_node(state: GraphState) -> GraphState:
            print("\n[NODE:PATTERN] Executando nó de análise de padrões")
            return await self.pattern_agent.analyze(state)
        return pattern_node

    def _should_continue(self, state: GraphState) -> str:
        """Decide se o workflow deve continuar ou terminar."""
        print("\n[WORKFLOW] Verificando continuação do workflow")
        
        # Verifica número máximo de iterações
        if state["iteration_count"] >= 3:
            print("[WORKFLOW] Máximo de iterações atingido")
            return "end"
            
        # Verifica erros excessivos
        if state.get("error_count", 0) >= 2:
            print("[WORKFLOW] Máximo de erros atingido")
            return "end"
            
        # Verifica progresso
        if state["session_progress"].get("overall_progress", 0) >= 1.0:
            print("[WORKFLOW] Objetivo de aprendizado atingido")
            return "end"
            
        print("[WORKFLOW] Continuando workflow")
        return "continue"

    async def _handle_error(
        self,
        state: GraphState,
        error: Exception
    ) -> GraphState:
        """Manipula erros em qualquer nó do grafo."""
        print(f"\n[ERROR] Erro no nó {state['current_node']}: {str(error)}")
        
        new_state = state.copy()
        new_state["error_count"] = state.get("error_count", 0) + 1
        
        # Adiciona mensagem de erro ao histórico
        error_message = AIMessage(
            content="Desculpe, encontrei um erro ao processar sua solicitação. "
                   "Vou tentar uma abordagem diferente."
        )
        new_state["messages"] = list(state["messages"]) + [error_message]
        new_state["chat_history"] = list(state["chat_history"]) + [error_message]
        
        # Registra erro para análise
        await self._log_error(state["current_node"], error, state)
        
        return new_state

    async def _log_error(
        self,
        node: str,
        error: Exception,
        state: GraphState
    ) -> None:
        """Registra erros para análise posterior."""
        try:
            error_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "node": node,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "state_snapshot": {
                    "current_node": state["current_node"],
                    "iteration_count": state["iteration_count"],
                    "error_count": state.get("error_count", 0)
                }
            }
            
            await self.config.db_handler.insert_one(
                collection="error_logs",
                document=error_log
            )
        except Exception as e:
            print(f"[ERROR] Erro ao registrar log: {str(e)}")

    async def invoke(
        self,
        query: str,
        context: dict
    ) -> Dict[str, Any]:
        """
        Executa o workflow do tutor.
        
        Args:
            query: Pergunta do usuário
            context: Contexto da sessão
            
        Returns:
            Dict contendo resultados da execução
        """
        try:
            print(f"\n[WORKFLOW] Iniciando execução para query: {query}")
            
            # Inicializa estado
            initial_state = self._initialize_state(query, context)
            
            # Executa workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            print("[WORKFLOW] Execução concluída com sucesso")
            return {
                "success": True,
                "messages": final_state["messages"],
                "analysis": final_state.get("analysis_results", {}),
                "metrics": {
                    "learning": final_state["learning_metrics"],
                    "progress": final_state["session_progress"],
                    "engagement": final_state["engagement_data"]
                }
            }
            
        except Exception as e:
            print(f"[WORKFLOW] Erro fatal na execução: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "messages": [
                    AIMessage(
                        content="Desculpe, encontrei um erro ao processar sua pergunta. "
                               "Por favor, tente novamente."
                    )
                ]
            }

    def _initialize_state(
        self,
        query: str,
        context: dict
    ) -> GraphState:
        """Inicializa o estado do workflow."""
        return GraphState(
            messages=[HumanMessage(content=query)],
            current_plan="",
            user_profile=context.get("user_profile", {}),
            extracted_context="",
            next_step=None,
            iteration_count=0,
            chat_history=context.get("chat_history", []),
            learning_metrics=context.get("learning_metrics", {}),
            session_progress=context.get("session_progress", {
                "current_step": "",
                "completed_steps": [],
                "step_metrics": {},
                "overall_progress": 0.0
            }),
            engagement_data=context.get("engagement_data", {
                "interaction_count": 0,
                "average_response_time": 0.0,
                "session_duration": 0,
                "focus_periods": []
            }),
            performance_metrics=context.get("performance_metrics", {
                "accuracy": 0.0,
                "completion_rate": 0.0,
                "learning_velocity": 0.0,
                "retention_rate": 0.0
            }),
            current_node="",
            error_count=0
        )