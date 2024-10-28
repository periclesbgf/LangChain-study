from typing import TypedDict, List, Dict, Optional, Any, Union
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, Graph
from youtubesearchpython import VideosSearch
import wikipediaapi
from langchain.tools import Tool
import json

from agent.tools import DatabaseUpdateTool
from database.mongo_database_manager import MongoDatabaseManager
from database.vector_db import QdrantHandler

@dataclass
class ExecutionStep:
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[Dict]
    atividade: Dict
    progresso: int

@dataclass
class WorkflowState:
    messages: List[BaseMessage]
    current_plan: str  # JSON string of the plan
    user_profile: Dict
    extracted_context: str
    next_step: Optional[str]
    iteration_count: int
    chat_history: List[BaseMessage]
    execution_step: Dict
    step_feedback: Dict
    session_id: str


class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_plan: str
    user_profile: dict
    extracted_context: str
    next_step: str | None
    iteration_count: int
    chat_history: List[BaseMessage]
    execution_step: ExecutionStep
    step_feedback: Dict
    session_id: str

class SearchTools:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='pt',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='EducationalTutor/1.0'
        )
        
        self.tools = [
            Tool(
                name="Youtube Search",
                description="Útil para encontrar vídeos explicativos sobre um tópico",
                func=self.search_youtube
            ),
            Tool(
                name="Wikipedia Search",
                description="Útil para encontrar informações conceituais e definições",
                func=self.search_wikipedia
            )
        ]

    def search_youtube(self, query: str) -> Dict:
        try:
            print(f"[DEBUG] Searching YouTube for: {query}")
            videos_search = VideosSearch(query, limit=1)
            results = videos_search.result()

            if not results['result']:
                return {
                    "status": "error",
                    "message": "Nenhum vídeo encontrado.",
                    "data": None
                }

            video_info = results['result'][0]
            return {
                "status": "success",
                "message": "Vídeo encontrado com sucesso",
                "data": {
                    "title": video_info['title'],
                    "link": video_info['link'],
                    "duration": video_info.get('duration', 'N/A'),
                    "views": video_info.get('viewCount', {}).get('text', 'N/A'),
                    "channel": video_info.get('channel', {}).get('name', 'N/A'),
                    "description": video_info.get('descriptionSnippet', [{}])[0].get('text', 'Sem descrição'),
                    "thumbnails": video_info.get('thumbnails', [{}])[0].get('url', None)
                }
            }

        except Exception as e:
            print(f"[ERROR] YouTube search error: {e}")
            return {
                "status": "error",
                "message": f"Erro ao buscar no YouTube: {str(e)}",
                "data": None
            }

    def search_wikipedia(self, query: str) -> Dict:
        try:
            print(f"[DEBUG] Searching Wikipedia for: {query}")
            page = self.wiki_wiki.page(query)
            
            if not page.exists():
                suggestions = self.wiki_wiki.search(query, results=5)
                if not suggestions:
                    return {
                        "status": "error",
                        "message": "Nenhum artigo encontrado.",
                        "data": None
                    }
                page = self.wiki_wiki.page(suggestions[0])

            sections = [
                {"title": section.title, "text": section.text[:200] + "..."}
                for section in page.sections[:3]
            ]

            return {
                "status": "success",
                "message": "Artigo encontrado com sucesso",
                "data": {
                    "title": page.title,
                    "summary": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                    "url": page.fullurl,
                    "sections": sections,
                    "is_disambiguation": page.is_disambiguation,
                    "categories": list(page.categories.keys())[:5],
                    "referencias": len(page.references),
                    "links_externos": len(page.external_links)
                }
            }

        except Exception as e:
            print(f"[ERROR] Wikipedia search error: {e}")
            return {
                "status": "error",
                "message": f"Erro ao buscar na Wikipedia: {str(e)}",
                "data": None
            }

    def format_video_response(self, video_data: Dict) -> str:
        if video_data["status"] == "error":
            return video_data["message"]
            
        data = video_data["data"]
        return (
            f"📺 Encontrei um vídeo relevante:\n\n"
            f"📗 Título: {data['title']}\n"
            f"🔗 Link: {data['link']}\n"
            f"⏱️ Duração: {data['duration']}\n"
            f"👁️ Visualizações: {data['views']}\n"
            f"🎥 Canal: {data['channel']}\n\n"
            f"📝 Descrição:\n{data['description']}"
        )

    def format_wiki_response(self, wiki_data: Dict) -> str:
        if wiki_data["status"] == "error":
            return wiki_data["message"]
            
        data = wiki_data["data"]
        sections_text = "\n\n".join([
            f"📌 {section['title']}:\n{section['text']}"
            for section in data['sections']
        ])
        
        return (
            f"📚 Artigo da Wikipedia:\n\n"
            f"📗 Título: {data['title']}\n\n"
            f"📝 Resumo:\n{data['summary']}\n\n"
            f"📑 Seções Principais:\n{sections_text}\n\n"
            f"🔗 Link para ler mais: {data['url']}"
        )

class RetrievalAgent:
    def __init__(self, qdrant_handler, student_email: str, disciplina: str):
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina

    async def process(self, state: AgentState) -> AgentState:
        try:
            latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
            
            # Criar query contextualizada
            query = f"{latest_message.content} {state['execution_step'].titulo} {state['execution_step'].descricao}"
            
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                disciplina=self.disciplina,
                k=5
            )
            
            context = "\n".join([doc.page_content for doc in results]) if results else "Nenhum contexto relevante encontrado."
            
            new_state = state.copy()
            new_state["extracted_context"] = context
            return new_state
            
        except Exception as e:
            print(f"[ERROR] Retrieval error: {e}")
            new_state = state.copy()
            new_state["extracted_context"] = "Erro na recuperação de contexto."
            return new_state

class PlanningAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        Você é um assistente educacional que cria planos de resposta adaptados ao perfil do aluno e ao momento atual do plano de execução.

        Perfil do Aluno: {user_profile}
        Etapa Atual: {current_step}
        Progresso Atual: {progress}%
        
        Pergunta do Aluno: {question}
        Contexto Recuperado: {context}
        
        Crie um plano de resposta que:
        1. Avalie o alinhamento da pergunta com a etapa atual
        2. Estruture uma resposta adaptada ao estilo de aprendizagem do aluno
        3. Sugira recursos e atividades práticas
        4. Defina indicadores de compreensão
        
        Responda em JSON:

            "alinhamento": boolean,
            "estrutura_resposta": [{"parte": string, "objetivo": string}],
            "recursos": [string],
            "atividades": string,
            "indicadores": [string]

        """)

    async def process(self, state: AgentState) -> AgentState:
        try:
            question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
            
            response = self.model.invoke(self.prompt.format(
                user_profile=state["user_profile"],
                current_step=state["execution_step"].__dict__,
                progress=state["execution_step"].progresso,
                question=question,
                context=state["extracted_context"]
            ))

            new_state = state.copy()
            new_state["current_plan"] = response.content
            return new_state
            
        except Exception as e:
            print(f"[ERROR] Planning error: {e}")
            raise

class TeachingAgent:
    def __init__(self, search_tools: SearchTools):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        self.search_tools = search_tools
        self.prompt = ChatPromptTemplate.from_template("""
        Você é um tutor personalizado que ajuda os alunos.
        
        Plano: {plan}
        Perfil: {profile}
        Contexto: {context}
        
        Recursos Adicionais:
        Video: {video_info}
        Wikipedia: {wiki_info}
        
        Pergunta: {question}
        
        Histórico: {history}
        
        Lembre-se:
        - Integre os recursos adicionais na sua explicação
        - Use os vídeos e artigos como referência
        - Mantenha o foco no objetivo da etapa atual
        
        Responda em português, de forma clara e objetiva.
        """)

    async def process(self, state: AgentState) -> AgentState:
        try:
            question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
            
            # Buscar recursos adicionais
            video_result = self.search_tools.search_youtube(question)
            wiki_result = self.search_tools.search_wikipedia(question)
            
            video_info = self.search_tools.format_video_response(video_result)
            wiki_info = self.search_tools.format_wiki_response(wiki_result)
            
            history = "\n".join([
                f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
                for m in state["chat_history"][-3:]
            ])

            response = self.model.invoke(self.prompt.format(
                plan=state["current_plan"],
                profile=state["user_profile"],
                context=state["extracted_context"],
                video_info=video_info,
                wiki_info=wiki_info,
                question=question,
                history=history
            ))

            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [AIMessage(content=response.content)]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=question),
                AIMessage(content=response.content)
            ]
            return new_state
            
        except Exception as e:
            print(f"[ERROR] Teaching error: {e}")
            raise

class FeedbackAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        Analise a interação e determine o progresso do aluno:
        
        Etapa Atual: {step}
        Interação: {interaction}
        Histórico: {history}
        
        Avalie:
        1. Compreensão do conteúdo
        2. Engajamento
        3. Necessidade de reforço
        
        Responda em JSON:
        {
            "compreensao": number (0-100),
            "engajamento": number (0-100),
            "precisa_reforco": boolean,
            "aspectos_positivos": [string],
            "aspectos_melhorar": [string],
            "recomendacoes": string
        }
        """)

    async def process(self, state: AgentState) -> AgentState:
        try:
            last_interaction = state["messages"][-2:]
            history = "\n".join([
                f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
                for m in state["chat_history"][-5:]
            ])

            feedback = self.model.invoke(self.prompt.format(
                step=state["execution_step"].__dict__,
                interaction="\n".join([m.content for m in last_interaction]),
                history=history
            ))

            new_state = state.copy()
            new_state["step_feedback"] = json.loads(feedback.content)
            return new_state
            
        except Exception as e:
            print(f"[ERROR] Feedback error: {e}")
            raise

class UpdateAgent:
    def __init__(self, db_tool: DatabaseUpdateTool):
        self.db_tool = db_tool

    async def process(self, state: AgentState) -> AgentState:
        try:
            feedback = state["step_feedback"]
            current_step = state["execution_step"]
            
            # Atualizar progresso
            new_progress = min(
                current_step.progresso + int(feedback["compreensao"] * 0.3),
                100
            )
            
            # Preparar dados para atualização
            execution_plan = json.loads(state["current_plan"])
            for step in execution_plan["plano_execucao"]:
                if step["titulo"] == current_step.titulo:
                    step["progresso"] = new_progress
                    break

            # Atualizar plano no banco de dados
            await self.db_tool.update_study_plan(
                state["session_id"],
                execution_plan
            )

            # Atualizar análise da sessão
            analysis_data = {
                "session_id": state["session_id"],
                "user_email": state["user_profile"]["Email"],
                "comportamental": {
                    "engajamento": feedback["engajamento"],
                    "interesses": feedback["aspectos_positivos"]
                },
                "aprendizado": {
                    "compreensao": feedback["compreensao"],
                    "pontos_fortes": feedback["aspectos_positivos"],
                    "areas_melhorar": feedback["aspectos_melhorar"]
                },
                "engajamento": {
                    "nivel": feedback["engajamento"]
                },
                "recomendacoes": [feedback["recomendacoes"]],
                "metricas": {
                    "progresso": new_progress,
                    "tempo_resposta": 0,  # Será implementado posteriormente
                    "interacoes": len(state["chat_history"]),
                    "recursos_utilizados": len(execution_plan["plano_execucao"])
                }
            }

            await self.db_tool.create_session_analysis(analysis_data)

            # Atualizar memória do usuário
            memory_data = {
                "memoria_curto_prazo": [
                    {
                        "topico": current_step.titulo,
                        "conceitos": feedback["aspectos_positivos"],
                        "nivel_compreensao": feedback["compreensao"],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ],
                "memoria_longo_prazo": [
                    {
                        "topico": current_step.titulo,
                        "status": "em_progresso" if feedback["precisa_reforco"] else "dominado",
                        "pontos_fortes": feedback["aspectos_positivos"],
                        "pontos_melhorar": feedback["aspectos_melhorar"],
                        "ultima_revisao": datetime.now(timezone.utc).isoformat()
                    }
                ] if feedback["compreensao"] > 70 else []
            }
            
            await self.db_tool.update_user_memory(
                state["user_profile"]["Email"],
                memory_data
            )

            new_state = state.copy()
            new_state["execution_step"] = ExecutionStep(
                **{**current_step.__dict__, "progresso": new_progress}
            )
            return new_state
            
        except Exception as e:
            print(f"[ERROR] Update error: {e}")
            raise

class SessionUtils:
    @staticmethod
    def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
        try:
            sorted_steps = sorted(plano_execucao, key=lambda x: x["progresso"])
            current_step = next(
                (step for step in sorted_steps if step["progresso"] < 100),
                sorted_steps[-1]
            )
            return ExecutionStep(**current_step)
        except Exception as e:
            print(f"[ERROR] Error identifying current step: {e}")
            return ExecutionStep(
                titulo="Erro",
                duracao="",
                descricao="Erro ao identificar etapa atual",
                conteudo=[],
                recursos=[],
                atividade={},
                progresso=0
            )

    @staticmethod
    def validate_execution_plan(execution_plan: Dict) -> bool:
        try:
            required_fields = ["plano_execucao", "duracao_total", "progresso_total"]
            return all(field in execution_plan for field in required_fields)
        except Exception as e:
            print(f"[ERROR] Error validating execution plan: {e}")
            return False

    @staticmethod
    def calculate_total_progress(plano_execucao: List[Dict]) -> int:
        try:
            if not plano_execucao:
                return 0
            total = sum(step["progresso"] for step in plano_execucao)
            return total // len(plano_execucao)
        except Exception as e:
            print(f"[ERROR] Error calculating total progress: {e}")
            return 0

    @staticmethod
    def validate_execution_plan(execution_plan: Dict) -> bool:
        required_fields = ["plano_execucao", "duracao_total", "progresso_total"]
        return all(field in execution_plan for field in required_fields)

class TutorOrchestrator:
    def __init__(
        self,
        qdrant_handler: 'QdrantHandler',
        db_manager: 'MongoDatabaseManager',
        student_email: str,
        disciplina: str
    ):
        print(f"[INFO] Initializing TutorOrchestrator for student {student_email} in discipline {disciplina}")
        
        self.search_tools = SearchTools()
        self.db_tool = DatabaseUpdateTool(db_manager)
        self.session_utils = SessionUtils()
        
        self.retrieval_agent = RetrievalAgent(qdrant_handler, student_email, disciplina)
        self.planning_agent = PlanningAgent()
        self.teaching_agent = TeachingAgent(self.search_tools)
        self.feedback_agent = FeedbackAgent()
        self.update_agent = UpdateAgent(self.db_tool)
        
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> Graph:
        """
        Creates the agent workflow graph using a proper state schema.
        """
        # Create workflow with WorkflowState dataclass as schema
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieval_agent.process)
        workflow.add_node("plan", self.planning_agent.process)
        workflow.add_node("teach", self.teaching_agent.process)
        workflow.add_node("feedback", self.feedback_agent.process)
        workflow.add_node("update", self.update_agent.process)
        
        # Define flow
        workflow.add_edge("retrieve", "plan")
        workflow.add_edge("plan", "teach")
        workflow.add_edge("teach", "feedback")
        workflow.add_edge("feedback", "update")
        
        # Add conditionals
        workflow.add_conditional_edges(
            "update",
            self._should_continue,
            {
                "continue": "retrieve",
                "end": END
            }
        )
        
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()

    def _should_continue(self, state: WorkflowState) -> str:
        if state.iteration_count >= 1:
            return "end"
            
        if state.step_feedback.get("compreensao", 0) >= 80:
            return "end"
            
        return "continue"

    async def process_message(
        self,
        message: str,
        session_id: str,
        execution_plan: Union[str, Dict],
        user_profile: Dict,
        chat_history: List[BaseMessage] = None
    ) -> Dict:
        print(f"[INFO] Processing message for session {session_id}")
        
        try:
            # Handle execution_plan whether it's a string or dict
            if isinstance(execution_plan, str):
                execution_plan = json.loads(execution_plan)
            
            if not self.session_utils.validate_execution_plan(execution_plan):
                raise ValueError("Invalid execution plan structure")

            if not chat_history:
                chat_history = []
            
            current_step = self.session_utils.identify_current_step(
                execution_plan["plano_execucao"]
            )
            
            # Create initial state using WorkflowState dataclass
            initial_state = WorkflowState(
                messages=[HumanMessage(content=message)],
                current_plan=json.dumps(execution_plan) if isinstance(execution_plan, dict) else execution_plan,
                user_profile=user_profile,
                extracted_context="",
                next_step=None,
                iteration_count=0,
                chat_history=chat_history[-10:],
                execution_step=asdict(current_step),
                step_feedback={},
                session_id=session_id
            )

            print("[INFO] Starting workflow execution")
            result = await self.workflow.ainvoke(initial_state)
            print("[INFO] Workflow execution completed")
            
            updated_plan = json.loads(result.current_plan)
            updated_plan["progresso_total"] = self.session_utils.calculate_total_progress(
                updated_plan["plano_execucao"]
            )
            
            return {
                "messages": result.messages,
                "execution_plan": updated_plan,
                "chat_history": result.chat_history,
                "feedback": result.step_feedback,
                "current_step": asdict(current_step),
                "context": result.extracted_context
            }
            
        except Exception as e:
            print(f"[ERROR] Error processing message: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "messages": [AIMessage(content="Desculpe, ocorreu um erro. Por favor, tente novamente.")],
                "chat_history": chat_history if chat_history else []
            }