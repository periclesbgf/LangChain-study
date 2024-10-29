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