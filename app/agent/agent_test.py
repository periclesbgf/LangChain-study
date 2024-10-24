from typing import TypedDict, Annotated, Sequence, List, Literal
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

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_plan: str
    user_profile: dict
    extracted_context: str
    next_step: str | None
    iteration_count: int
    chat_history: List[BaseMessage]  # Added chat history

class RetrievalTools:
    def __init__(self, qdrant_handler, student_email, disciplina):
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina

    def retrieve_context(self, query: str) -> str:
        try:
            filter_results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                disciplina=self.disciplina,
                k=3  # Reduzido para 3 resultados mais relevantes
            )
            
            if filter_results:
                context = "\n".join([doc.page_content for doc in filter_results])
                return context
                
            return "Nenhum contexto relevante encontrado."
            
        except Exception as e:
            return "Nenhum contexto relevante encontrado."

def create_retrieval_node(tools: RetrievalTools):
    def retrieve_context(state: AgentState) -> AgentState:
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        
        result = tools.retrieve_context(latest_message.content)
        
        new_state = state.copy()
        new_state["extracted_context"] = result
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        return new_state

    return retrieve_context

def create_planning_node():
    PLANNING_PROMPT = """Você é um assistente educacional que cria planos de aprendizado personalizados.
    Com base no perfil do aluno e no contexto, crie um plano de aprendizado que se adapte ao estilo de aprendizagem dele.

    Perfil do Aluno:
    {user_profile}
    
    Plano de Aprendizado:
    {learning_plan}
    
    Pergunta Atual:
    {question}
    
    Histórico da Conversa:
    {chat_history}
    
    Crie um plano detalhado que:
    1. Se adapte às preferências de aprendizado do aluno
    2. Divida o conceito em etapas gerenciáveis
    3. Forneça exemplos e exercícios apropriados
    4. Sugira métodos práticos relevantes
    
    Plano de Aprendizado:"""
    
    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def generate_plan(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        
        # Format chat history
        chat_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in state["chat_history"][-3:]  # Only last 3 messages
        ])
        
        response = model.invoke(prompt.format(
            user_profile=state["user_profile"],
            learning_plan=state["current_plan"],
            question=latest_question,
            chat_history=chat_history
        ))
        
        new_state = state.copy()
        new_state["current_plan"] = response.content
        print(f"[DEBUG] Generated plan: {response.content}")
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
    
    Forneça orientação que:
    1. Faça perguntas que estimulem o pensamento
    2. Sugira exemplos relevantes
    3. Aponte conceitos-chave
    4. Incentive a autodescoberta
    
    Lembre-se: 
        - O usuario é LEIGO, entao tome a liderança na explicação.
        - Responda SEMPRE em português do Brasil de forma clara e objetiva.
        - evite respostas longas e complexas.
        - Foque em respostas que ajudem o aluno a entender o conceito.
        - Forneca exemlos e exercícios práticos sempre que possível.
    
    Sua resposta:"""
    
    prompt = ChatPromptTemplate.from_template(TEACHING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    def generate_teaching_response(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        
        # Format chat history
        chat_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in state["chat_history"][-3:]  # Only last 3 messages
        ])
        
        response = model.invoke(prompt.format(
            learning_plan=state["current_plan"],
            user_profile=state["user_profile"],
            context=state["extracted_context"],
            question=latest_question,
            chat_history=chat_history
        ))
        
        new_state = state.copy()
        new_state["messages"] = list(state["messages"]) + [AIMessage(content=response.content)]
        new_state["chat_history"] = list(state["chat_history"]) + [
            HumanMessage(content=latest_question),
            AIMessage(content=response.content)
        ]
        return new_state
    
    return generate_teaching_response

def should_continue(state: AgentState) -> str:
    MAX_ITERATIONS = 1  # Reduzido para 1 iteração para resposta mais rápida
    current_iterations = state.get("iteration_count", 0)
    
    if current_iterations >= MAX_ITERATIONS:
        return "end"
    
    # Simplified continue check
    return "end"  # Always end after first iteration

class TutorWorkflow:
    def __init__(self, qdrant_handler, student_email, disciplina):
        self.tools = RetrievalTools(qdrant_handler, student_email, disciplina)
        self.workflow = self.create_workflow()
    
    def create_workflow(self) -> Graph:
        retrieval_node = create_retrieval_node(self.tools)
        planning_node = create_planning_node()
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
        return workflow.compile()
    
    async def invoke(self, query: str, student_profile: dict, current_plan=None, chat_history=None) -> dict:
        """
        Invokes the tutor workflow
        :param query: User's question
        :param student_profile: Student profile data
        :param current_plan: Current learning plan (if any)
        :param chat_history: List of messages from CustomMongoDBChatMessageHistory
        """
        print(f"[DEBUG] Invoking workflow with query: {query}")
        
        # Convert MongoDB history to a list if it's not already
        if chat_history is None:
            chat_history = []
        elif not isinstance(chat_history, list):
            chat_history = list(chat_history)  # Convert to list if it's a cursor
            
        # Limit history to last 5 interactions for context
        recent_history = chat_history[-10:]  
            
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
            print("[DEBUG] Starting workflow execution")
            result = await self.workflow.ainvoke(initial_state)
            print("[DEBUG] Workflow execution completed")
            
            return {
                "messages": result["messages"],
                "final_plan": result["current_plan"],
                "chat_history": result["chat_history"]
            }
        except Exception as e:
            print(f"[DEBUG] Workflow execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Erro na execução do workflow: {str(e)}",
                "messages": [AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")],
                "chat_history": recent_history
            }