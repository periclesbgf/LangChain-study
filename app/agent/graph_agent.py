from typing import TypedDict, Annotated, Sequence, List
from typing_extensions import TypeVar
from langgraph.graph import Graph, END
from langgraph.prebuilt.tool_nodes import ToolExecutor
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.utils.function_calling import convert_to_openai_function
import json

# Define state type
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_plan: str
    user_profile: dict
    extracted_context: str
    next_step: str | None

# Tool definitions
class RetrievalTools:
    def __init__(self, qdrant_handler, student_email, disciplina):
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina

    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant educational context from vector database"""
        try:
            filter_results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                disciplina=self.disciplina,
                k=5
            )

            if filter_results:
                return "\n".join([doc.page_content for doc in filter_results])
            return "No relevant context found."
        except Exception as e:
            return f"Error retrieving context: {str(e)}"

# Node functions
def create_retrieval_node(tools: RetrievalTools):
    """Creates a node for context retrieval"""
    tool_executor = ToolExecutor([
        Tool(
            name="retrieve_context",
            func=tools.retrieve_context,
            description="Retrieve relevant educational context for a given query"
        )
    ])
    
    def retrieve_context(state: AgentState) -> AgentState:
        # Extract the latest query from messages
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        
        # Execute retrieval
        result = tool_executor.invoke({
            "tool_name": "retrieve_context",
            "input": latest_message.content
        })
        
        # Update state with retrieved context
        state["extracted_context"] = result
        return state
    
    return retrieve_context

def create_planning_node():
    """Creates a node for learning plan generation"""
    PLANNING_PROMPT = """You are an educational planning assistant.
    Given the student profile and context, create a step-by-step learning plan that matches their learning style.
    
    Student Profile:
    {user_profile}
    
    Retrieved Context:
    {context}
    
    Current Question:
    {question}
    
    Create a detailed learning plan that:
    1. Matches the student's learning preferences
    2. Breaks down the concept into manageable steps
    3. Provides appropriate examples and exercises
    4. Suggests relevant practice methods
    
    Learning Plan:"""
    
    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    model = ChatOpenAI(temperature=0.7)
    
    def generate_plan(state: AgentState) -> AgentState:
        # Get latest question
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        
        # Generate learning plan
        response = model.invoke(prompt.format(
            user_profile=state["user_profile"],
            context=state["extracted_context"],
            question=latest_question
        ))
        
        state["current_plan"] = response.content
        return state
    
    return generate_plan

def create_teaching_node():
    """Creates a node for generating teaching responses"""
    TEACHING_PROMPT = """You are a personalized tutor helping a student learn through critical thinking.
    Instead of providing direct answers, guide them through the problem-solving process.
    
    Learning Plan:
    {learning_plan}
    
    Student Profile:
    {user_profile}
    
    Current Context:
    {context}
    
    Question:
    {question}
    
    Provide guidance that:
    1. Asks thought-provoking questions
    2. Suggests relevant examples
    3. Points to key concepts
    4. Encourages self-discovery
    
    Your response:"""
    
    prompt = ChatPromptTemplate.from_template(TEACHING_PROMPT)
    model = ChatOpenAI(temperature=0.7)
    
    def generate_teaching_response(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        
        response = model.invoke(prompt.format(
            learning_plan=state["current_plan"],
            user_profile=state["user_profile"],
            context=state["extracted_context"],
            question=latest_question
        ))
        
        # Add response to messages
        state["messages"] = list(state["messages"]) + [AIMessage(content=response.content)]
        return state
    
    return generate_teaching_response

def should_continue(state: AgentState) -> str:
    """Determines if the teaching process should continue"""
    # Check if there are follow-up questions or if the current plan is completed
    latest_messages = state["messages"][-2:]  # Get last human and AI message pair
    
    # Simple heuristic: if the last AI message contains a question, expect more interaction
    if any("?" in msg.content for msg in latest_messages if isinstance(msg, AIMessage)):
        return "continue"
    return "end"

# Main workflow class
class TutorWorkflow:
    def __init__(self, qdrant_handler, student_email, disciplina):
        self.tools = RetrievalTools(qdrant_handler, student_email, disciplina)
        self.workflow = self.create_workflow()
    
    def create_workflow(self) -> Graph:
        # Create nodes
        retrieval_node = create_retrieval_node(self.tools)
        planning_node = create_planning_node()
        teaching_node = create_teaching_node()
        
        # Create workflow
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("retrieve_context", retrieval_node)
        workflow.add_node("generate_plan", planning_node)
        workflow.add_node("teach", teaching_node)
        
        # Add edges
        workflow.add_edge("retrieve_context", "generate_plan")
        workflow.add_edge("generate_plan", "teach")
        workflow.add_conditional_edges(
            "teach",
            should_continue,
            {
                "continue": "retrieve_context",
                "end": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("retrieve_context")
        
        return workflow.compile()
    
    async def invoke(self, query: str, student_profile: dict) -> dict:
        """Invokes the tutor workflow"""
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            current_plan="",
            user_profile=student_profile,
            extracted_context="",
            next_step=None
        )
        
        try:
            result = await self.workflow.ainvoke(initial_state)
            return {
                "messages": result["messages"],
                "final_plan": result["current_plan"]
            }
        except Exception as e:
            return {
                "error": f"Workflow execution failed: {str(e)}",
                "messages": [AIMessage(content="I encountered an error while processing your request.")]
            }