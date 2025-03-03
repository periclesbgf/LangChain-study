import time
from typing import Annotated, Any
import json
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from agent_system.tools.teach_tools import RetrievalTools
#from agent_system.tools.web_search_tools import WebSearchTools
from agent_system.managers.progress_manager import StudyProgressManager
from agent_system.managers.memory_manager import MemoryManager
from agent_system.states.common_states import AgentState, UserProfile, ExecutionStep
from agent_system.prompts.teach_agent import create_react_prompt


def update_state(state: AgentState, new_state: Dict[str, Any]) -> AgentState:
    for key, value in new_state.items():
        state[key] = value
    return state

def filter_chat_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage):
            try:
                content = json.loads(msg.content)
                if isinstance(content, dict):
                    if content.get("type") == "multimodal":
                        filtered_content = {
                            "type": "multimodal",
                            "content": content["content"]
                        }
                        filtered_messages.append(AIMessage(content=filtered_content["content"]))
                    else:
                        filtered_messages.append(msg)
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
        content = msg.content
        if isinstance(content, str):
            formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history)




class TutorReActAgent():
    def __init__(
        self,
        qdrant_handler,
        student_email: str,
        disciplina: str,
        session_id: str,
        image_collection
    ):
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.progress_manager = StudyProgressManager()
        self.state = AgentState(
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
            session_id=session_id,
            thoughts="",
            memories=[],
            actions_history=[],
            thoughts_history=[],
        )
        self.retrieval_tools = RetrievalTools(
            qdrant_handler=qdrant_handler,
            student_email=student_email,
            disciplina=disciplina,
            session_id=session_id,
            image_collection=image_collection,
            state=self.state
        )

        self.memory_manager = MemoryManager(

        )
        self.think_node =  self.create_think_node(self.state, self.retrieval_tools, self.progress_manager)

        self.workflow = self.create_workflow()

    def create_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node(START,self.think_node)
        graph.add_conditional_edges(START, self.retrieval_tools)
        graph.add_conditional_edges(
            "route_after_plan",
            self._route_after_plan,
            {
                "action": "retrieve_context",
                "web_search": "web_search",
                "direct_answer": "direct_answer"
            }
        )
        graph.add_edge(END, self.react_node)
        graph.set_entry_point(START)
        return graph.compile()

    #async def create_memory_node(memory_manager: MemoryManager):


    async def create_retrieval_node(tools: RetrievalTools):
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


    async def direct_answer(self, state: AgentState, retrieval_tools: RetrievalTools, progress_manager: StudyProgressManager):
        prompt = create_react_prompt()
        model = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)
        chat_history = format_chat_history(state["chat_history"], max_messages=5)
        state["chat_history"] = chat_history

        print("\n[NODE:REACT] Iniciando execução do nó ReAct")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:REACT] Processando a pergunta: {latest_question}")

        try:
            stream = model.astream(prompt.format(**state))
            async for chunk in stream:
                if hasattr(chunk, "image") and chunk.image is not None:
                    yield {
                        "type": "image",
                        "content": chunk.content,
                        "image": chunk.image
                    }
                else:
                    yield {"type": "chunk", "content": chunk.content}
                state["messages"].append(chunk)
        except Exception as e:
            print(f"[NODE:REACT] Erro: {e}")
            yield {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua pergunta: {str(e)}"}

    def _route_after_plan(self) -> AgentState:
        try:
            latest_question = [m for m in self.state["messages"] if isinstance(m, HumanMessage)][-1].content
            if self.state["next_step"] not in ["websearch", "retrieval", "direct_answer"]:
                self.state["next_step"] = "think"

            return self.state["next_step"]

        except Exception as e:
            #print(f"[ROUTING] Error in routing: {str(e)}")
            # Em caso de erro, usamos uma lógica simples baseada em palavras-chave
            question_lower = latest_question.lower()
            if any(keyword in question_lower for keyword in ["youtube", "video", "wikipedia", "web"]):
                next_step = "websearch"
            else:
                next_step = "retrieval"

            #print(f"[ROUTING] Fallback routing decision: {next_step}")
            self.state["next_step"] = next_step
            return next_step

