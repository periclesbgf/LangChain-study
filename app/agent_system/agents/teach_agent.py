import asyncio
import time
from typing import Annotated, Any
import json
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END, Graph
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from agent_system.tools.teach_tools import RetrievalTools
#from agent_system.tools.web_search_tools import WebSearchTools
from agent_system.managers.progress_manager import StudyProgressManager
#from agent_system.managers.memory_manager import MemoryManager
from agent_system.states.common_states import AgentState, UserProfile, ExecutionStep
from agent_system.prompts.teach_prompt import create_react_prompt


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

        self.think_node = self.create_think_node()
        self.tools_node = self.create_tools_node()
        self.react_node = self.create_react_node()

        self.workflow = self.create_workflow()

    def create_workflow(self) -> StateGraph:
        # Create a dictionary for state instead of using TypedDict directly
        # Create the graph with the dictionary state
        graph = StateGraph(AgentState)
        graph.add_node("think", self.think_node)
        graph.add_node("tools", self.tools_node)
        graph.add_node("retrieve_context", self.create_retrieval_node(self.retrieval_tools))
        graph.add_node("direct_answer", self.react_node)
        
        # Define edges
        graph.add_conditional_edges(
            "think",
            self._route_after_thinking,
            {
                "tools": "tools",
                "direct_answer": "direct_answer"
            }
        )
        graph.add_edge("tools", "think")
        graph.add_edge("direct_answer", END)

        graph.set_entry_point("think")
        return graph.compile()

    def create_think_node(self):
        """
        Creates the thinking node for the ReAct agent.
        This node uses the predefined ReAct prompt to analyze the question and determine
        if tools are needed or if a direct answer can be provided.
        """
        model = ChatOpenAI(model="gpt-4o", temperature=0.1)
        prompt = create_react_prompt()

        async def think(state: AgentState) -> AgentState:
            latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
            print(f"[NODE:THINK] Processing message: {latest_message.content}")
            question = latest_message.content

            # Create formatted user profile for prompt
            user_profile = state.get("user_profile", {})
            print(f"[NODE:THINK] User profile: {user_profile}")
            if not user_profile:
                # Default profile if empty
                user_profile = {
                    "Nome": "Estudante",
                    "EstiloAprendizagem": {
                        "percepcao": "Sensorial",
                        "entrada": "Visual",
                        "processamento": "Ativo",
                        "entendimento": "Sequencial"
                    }
                }
            
            # Prepare execution step context if available
            current_step = state.get("current_progress", {})
            print(f"[NODE:THINK] Current step: {current_step}")
            if not current_step:
                current_step = {
                    "titulo": "Aprendizado Geral",
                    "descricao": "Explorando conceitos gerais da disciplina",
                    "progresso": 0
                }
            
            # Format chat history
            chat_history = format_chat_history(state.get("chat_history", []), max_messages=5)
            
            # Prepare the prompt variables for the ReAct prompt
            prompt_variables = {
                "nome": user_profile.get("Nome", "Estudante"),
                "percepcao": user_profile.get("EstiloAprendizagem", {}).get("percepcao", "Sensorial"),
                "entrada": user_profile.get("EstiloAprendizagem", {}).get("entrada", "Visual"),
                "processamento": user_profile.get("EstiloAprendizagem", {}).get("processamento", "Ativo"),
                "entendimento": user_profile.get("EstiloAprendizagem", {}).get("entendimento", "Sequencial"),
                "titulo": current_step.get("titulo", ""),
                "descricao": current_step.get("descricao", ""),
                "progresso": current_step.get("progresso", 0),
                "last_message": chat_history,
                "question": question,
                "tools": list[self.retrieval_tools.parallel_context_retrieval],
                "thoughts": state.get("thoughts", ""),
                "observations": state.get("observations", "")
            }
            print(f"[NODE:THINK] Nome: {prompt_variables['nome']}")
            print(f"[NODE:THINK] Percepção: {prompt_variables['percepcao']}")
            print(f"[NODE:THINK] Entrada: {prompt_variables['entrada']}")
            print(f"[NODE:THINK] Processamento: {prompt_variables['processamento']}")
            print(f"[NODE:THINK] Entendimento: {prompt_variables['entendimento']}")
            print(f"[NODE:THINK] Título: {prompt_variables['titulo']}")
            print(f"[NODE:THINK] Descrição: {prompt_variables['descricao']}")
            print(f"[NODE:THINK] Progresso: {prompt_variables['progresso']}")
            print(f"[NODE:THINK] Última mensagem: {prompt_variables['last_message']}")
            print(f"[NODE:THINK] Pergunta: {prompt_variables['question']}")
            print(f"[NODE:THINK] Ferramentas: {prompt_variables['tools']}")
            print(f"[NODE:THINK] Pensamentos: {prompt_variables['thoughts']}")
            print(f"[NODE:THINK] Observações: {prompt_variables['observations']}")
            # Invoke model with the ReAct prompt
            response = await model.ainvoke(prompt.format(**prompt_variables))
            print(f"[NODE:THINK] Response: {response.content}")
            # Parse the response to determine if tools are needed
            print(f"[NODE:THINK] Parsing response")
            response_content = self._process_model_response(response.content)
            
            # Check if we need to use tools based on response
            needs_tools = False
            try:
                # Try to parse as JSON to check if there's an action
                print(f"[NODE:THINK] response_content: {response_content}")
                response_type = response_content.get("type")
                print(f"[NODE:THINK] Response type: {response_type}")
                if response_type == "call_tool":
                    needs_tools = True
            except Exception as e:
                print(f"[NODE:THINK] Error parsing response: {e}")

            # Update state with thinking and decision
            new_state = state.copy()
            new_state["thoughts"] = response_content
            new_state["thoughts_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(), 
                "thought": response_content
            })
            print(f"[NODE:THINK] Needs tools: {needs_tools}")
            # Set next step based on whether tools are needed
            if needs_tools:
                new_state["next_step"] = "tools"
            else:
                new_state["next_step"] = "direct_answer"
                new_state["final_answer"] = response_content

            return new_state

        return think



    def create_tools_node(self):
        """
        Creates the tools node that executes tools based on the agent's request.
        """
        async def tools_node(state: AgentState) -> AgentState:
            # Get the thinking output that contains the tool request
            thought = state.get("thoughts", "")
            print(f"[NODE:TOOLS] Processing thought: {thought}")
            tool_result = None
            tool_input = None
            tool_name = None
            
            try:
                # Try to parse as JSON to extract tool info
                try:
                    parsed_thought = json.loads(thought)
                    print(f"[NODE:TOOLS] Parsed thought: {parsed_thought}")
                    if "action" in parsed_thought and isinstance(parsed_thought["action"], dict):
                        tool_name = parsed_thought["action"].get("name")
                        tool_input = parsed_thought["action"].get("input")
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract using string parsing
                    if "action:" in thought.lower():
                        # Extract action name after "action:"
                        action_parts = thought.lower().split("action:")
                        if len(action_parts) > 1:
                            # Extract potential tool name from the text
                            tool_line = action_parts[1].strip().split("\n")[0]
                            tool_name = tool_line.strip()
                    
                    # Try to extract input after "input:"
                    if "input:" in thought.lower():
                        input_parts = thought.lower().split("input:")
                        if len(input_parts) > 1:
                            # Extract potential input from the text
                            tool_input = input_parts[1].strip().split("\n")[0]
                
                # Execute tool if we have both name and input
                if tool_name and tool_input:
                    # Currently we only have the parallel_context_retrieval tool
                    if "context" in tool_name.lower() or "retrieval" in tool_name.lower():
                        # Execute the retrieval tool
                        self.retrieval_tools.state = state
                        tool_result = await self.retrieval_tools.parallel_context_retrieval(tool_input)
                    else:
                        tool_result = {"error": f"Tool '{tool_name}' not implemented"}
                else:
                    tool_result = {"error": "Could not extract tool name or input from request"}
            
            except Exception as e:
                print(f"[TOOLS] Error executing tool: {e}")
                tool_result = {"error": str(e)}
            
            # Record the tool usage in actions_history
            new_state = state.copy()
            new_state["actions_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": tool_name or "unknown_tool",
                "input": tool_input,
                "result": tool_result
            })
            
            # Add the tool result to extracted_context if it's retrieval
            if tool_result and not isinstance(tool_result, str) and not tool_result.get("error"):
                new_state["extracted_context"] = tool_result
            
            return new_state
            
        return tools_node
    
    def create_react_node(self):
        """
        Creates the ReAct node that handles streaming the final response to the user.
        In case of direct answer, it will stream the existing answer.
        If coming from tools, it will generate a new response incorporating tool results.
        """
        model = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)
        prompt = create_react_prompt()
        
        async def react_node(state: AgentState):
            # Check if we already have a final answer from thinking node
            final_answer = state.get("final_answer")
            
            # If we have a tool execution result, we need to generate a new response
            if state.get("next_step") == "tools" or not final_answer:
                # Get context from tool execution if available
                extracted_context = state.get("extracted_context", {})
                
                # Format chat history
                chat_history = format_chat_history(state.get("chat_history", []), max_messages=5)
                latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
                question = latest_message.content
                
                # Prepare execution step context
                current_step = state.get("current_progress", {})
                if not current_step:
                    current_step = {
                        "titulo": "Aprendizado Geral",
                        "descricao": "Explorando conceitos gerais da disciplina",
                        "progresso": 0
                    }
                
                # Prepare the prompt variables with tool results included
                prompt_variables = {
                    "nome": state.get("user_profile", {}).get("Nome", "Estudante"),
                    "percepcao": state.get("user_profile", {}).get("EstiloAprendizagem", {}).get("percepcao", "Sensorial"),
                    "entrada": state.get("user_profile", {}).get("EstiloAprendizagem", {}).get("entrada", "Visual"),
                    "processamento": state.get("user_profile", {}).get("EstiloAprendizagem", {}).get("processamento", "Ativo"),
                    "entendimento": state.get("user_profile", {}).get("EstiloAprendizagem", {}).get("entendimento", "Sequencial"),
                    "titulo": current_step.get("titulo", ""),
                    "descricao": current_step.get("descricao", ""),
                    "progresso": current_step.get("progresso", 0),
                    "last_message": chat_history,
                    "question": question,
                    "tools": str(self.retrieval_tools.parallel_context_retrieval.__doc__),
                    "thoughts": state.get("thoughts", ""),
                    "observations": state.get("observations", "")
                }
                
                # Add a special observation field if we have tool results
                tool_result = next((a["result"] for a in state.get("actions_history", []) 
                                   if isinstance(a, dict) and a.get("action") != "unknown_tool"), None)
                
                if tool_result:
                    # Add observations from tool execution to the state
                    state["observations"] = f"Resultado da ferramenta: {json.dumps(tool_result, ensure_ascii=False)}"
                
                try:
                    # Stream the response
                    stream = model.astream(prompt.format(**prompt_variables))
                    async for chunk in stream:
                        # Handle image content if present
                        if hasattr(chunk, "image") and chunk.image is not None:
                            yield {
                                "type": "image",
                                "content": chunk.content,
                                "image": chunk.image
                            }
                        else:
                            yield {"type": "chunk", "content": chunk.content}
                        
                        # Update state with the response chunks
                        state["messages"].append(chunk)
                        
                except Exception as e:
                    print(f"[NODE:REACT] Error generating response after tools: {e}")
                    yield {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua pergunta: {str(e)}"}
            else:
                # We already have a final answer from the thinking node, so stream it directly
                try:
                    # Split the final answer into chunks to simulate streaming
                    chunks = [final_answer[i:i+15] for i in range(0, len(final_answer), 15)]
                    
                    for chunk in chunks:
                        yield {"type": "chunk", "content": chunk}
                        await asyncio.sleep(0.01)  # Small delay to simulate streaming
                        
                    # Record the streamed response in messages
                    state["messages"].append(AIMessage(content=final_answer))
                    
                except Exception as e:
                    print(f"[NODE:REACT] Error streaming direct answer: {e}")
                    yield {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua resposta: {str(e)}"}
                
        return react_node
    
    def create_retrieval_node(self, tools: RetrievalTools):
        """
        Creates a node for context retrieval.
        """
        async def retrieve_context(state: AgentState) -> AgentState:
            #print("\n[NODE:RETRIEVAL] Starting retrieval node execution")
            latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
            #print(f"[NODE:RETRIEVAL] Processing message: {latest_message.content}")

            tools.state = state
            context_results = await tools.parallel_context_retrieval(latest_message.content)

            new_state = state.copy()
            new_state["extracted_context"] = context_results
            new_state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            # Record the retrieval action in actions_history
            new_state["actions_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "context_retrieved",
                "input": latest_message.content,
                "result": "Context retrieved successfully"
            })
            
            #print(f"[NODE:RETRIEVAL] Updated iteration count: {new_state['iteration_count']}")
            return new_state

        return retrieve_context


    async def run(self, question: str):
        """
        Run the ReAct agent with the given question.
        Returns a streaming response.
        """
        # Prepare the initial state with the new question
        self.state["messages"].append(HumanMessage(content=question))
        self.state["chat_history"].append(HumanMessage(content=question))
        
        # Run the workflow
        async for event in self.workflow.astream(self.state):
            # Stream node output if it's the react_node (direct_answer)
            node_name = event.get("node")
            if node_name == "direct_answer":
                # Pass through the streaming chunks from the react node
                async for chunk in event["result"]:
                    yield chunk
                    
            # For other nodes, we can log their execution but don't need to yield their results
            elif node_name in ["think", "tools", "retrieve_context"]:
                print(f"[AGENT] Executed node: {node_name}")
        
        # Update progress after workflow completes
        #await self.progress_manager.update_step_progress(self.state.get("session_id", ""), 5)

    def _route_after_thinking(self, state: AgentState) -> str:
        """
        Routes to the next node based on the thinking node's decision.
        """
        try:
            next_step = state["next_step"]
            print(f"[ROUTING] Next step after thinking: {next_step}")
            if next_step in ["retrieve_context", "tools", "direct_answer"]:
                return next_step
            else:
                # Default to direct_answer if no valid decision was made
                return "direct_answer"
        except Exception as e:
            print(f"[ROUTING] Error in routing after thinking: {str(e)}")
            # Fallback to direct_answer in case of error
            return "direct_answer"


    def _process_model_response(self, content: str) -> Dict[str, Any]:
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
            json_content = json.loads(cleaned_content)
            #print(f"[PLANNING] Processed model response: {plan}")
            return json_content

        except json.JSONDecodeError as e:
            #print(f"[PLANNING] Error parsing model response: {e}")
            raise ValueError("Invalid JSON in model response")

    def validate_plan_structure(plan: Dict[str, Any]) -> None:
        """Valida a estrutura do plano gerado."""
        required_fields = [
            "type",
            "thought",
        ]

        missing_fields = [field for field in required_fields if field not in plan]
        if missing_fields:
            raise ValueError(f"Missing required fields in plan: {missing_fields}")

    # async def add_steps_to_state(self, plan: Dict[str, Any]) -> None:
    #     """Adiciona os passos do plano ao estado do agente."""
    #     #print(f"[PLANNING] Adding steps to state: {plan}")
    #     # Adicionar os passos do plano ao estado do agente
    #     for step in plan["steps