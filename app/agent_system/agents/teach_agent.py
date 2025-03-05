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
        self.state = AgentState( #trocar para ser preenchida antes
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
        model = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)
        prompt = create_react_prompt()

        async def think(state: AgentState) -> AgentState:
            state["iteration_count"] += 1
            print(f"[NODE:THINK] Iteration count: {state['iteration_count']}")
            # Make sure streaming_results exists
            if "streaming_results" not in state:
                state["streaming_results"] = []

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
            print(f"[NODE:THINK] Pergunta: {prompt_variables['question']}")
            print(f"[NODE:THINK] Ferramentas: {prompt_variables['tools']}")
            print(f"[NODE:THINK] Pensamentos: {prompt_variables['thoughts']}")
            print(f"[NODE:THINK] Observações: {prompt_variables['observations']}")
            # Start timing the response
            start_time = time.time()

            # Add initial processing message to state
            state["streaming_results"].append(
                {"type": "processing", "content": "Analisando sua pergunta..."}
            )

            # Variables to track response type and accumulated content
            is_final_answer = False
            accumulated_response = ""
            response_content = {}
            needs_tools = False

            try:
                # Use non-streaming model first to avoid async generator issues
                response = await model.ainvoke(prompt.format(**prompt_variables))
                content = response.content if hasattr(response, "content") else str(response)

                # Check raw content for the final_answer pattern first
                is_final_answer = '"type": "final_answer"' in content

                # Process the response to get structured content
                response_content = self._process_model_response(content)
                print(f"[NODE:THINK] Processed response: {response_content}")

                # If the pattern wasn't found but the type is final_answer, still mark it
                if not is_final_answer and response_content.get("type") == "final_answer":
                    is_final_answer = True

                # Check if we need tools
                needs_tools = response_content.get("type") == "call_tool"

                # If this is a final answer, add chunks to streaming_results
                if is_final_answer:
                    print("[NODE:THINK] Detected final_answer type")

                    # Get the final answer text
                    answer_text = response_content.get("answer", "")

                    # Split into chunks and add to streaming_results
                    for i in range(0, len(answer_text), 15):
                        chunk = answer_text[i:i+15]
                        state["streaming_results"].append(
                            {"type": "chunk", "content": chunk}
                        )
                    
                    # Add completion message
                    state["streaming_results"].append(
                        {"type": "complete", "content": f"Resposta completa em {time.time() - start_time:.2f}s."}
                    )
            
            except Exception as e:
                print(f"[NODE:THINK] Error in processing response: {e}")
                state["streaming_results"].append(
                    {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua pergunta: {str(e)}"}
                )
            
            # Create new state with updated information
            new_state = state.copy()
            
            # Update state with thinking results
            new_state["thoughts"] = response_content
            new_state["thoughts_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(), 
                "thought": response_content
            })
            print(f"[NODE:THINK] Updated thoughts: {response_content}")
            print(f"[NODE:THINK] Updated thoughts history: {new_state['thoughts_history']}")
            
            print(f"[NODE:THINK] Needs tools: {needs_tools}")
            
            # Set next step based on whether tools are needed
            if needs_tools:
                new_state["next_step"] = "tools"
            else:
                new_state["next_step"] = "direct_answer"
                # If it was a final answer, store it in state
                if is_final_answer and "answer" in response_content:
                    new_state["final_answer"] = response_content.get("answer", "")
                else:
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
                        print(f"[NODE:TOOLS] Retrieval tool result: {tool_result}")
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
        
        async def react_node(state: AgentState) -> AgentState:
            print("[NODE:REACT] Starting react node execution")
            # Check if we already have a final answer from thinking node
            final_answer = state.get("final_answer")
            
            # Create a list to store streaming results for pickup in run()
            if "streaming_results" not in state:
                state["streaming_results"] = []
                
            start_time = time.time()
            
            # Record that we're processing
            state["streaming_results"].append(
                {"type": "processing", "content": "Processando resposta final..."}
            )
            
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
                    "tools": list[self.retrieval_tools.parallel_context_retrieval],
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
                    # Get the full response first (non-streaming) to avoid async iterator issues
                    response = await model.ainvoke(prompt.format(**prompt_variables))
                    content = response.content if hasattr(response, "content") else str(response)
                    
                    # Check if this is a final answer by looking for the pattern
                    is_final_answer = '"type": "final_answer"' in content
                    
                    # Parse the response if possible
                    try:
                        parsed_response = self._process_model_response(content)
                        if not is_final_answer and parsed_response.get("type") == "final_answer":
                            is_final_answer = True
                    except Exception:
                        # If parsing fails, just proceed with the raw content
                        pass
                        
                    # Add final content to messages
                    state["messages"].append(AIMessage(content=content))
                    
                    # Manually split into chunks to simulate streaming
                    for i in range(0, len(content), 15):
                        chunk = content[i:i+15]
                        # Add to state for pickup in run()
                        state["streaming_results"].append(
                            {"type": "chunk", "content": chunk}
                        )
                    
                    # Send completion message
                    state["streaming_results"].append(
                        {"type": "complete", "content": f"Resposta completa em {time.time() - start_time:.2f}s."}
                    )
                        
                except Exception as e:
                    print(f"[NODE:REACT] Error generating response after tools: {e}")
                    state["streaming_results"].append(
                        {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua pergunta: {str(e)}"}
                    )
            else:
                # We already have a final answer from the thinking node, so stream it directly
                try:
                    # Get the final answer
                    if isinstance(final_answer, dict) and "answer" in final_answer:
                        content = final_answer["answer"]
                    else:
                        content = str(final_answer)
                        
                    # Split into chunks to simulate streaming
                    for i in range(0, len(content), 15):
                        chunk = content[i:i+15]
                        # Add to state for pickup in run()
                        state["streaming_results"].append(
                            {"type": "chunk", "content": chunk}
                        )
                    
                    # Record the streamed response in messages
                    state["messages"].append(AIMessage(content=content))
                    
                    # Send completion message
                    state["streaming_results"].append(
                        {"type": "complete", "content": f"Resposta completa em {time.time() - start_time:.2f}s."}
                    )
                    
                except Exception as e:
                    print(f"[NODE:REACT] Error streaming direct answer: {e}")
                    state["streaming_results"].append(
                        {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua resposta: {str(e)}"}
                    )
            
            # Return the updated state
            return state
                
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
        # Record start time to measure overall execution
        start_time = time.time()
        
        # Initialize streaming results storage in the state
        self.state["streaming_results"] = []
        
        # Prepare the initial state with the new question
        self.state["messages"].append(HumanMessage(content=question))
        self.state["chat_history"].append(HumanMessage(content=question))
        
        # Send initial processing message
        yield {"type": "processing", "content": "Iniciando processamento..."}
        
        # Keep track of whether we've streamed results
        has_streamed_result = False
        
        # Keep track of which streaming results we've already processed
        processed_results = set()
        last_results_count = 0
        
        # Run the workflow
        try:
            print(f"[AGENT:RUN] Starting workflow execution for question: {question[:50]}...")
            execution_count = 0
            
            async for event in self.workflow.astream(self.state):
                execution_count += 1
                # Get the node name that just finished execution
                node_name = event.get("node")
                print(f"[AGENT:RUN] Node {execution_count} completed: {node_name}")
                
                # Print any available keys in the event for debugging
                event_keys = list(event.keys())
                print(f"[AGENT:RUN] Event keys: {event_keys}")
                
                # Log state streaming results count
                if "streaming_results" in self.state:
                    print(f"[AGENT:RUN] Current streaming results count: {len(self.state['streaming_results'])}")
                
                # Process any new streaming results in the state
                # This will catch results from ALL nodes since they all update the same state
                if "streaming_results" in self.state:
                    current_count = len(self.state["streaming_results"])
                    
                    # Check if we have new results
                    if current_count > last_results_count:
                        print(f"[AGENT:RUN] New streaming results detected: {current_count - last_results_count} new items")
                        
                        # Process only new results
                        for i in range(last_results_count, current_count):
                            result = self.state["streaming_results"][i]
                            
                            # Generate a unique identifier for this result
                            result_id = f"{i}:{result.get('type', 'unknown')}:{result.get('content', '')[:20]}"
                            
                            # Only yield results we haven't processed before
                            if result_id not in processed_results:
                                processed_results.add(result_id)
                                has_streamed_result = True
                                yield result
                                print(f"[AGENT:RUN] Yielded result from {node_name}: {result.get('type')}")
                        
                        # Update our counter
                        last_results_count = current_count
                
                # For direct answer node, we need special handling for backward compatibility
                if node_name == "direct_answer" and "result" in event and hasattr(event["result"], "__aiter__"):
                    try:
                        # This is likely a direct generator, so we need to stream results directly
                        async for chunk in event["result"]:
                            if isinstance(chunk, dict) and "type" in chunk:
                                has_streamed_result = True
                                yield chunk
                                print(f"[AGENT:RUN] Yielded direct result: {chunk.get('type')}")
                    except Exception as e:
                        print(f"[AGENT:RUN] Error streaming direct output: {e}")
                        yield {"type": "error", "content": f"Erro ao processar resposta: {str(e)}"}
        except Exception as e:
            print(f"[AGENT:RUN] ERROR IN WORKFLOW EXECUTION: {str(e)}")
            yield {"type": "error", "content": f"Erro na execução do workflow: {str(e)}"}
            return  # Exit if workflow fails
        
        # Process any final streaming results that might have been added after the last event
        if "streaming_results" in self.state:
            current_count = len(self.state["streaming_results"])
            if current_count > last_results_count:
                for i in range(last_results_count, current_count):
                    result = self.state["streaming_results"][i]
                    result_id = f"{i}:{result.get('type', 'unknown')}:{result.get('content', '')[:20]}"
                    if result_id not in processed_results:
                        processed_results.add(result_id)
                        has_streamed_result = True
                        yield result
                        print(f"[AGENT:RUN] Yielded final result: {result.get('type')}")
        
        # If no streaming results were yielded, send a complete message
        if not has_streamed_result:
            yield {"type": "complete", "content": f"Processamento concluído em {time.time() - start_time:.2f}s."}
        
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
                print("[ROUTING] Valid decision made after thinking")
                print(f"[ROUTING] Next step: {next_step}")
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