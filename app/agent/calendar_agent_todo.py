from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
from datetime import datetime, timedelta
import json
import pytz
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

@dataclass
class MessagesState:
    """State for tracking messages in the conversation."""
    messages: List[BaseMessage]
    current_input: Optional[str] = None
    text_response: Optional[str] = None
    operation_result: Optional[Dict] = None
    user_email: Optional[str] = None
    response_plan: Optional[Dict] = None
    datetime_context: Optional[Dict] = None

class DateTools:
    """Tools for date operations"""
    def __init__(self):
        self.tz = pytz.timezone('America/Sao_Paulo')

    def get_current_datetime(self) -> Dict:
        """Get current date and time in Brazil timezone"""
        try:
            now = datetime.now(self.tz)
            week_start = now - timedelta(days=now.weekday())
            week_end = week_start + timedelta(days=6)
            tomorrow = now + timedelta(days=1)
            yesterday = now - timedelta(days=1)
            
            return {
                "status": "success",
                "current_date": now.strftime('%Y-%m-%d'),
                "current_time": now.strftime('%H:%M:%S'),
                "iso_datetime": now.isoformat(),
                "weekday": now.strftime('%A'),
                "formatted": now.strftime('%d/%m/%Y %H:%M'),
                "reference_dates": {
                    "today": now.strftime('%Y-%m-%d'),
                    "tomorrow": tomorrow.strftime('%Y-%m-%d'),
                    "yesterday": yesterday.strftime('%Y-%m-%d'),
                    "week_start": week_start.strftime('%Y-%m-%d'),
                    "week_end": week_end.strftime('%Y-%m-%d')
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao obter data atual: {str(e)}"
            }

class CalendarTools:
    """Tools for calendar operations"""
    def __init__(self, calendar_controller):
        self.calendar_controller = calendar_controller
        self.date_tools = DateTools()
        
        self.tools = [
            Tool(
                name="get_all_events",
                description="Recupera todos os eventos do calendário do usuário",
                func=self.get_all_events
            ),
            Tool(
                name="create_event",
                description="Cria um novo evento no calendário",
                func=self.create_event
            ),
            Tool(
                name="update_event",
                description="Atualiza um evento existente no calendário",
                func=self.update_event
            ),
            Tool(
                name="delete_event",
                description="Remove um evento do calendário",
                func=self.delete_event
            )
        ]

    async def get_all_events(self, user_email: str) -> Dict:
        """Get all calendar events for user"""
        try:
            print(f"Fetching calendar events for user: {user_email}")
            events = self.calendar_controller.get_all_events(user_email)
            print(f"Found {len(events)} events for user {user_email}")
            print(events)

            if not events:
                return {
                    "status": "success",
                    "events": []
                }

            return {
                "status": "success",
                "events": events
            }
        except HTTPException as e:
            return {
                "status": "error",
                "message": str(e.detail)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao recuperar eventos: {str(e)}"
            }

    async def create_event(self, data: str, convert_timezone: bool = False) -> Dict:
        """Create a new calendar event"""
        try:
            print(f"Creating event with data: {data}")
            parsed_data = json.loads(data)
            event_data = parsed_data["event_data"]
            user_email = parsed_data["user_email"]

            # Map incoming keys to expected format
            title = event_data.get("title") or event_data.get("Titulo")
            description = event_data.get("description") or event_data.get("Descricao", "")
            start_time = event_data.get("start_time") or event_data.get("Inicio")
            end_time = event_data.get("end_time") or event_data.get("Fim")
            location = event_data.get("location") or event_data.get("Local", "")

            if not title:
                return {
                    "status": "error",
                    "message": "Título do evento é obrigatório"
                }

            if not start_time or not end_time:
                return {
                    "status": "error",
                    "message": "Horário de início e fim são obrigatórios"
                }

            # Convert string to datetime if needed
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except ValueError:
                    return {
                        "status": "error",
                        "message": "Formato de data/hora de início inválido"
                    }

            if isinstance(end_time, str):
                try:
                    end_time = datetime.fromisoformat(end_time)
                except ValueError:
                    return {
                        "status": "error",
                        "message": "Formato de data/hora de fim inválido"
                    }

            print(f"Creating event for user {user_email}:")
            print(f"Title: {title}")
            print(f"Start time (before conversion): {start_time}")
            print(f"End time (before conversion): {end_time}")
            
            # Only convert timezone if flag is True
            if not convert_timezone:
                # Format datetime to string without timezone conversion
                start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Use controller's timezone conversion
                start_time_str = self.calendar_controller.convert_to_brasilia_time(start_time)
                end_time_str = self.calendar_controller.convert_to_brasilia_time(end_time)
                
            print(f"Final start time: {start_time_str}")
            print(f"Final end time: {end_time_str}")
            
            result = self.calendar_controller.create_event(
                title=title,
                description=description,
                start_time=start_time_str,
                end_time=end_time_str,
                location=location,
                current_user=user_email,
                convert_timezone=convert_timezone
            )
            
            return {
                "status": "success",
                "message": "Evento criado com sucesso",
                "event_id": result["IdEvento"]
            }
        except HTTPException as e:
            return {
                "status": "error",
                "message": str(e.detail)
            }
        except Exception as e:
            print(f"Error creating event: {str(e)}")
            print(f"Event data: {event_data}")
            return {
                "status": "error",
                "message": f"Erro ao criar evento: {str(e)}"
            }

    async def update_event(self, data: str) -> Dict:
        """Update an existing calendar event"""
        try:
            parsed_data = json.loads(data)
            event_id = parsed_data["event_id"]
            updated_data = parsed_data["event_data"]
            user_email = parsed_data["user_email"]

            # Extract updateable fields
            title = updated_data.get("Titulo")
            description = updated_data.get("Descricao")
            start_time = updated_data.get("Inicio")
            end_time = updated_data.get("Fim")
            location = updated_data.get("Local")

            success = self.calendar_controller.update_event(
                event_id=event_id,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                location=location,
                current_user=user_email
            )
            
            return {
                "status": "success" if success else "error",
                "message": "Evento atualizado com sucesso" if success else "Falha ao atualizar evento"
            }
        except HTTPException as e:
            return {
                "status": "error",
                "message": str(e.detail)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao atualizar evento: {str(e)}"
            }

    async def delete_event(self, data: str) -> Dict:
        """Delete a calendar event"""
        try:
            parsed_data = json.loads(data)
            event_id = parsed_data["event_id"]
            user_email = parsed_data["user_email"]

            success = self.calendar_controller.delete_event(event_id, user_email)
            return {
                "status": "success" if success else "error",
                "message": "Evento deletado com sucesso" if success else "Falha ao deletar evento"
            }
        except HTTPException as e:
            return {
                "status": "error",
                "message": str(e.detail)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao deletar evento: {str(e)}"
            }

class ResponsePlanner:
    """Plans the response using available tools and context"""
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        self.prompt = ChatPromptTemplate.from_template("""
        Você é um assistente de calendário responsável por analisar entradas do usuário e planejar respostas.
        
        Entrada do usuário:
        {user_input}
        
        Contexto temporal:
        {datetime_context}
        
        Forneça um plano de resposta no formato JSON com a seguinte estrutura:
            "intent": "get|create|update|delete",
            "required_tools": ["tool1", "tool2"],
            "parameters":
                "event_data": ,
            "response_format":
                "structure": "list",
                "datetime_format": "d/m/Y às H:M"
        
        Lembre-se:
        - Analise cuidadosamente a intenção do usuário
        - Identifique todas as ferramentas necessárias
        - Extraia os parâmetros relevantes
        - Defina um formato de resposta apropriado
        
        Responda APENAS com o JSON, sem explicações adicionais.
        """)

    async def create_plan(self, user_input: str, datetime_context: Dict) -> Dict:
        try:
            response = await self.model.ainvoke(
                self.prompt.format(
                    user_input=user_input,
                    datetime_context=json.dumps(datetime_context, indent=2, ensure_ascii=False)
                )
            )
            
            # Use json.loads() instead of eval()
            try:
                plan = json.loads(response.content)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, try to extract JSON from the response
                # Look for content between curly braces
                json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
                plan = json.loads(json_str)
            
            # Add datetime context to the plan
            plan["datetime_context"] = datetime_context
            return plan
            
        except Exception as e:
            print(f"[ERROR] Planning error: {str(e)}")
            # Return default plan in case of error
            return {
                "intent": "get",
                "required_tools": ["get_all_events"],
                "datetime_context": datetime_context,
                "parameters": {"event_data": {}},
                "response_format": {
                    "structure": "list",
                    "datetime_format": "%d/%m/%Y às %H:%M"
                }
            }

class CalendarAgent:
    """Agent that processes text input and interacts with calendar tools."""
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        Você é um assistente de calendário em português responsável por responder perguntas relacionadas ao calendário do usuário.
        
        Contexto Temporal:
        {datetime_context}
        
        Plano de Resposta:
        {response_plan}

        Eventos do Usuário:
        {user_events}
        
        Resposte a seguinte solicitação do usuário considerando o contexto fornecido:
        {text_input}
        
        Lembre-se:
        - Confirme as ações importantes
        - Seja amigável e prestativo
        - Leve em consideração apenas eventos agendados

        ATENÇÃO:
            - O formato do horario é formato de 24 horas
            - Sua resposta será transformada em áudio, então certifique-se de que a resposta seja o mais natural possível.
            - TODO O CONTEXTO SE REMETE AO USUÁRIO QUE ESTA INTERAGINDO COM O ASSISTENTE

        Formato de resposta:
            - Texto corrido com a resposta ao usuário.
        """)

    async def process(self, state: MessagesState, calendar_tools: CalendarTools) -> MessagesState:
        try:
            print(f"[DEBUG] Processing message: {state.current_input}")
            print(f"[DEBUG] Response plan: {state.response_plan}")
            
            # Execute calendar operation based on intent
            if state.response_plan["intent"] == "get":
                operation_result = await calendar_tools.get_all_events(state.user_email)
            elif state.response_plan["intent"] == "create":
                operation_result = await calendar_tools.create_event(json.dumps({
                    "event_data": state.response_plan["parameters"]["event_data"],
                    "user_email": state.user_email
                }))
            elif state.response_plan["intent"] == "update":
                operation_result = await calendar_tools.update_event(json.dumps({
                    "event_id": state.response_plan["parameters"]["event_id"],
                    "event_data": state.response_plan["parameters"]["event_data"],
                    "user_email": state.user_email
                }))
            elif state.response_plan["intent"] == "delete":
                operation_result = await calendar_tools.delete_event(json.dumps({
                    "event_id": state.response_plan["parameters"]["event_id"],
                    "user_email": state.user_email
                }))
            else:
                raise ValueError(f"Intent não suportada: {state.response_plan['intent']}")

            print(f"[DEBUG] Operation result: {operation_result}")
            
            # Generate AI response using all context
            response = await self.model.ainvoke(
                self.prompt.format(
                    datetime_context=json.dumps(state.datetime_context, indent=2, ensure_ascii=False),
                    response_plan=json.dumps(state.response_plan, indent=2, ensure_ascii=False),
                    text_input=state.current_input,
                    user_events=json.dumps(operation_result, indent=2, ensure_ascii=False)
                )
            )
            
            text_response = response.content
            print(f"[DEBUG] AI response: {text_response}")
            
            # Update conversation history
            new_messages = state.messages + [
                HumanMessage(content=state.current_input),
                AIMessage(content=text_response)
            ]
            print(f"[DEBUG] New messages: {new_messages}")
            return MessagesState(
                messages=new_messages,
                current_input=state.current_input,
                text_response=text_response,
                operation_result=operation_result,
                user_email=state.user_email,
                response_plan=state.response_plan,
                datetime_context=state.datetime_context,
            )
            
        except Exception as e:
            error_msg = f"Erro no processamento: {str(e)}"
            print(f"[ERROR] {error_msg}")
            
            return MessagesState(
                messages=state.messages + [
                    HumanMessage(content=state.current_input),
                    AIMessage(content="Desculpe, ocorreu um erro ao processar sua solicitação.")
                ],
                text_response="Desculpe, ocorreu um erro ao processar sua solicitação.",
                operation_result={"status": "error", "message": str(e)},
                user_email=state.user_email,
                response_plan=state.response_plan,
                datetime_context=state.datetime_context
            )


class CalendarOrchestrator:
    """Orchestrates the calendar interaction flow."""
    def __init__(self, calendar_controller):
        print("[INIT] Initializing CalendarOrchestrator...")
        self.calendar_tools = CalendarTools(calendar_controller)
        self.date_tools = DateTools()
        self.planner = ResponsePlanner()
        self.agent = CalendarAgent()
        self.workflow = self._create_workflow()
        print("[INIT] CalendarOrchestrator initialized")

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(MessagesState)

        async def plan_wrapper(state: MessagesState) -> MessagesState:
            print("[DEBUG] Planning response...")
            datetime_context = self.date_tools.get_current_datetime()
            plan = await self.planner.create_plan(state.current_input, datetime_context)
            print(f"[DEBUG] Response plan: {plan}")
            return MessagesState(
                messages=state.messages,
                current_input=state.current_input,
                user_email=state.user_email,
                response_plan=plan,
                datetime_context=datetime_context
            )

        async def process_wrapper(state: MessagesState) -> MessagesState:
            print("[DEBUG] Processing message...")
            response = await self.agent.process(state, self.calendar_tools)
            print(f"[DEBUG] Processed response: {response.text_response}")
            return response

        workflow.add_node("plan_response", plan_wrapper)
        workflow.add_node("process_message", process_wrapper)

        workflow.add_edge(START, "plan_response")
        workflow.add_edge("plan_response", "process_message")
        workflow.add_edge("process_message", END)
        return workflow.compile()

    async def process_input(self, text_input: str, user_email: str) -> str:
        """Process text input and return response."""
        try:
            print(f"\n[INFO] Processing new input from {user_email}: {text_input}")

            if not text_input or not user_email:
                raise ValueError("Missing required input parameters")

            initial_state = MessagesState(
                messages=[],
                current_input=text_input,
                user_email=user_email,
                response_plan=None,
                datetime_context=None,
                operation_result=None,
                text_response=None
            )

            final_state = await self.workflow.ainvoke(initial_state)

            return final_state



        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                "error": str(e),
                "text_response": "Desculpe, ocorreu um erro ao processar sua solicitação.",
                "messages": []
            }