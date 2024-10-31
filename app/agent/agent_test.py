from typing import Any, TypedDict, List, Dict, Optional, Union
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
from database.vector_db import QdrantHandler, TextSplitter
from dataclasses import dataclass
import base64
import asyncio
from typing import Tuple, Dict, Any

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
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
        print(f"[RETRIEVAL] Initializing RetrievalTools:")
        print(f"[RETRIEVAL] - Student: {student_email}")
        print(f"[RETRIEVAL] - Disciplina: {disciplina}")
        print(f"[RETRIEVAL] - Session: {session_id}")
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.image_collection = image_collection
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.state = state
        
        self.QUESTION_TRANSFORM_PROMPT = """
        Você é um especialista em transformar perguntas para melhorar a recuperação de contexto.
        
        hitorico da conversa: {chat_history}

        Pergunta original: {question}
        Tipo de busca: {search_type}
        
        Para o tipo de busca '{search_type}', reescreva a pergunta para maximizar a recuperação de informações relevantes.
        Mantenha o foco em {focus_points}.
        
        O usuario pode fazer perguntas que remetam a perguntas anteriores, então é importante analisar o histórico da conversa.

        Retorne apenas a pergunta reescrita, sem explicações adicionais.
        """
        
        self.RELEVANCE_ANALYSIS_PROMPT = """
        Analise a relevância dos contextos recuperados para a pergunta do usuário.
        
        Pergunta: {question}
        
        Contextos recuperados:
        Texto: {text_context}
        Imagem: {image_context}
        Tabela: {table_context}
        
        Para cada contexto, avalie a relevância em uma escala de 0 a 1 e explique brevemente por quê.
        Retorne um JSON no formato:
        {{
            "text": {{"score": 0.0, "reason": "string"}},
            "image": {{"score": 0.0, "reason": "string"}},
            "table": {{"score": 0.0, "reason": "string"}},
            "recommended_context": "text|image|table|combined"
        }}
        
        Mantenha o formato JSON exato e use apenas aspas duplas.
        """

    async def transform_question(self, question: str, search_type: str) -> str:
        """
        Transforma a pergunta para melhor recuperação de contexto baseado no tipo de busca.
        """
        focus_points = {
            "text": "conceitos, definições e explicações textuais",
            "image": "elementos visuais, diagramas e representações gráficas",
            "table": "dados estruturados, estatísticas e comparações numéricas"
        }
        
        # Formata o histórico de chat das últimas 4 mensagens
        chat_history = self.state["chat_history"][-4:] if self.state["chat_history"] else []
        formatted_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in chat_history
        ])
        
        print(f"[RETRIEVAL] Using chat history for question transformation:")
        print(formatted_history)
        
        prompt = ChatPromptTemplate.from_template(self.QUESTION_TRANSFORM_PROMPT)
        response = await self.model.ainvoke(prompt.format(
            chat_history=formatted_history,
            question=question,
            search_type=search_type,
            focus_points=focus_points[search_type]
        ))
        
        transformed_question = response.content.strip()
        print(f"[RETRIEVAL] Transformed question for {search_type}: {transformed_question}")
        return transformed_question

    async def parallel_context_retrieval(self, question: str) -> Dict[str, Any]:
        """
        Executa buscas paralelas por diferentes tipos de contexto.
        """
        print(f"\n[RETRIEVAL] Starting parallel context retrieval for: {question}")
        
        # Transform questions for each context type
        text_question, image_question, table_question = await asyncio.gather(
            self.transform_question(question, "text"),
            self.transform_question(question, "image"),
            self.transform_question(question, "table")
        )
        
        # Execute parallel context retrieval
        text_context, image_context, table_context = await asyncio.gather(
            self.retrieve_text_context(text_question),
            self.retrieve_image_context(image_question),
            self.retrieve_table_context(table_question)
        )
        
        # Analyze relevance
        relevance_analysis = await self.analyze_context_relevance(
            original_question=question,
            text_context=text_context,
            image_context=image_context,
            table_context=table_context
        )
        
        return {
            "text": text_context,
            "image": image_context,
            "table": table_context,
            "relevance_analysis": relevance_analysis
        }

    async def retrieve_text_context(self, query: str) -> str:
        """Recupera contexto textual."""
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "text"}
            )
            return "\n".join([doc.page_content for doc in results]) if results else ""
        except Exception as e:
            print(f"[RETRIEVAL] Error in text retrieval: {e}")
            return ""

    async def retrieve_image_context(self, query: str) -> Dict[str, Any]:
        """Recupera contexto de imagem."""
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "image"}
            )
            
            if not results:
                return {"type": "image", "content": None, "description": ""}
                
            image_uuid = results[0].metadata.get("image_uuid")
            if not image_uuid:
                return {"type": "image", "content": None, "description": ""}
                
            return await self.retrieve_image_and_description(image_uuid)
        except Exception as e:
            print(f"[RETRIEVAL] Error in image retrieval: {e}")
            return {"type": "image", "content": None, "description": ""}

    async def retrieve_table_context(self, query: str) -> Dict[str, Any]:
        """Recupera contexto de tabela."""
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "table"}
            )
            
            if not results:
                return {"type": "table", "content": None}
                
            return {
                "type": "table",
                "content": results[0].page_content,
                "metadata": results[0].metadata
            }
        except Exception as e:
            print(f"[RETRIEVAL] Error in table retrieval: {e}")
            return {"type": "table", "content": None}

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        """
        Recupera a imagem e sua descrição de forma assíncrona.
        """
        try:
            print(f"[RETRIEVAL] Recuperando imagem com UUID: {image_uuid}")
            image_data = await self.image_collection.find_one({"_id": image_uuid})

            if not image_data:
                print(f"[RETRIEVAL] Imagem não encontrada: {image_uuid}")
                return {"type": "error", "message": "Imagem não encontrada"}

            image_bytes = image_data.get("image_data")
            if not image_bytes:
                print("[RETRIEVAL] Dados da imagem ausentes")
                return {"type": "error", "message": "Dados da imagem ausentes"}

            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                print(f"[RETRIEVAL] Formato de imagem não suportado: {type(image_bytes)}")
                return {"type": "error", "message": "Formato de imagem não suportado"}
            
            results = self.qdrant_handler.similarity_search_with_filter(
                query="",
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                k=1,
                specific_metadata={"image_uuid": image_uuid, "type": "image"}
            )

            return {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content if results else ""
            }

        except Exception as e:
            print(f"[RETRIEVAL] Error retrieving image: {e}")
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
        """
        Analisa a relevância dos diferentes contextos recuperados.
        """
        try:
            if not text_context and not image_context and not table_context:
                print("[RETRIEVAL] No context available for relevance analysis")
                return self._get_default_analysis()

            prompt = ChatPromptTemplate.from_template(self.RELEVANCE_ANALYSIS_PROMPT)
            
            # Prepare context descriptions
            image_description = image_context.get("description", "") if image_context else ""
            table_content = table_context.get("content", "") if table_context else ""
            
            # Truncate long contexts to prevent token limits
            text_preview = text_context[:500] + "..." if len(text_context) > 500 else text_context
            image_preview = image_description[:500] + "..." if len(image_description) > 500 else image_description
            table_preview = table_content[:500] + "..." if len(table_content) > 500 else table_content
            


            response = await self.model.ainvoke(prompt.format(
                question=original_question,
                text_context=text_preview,
                image_context=image_preview,
                table_context=table_preview
            ))
            
            try:
                # Clean up the response to ensure valid JSON
                cleaned_content = response.content.strip()
                # Remove any markdown code block indicators if present
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()
                
                analysis = json.loads(cleaned_content)
                print(f"[RETRIEVAL] Relevance analysis: {analysis}")
                
                # Validate the required fields
                required_fields = ["text", "image", "table", "recommended_context"]
                if not all(field in analysis for field in required_fields):
                    raise ValueError("Missing required fields in analysis")
                
                return analysis
                
            except json.JSONDecodeError as e:
                print(f"[RETRIEVAL] Error parsing relevance analysis: {e}")
                print(f"[RETRIEVAL] Invalid JSON content: {cleaned_content}")
                return self._get_default_analysis()
            except ValueError as e:
                print(f"[RETRIEVAL] Validation error: {e}")
                return self._get_default_analysis()
                
        except Exception as e:
            print(f"[RETRIEVAL] Error in analyze_context_relevance: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Returns a default analysis when the actual analysis fails."""
        return {
            "text": {"score": 0, "reason": "Default fallback due to analysis error"},
            "image": {"score": 0, "reason": "Default fallback due to analysis error"},
            "table": {"score": 0, "reason": "Default fallback due to analysis error"},
            "recommended_context": "combined"
        }

def create_retrieval_node(tools: RetrievalTools):
    async def retrieve_context(state: AgentState) -> AgentState:
        print("\n[NODE:RETRIEVAL] Starting retrieval node execution")
        latest_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        print(f"[NODE:RETRIEVAL] Processing message: {latest_message.content}")
        
        # Atualiza o state no RetrievalTools antes de processar
        tools.state = state
        
        context_results = await tools.parallel_context_retrieval(latest_message.content)

        new_state = state.copy()
        new_state["extracted_context"] = context_results
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        print(f"[NODE:RETRIEVAL] Updated iteration count: {new_state['iteration_count']}")
        return new_state

    return retrieve_context

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
    
    3. RECURSOS E ATIVIDADES (OPCIONAL):
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
        print("[NODE:PLANNING] Loaded current study plan")
        current_step = identify_current_step(plano_execucao)
        print("[NODE:PLANNING] Identified current step")
        chat_history = "\n".join([f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in state["chat_history"][-3:]
        ])
        print("[NODE:PLANNING] Formatted chat history")
        print(chat_history)

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
    CONTEXT_TEACHING_PROMPT = """Você é um tutor personalizado que ajuda os alunos através do pensamento crítico.
    
    Plano de Aprendizado:
    {learning_plan}
    
    Perfil do Aluno:
    {user_profile}
    
    Contextos Disponíveis:
    Texto: {text_context}
    Imagem: {image_context}
    Tabela: {table_context}
    
    Análise de Relevância:
    {relevance_analysis}
    
    Histórico da Conversa:
    {chat_history}
    
    Pergunta:
    {question}
    
    Baseado na análise de relevância, utilize os contextos mais apropriados para criar uma explicação que:
    1. Integre os diferentes tipos de contexto de forma coerente
    2. Priorize o tipo de contexto mais relevante segundo a análise
    3. Adapte a explicação ao estilo de aprendizagem do aluno
    
    Lembre-se: 
    - Se houver contexto de imagem, inclua a descrição da imagem na explicação
    - Se houver contexto de tabela, inclua os dados relevantes na explicação
    - O usuário é LEIGO, então tome a liderança na explicação
    - Responda SEMPRE em português do Brasil de forma clara e objetiva
    - Evite respostas longas e complexas
    - Referencie elementos específicos dos contextos utilizados
    - Faca perguntas para verificar a compreensão do aluno (opcional)
    ATENÇÃO: VOCÊ RESPONDE DIRETAMENTE AO ALUNO, NÃO AO SISTEMA.
    """
    
    context_prompt = ChatPromptTemplate.from_template(CONTEXT_TEACHING_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    def generate_teaching_response(state: AgentState) -> AgentState:
        print("\n[NODE:TEACHING] Starting teaching response generation")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        print(f"[NODE:TEACHING] Processing question: {latest_question}")

        # Filter out multimodal messages from chat history
        filtered_chat_history = []
        for msg in state["chat_history"][-3:]:
            if isinstance(msg, HumanMessage):
                filtered_chat_history.append(msg)
            elif isinstance(msg, AIMessage):
                try:
                    # Try to parse as JSON to check if it's a multimodal message
                    content = json.loads(msg.content)
                    if isinstance(content, dict) and content.get("type") == "multimodal":
                        # Only add the text content to history
                        filtered_chat_history.append(AIMessage(content=content["content"]))
                    else:
                        filtered_chat_history.append(msg)
                except json.JSONDecodeError:
                    # If it's not JSON, it's a regular message
                    filtered_chat_history.append(msg)

        chat_history = "\n".join([
            f"{'Aluno' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
            for m in filtered_chat_history
        ])
        
        contexts = state["extracted_context"]
        relevance = contexts["relevance_analysis"]
        
        try:
            # Prepare image content if available and relevant
            image_context = contexts["image"]
            image_content = None
            if (image_context.get("type") == "image" and 
                image_context.get("image_bytes") and 
                relevance["image"]["score"] > 0.3):
                
                base64_image = base64.b64encode(image_context["image_bytes"]).decode('utf-8')
                image_content = f"data:image/jpeg;base64,{base64_image}"
            
            # Generate explanation using available contexts
            explanation = model.invoke(context_prompt.format(
                learning_plan=state["current_plan"],
                user_profile=state["user_profile"],
                text_context=contexts["text"][:1000] + "..." if len(contexts["text"]) > 1000 else contexts["text"],
                image_context=contexts["image"].get("description", ""),
                table_context=contexts["table"].get("content", ""),
                relevance_analysis=json.dumps(relevance, indent=2),
                question=latest_question,
                chat_history=chat_history
            ))
            
            # Create response with or without image
            if image_content:
                response_content = {
                    "type": "multimodal",
                    "content": explanation.content,
                    "image": image_content
                }
                response = AIMessage(content=json.dumps(response_content))
                # Store only text content in chat history
                history_response = AIMessage(content=explanation.content)
            else:
                response = explanation
                history_response = response
                
        except Exception as e:
            print(f"[NODE:TEACHING] Error generating response: {str(e)}")
            response = AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")
            history_response = response
        
        new_state = state.copy()
        new_state["messages"] = list(state["messages"]) + [response]
        new_state["chat_history"] = list(state["chat_history"]) + [
            HumanMessage(content=latest_question),
            history_response
        ]
        return new_state

    return generate_teaching_response

def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    print("\n[PLAN] Identifying current execution step")
    sorted_steps = sorted(plano_execucao, key=lambda x: x["progresso"])
    
    current_step = next(
        (step for step in sorted_steps if step["progresso"] < 100),
        sorted_steps[-1]
    )
    
    print(f"[PLAN] Selected step: {current_step['titulo']} (Progress: {current_step['progresso']}%)")
    return ExecutionStep(**current_step)

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
    def __init__(self, qdrant_handler, student_email: str, disciplina: str, session_id: str, image_collection):
        print(f"\n[WORKFLOW] Initializing TutorWorkflow")
        print(f"[WORKFLOW] Parameters: student_email={student_email}, disciplina={disciplina}, session_id={session_id}")
        
        initial_state = AgentState(
            messages=[],
            current_plan="",
            user_profile={},
            extracted_context="",
            next_step=None,
            iteration_count=0,
            chat_history=[]
        )
        
        self.tools = RetrievalTools(
            qdrant_handler=qdrant_handler,
            student_email=student_email,
            disciplina=disciplina,
            session_id=session_id,
            image_collection=image_collection,
            state=initial_state
        )
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