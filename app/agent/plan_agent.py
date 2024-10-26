from typing import TypedDict,List,Sequence,Dict,Any,Optional
from datetime import datetime,timezone
from langgraph.graph import END,Graph
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
import motor.motor_asyncio
from pymongo import errors
import json
from database.mongo_database_manager import MongoDatabaseManager

class ActivityResource(BaseModel):
    tipo:str
    descricao:str
    url:str=Field(default="")

class Activity(BaseModel):
    descricao:str
    tipo:str
    formato:str

class PlanStep(BaseModel):
    titulo:str
    duracao:str
    descricao:str
    conteudo:List[str]
    recursos:List[ActivityResource]
    atividade:Activity
    progresso:int

class StudyPlan(BaseModel):
    plano_execucao:List[PlanStep]
    duracao_total:str
    progresso_total:int
    created_at:datetime

class PlanningState(TypedDict):
    messages:Sequence[BaseMessage]
    study_plan:dict
    user_profile:dict
    next_step:str|None
    review_feedback:str|None
    id_sessao:str
    update_status:bool|None

class DatabaseUpdateTool:
    def __init__(self, db_manager: 'MongoDatabaseManager'):
        self.db_manager = db_manager

    async def update_study_plan(self, id_sessao: str, plan_data: dict) -> bool:
        try:
            print("[DEBUG]: Preparing update data", plan_data)
            
            # Prepare update data preserving existing fields
            update_data = {
                "plano_execucao": plan_data.get("plano_execucao", []),
                "duracao_total": plan_data.get("duracao_total", ""),
                "updated_at": datetime.now(timezone.utc)
            }
            
            print("[DEBUG]: Update data", update_data)
            success = await self.db_manager.update_study_plan(id_sessao, update_data)
            print("[DEBUG]: Update success status", success)
            return success
        except Exception as e:
            print(f"[DEBUG]: Error updating study plan: {e}")
            return False

class SessionPlanWorkflow:
    def __init__(self, db_manager: MongoDatabaseManager):
        self.db_manager = db_manager
        self.db_tool = DatabaseUpdateTool(db_manager)
        self.workflow = self.create_workflow()

    def create_workflow(self) -> Graph:
        print("[DEBUG]: Creating workflow")
        workflow = Graph()
        workflow.add_node("generate_plan", self.create_planning_node())
        workflow.add_node("review_plan", self.create_review_node())
        workflow.add_node("update_database", self.create_database_node())
        workflow.add_edge("generate_plan", "review_plan")
        workflow.add_edge("review_plan", "update_database")
        workflow.add_edge("update_database", END)
        workflow.set_entry_point("generate_plan")
        compiled_workflow = workflow.compile()
        print("[DEBUG]: Workflow compiled")
        return compiled_workflow

    def create_planning_node(self):
        PLANNING_PROMPT = """Você é um especialista em planejamento educacional que cria planos de estudo detalhados e personalizados.
        O plano deve conter instruções claras e específicas para guiar uma LLM a ensinar um tópico específico.
        A LLM vai seguir o plano para criar uma sessão de estudo eficaz para um aluno.

        Tema da Sessão: {topic}
        
        Estilo de Aprendizagem do Aluno:
        {learning_style}

        Crie um plano de estudo detalhado que:
        1. Seja adaptado ao estilo de aprendizagem específico do aluno
        2. Divida o conteúdo em etapas claras e gerenciáveis
        3. Inclua atividades e recursos apropriados ao estilo do aluno
        4. Foque em ensinar o tópico de forma eficaz e envolvente
        5. Seja específico para o tema da programação, incluindo exemplos práticos e exercícios de código

        Regras importantes:
        1. SEMPRE retorne um JSON válido
        2. Siga EXATAMENTE a estrutura fornecida
        3. Adapte o conteúdo ao estilo de aprendizagem do aluno
        4. Todas as durações devem somar o total especificado em duracao_total
        5. SEMPRE inclua sobre qual assunto abordar em cada seção
        6. Foque em programação e desenvolvimento de software
        7. Inclua exemplos de código quando apropriado

        O plano deve seguir EXATAMENTE esta estrutura JSON:
        INICIO JSON:
            "plano_execucao": [
                
                    "titulo": "Título da seção",
                    "duracao": "XX minutos",
                    "descricao": "Descrição detalhada sobre qual assunto abordar",
                    "conteudo": ["Item 1", "Item 2"],
                    "recursos": [
                        
                            "tipo": "Tipo do recurso",
                            "descricao": "Descrição do recurso",
                            "url": "URL opcional"
                        
                    ],
                    "atividade": 
                        "descricao": "Descrição da atividade",
                        "tipo": "Tipo da atividade",
                        "formato": "Formato da atividade"
                    ,
                    "progresso": XX
                
            ],
            "duracao_total": "60 minutos",
            "progresso_total": 0
        FIM JSON

        Adapte as atividades e recursos ao estilo de aprendizagem:
        - Para aluno Visual: Priorize recursos visuais como videos
        - Para aluno Verbal: Foque em explicações escritas e discussões
        - Para aluno Ativo: Inclua exercícios práticos e experimentação
        - Para aluno Reflexivo: Adicione momentos de análise e reflexão
        - Para aluno Sequencial: Organize em passos pequenos e conectados
        - Para aluno Global: Forneça visão geral e conexões com outros temas

        Retorne APENAS o JSON do plano, sem explicações adicionais."""

        prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
        model = ChatOpenAI(model="gpt-4o", temperature=0.2)

        def generate_plan(state: PlanningState) -> PlanningState:
            print("[DEBUG]: Generating plan with state", state)
            
            topic = state["messages"][0].content if state["messages"] else "Tema não especificado"
            learning_style = state["user_profile"].get("EstiloAprendizagem", {})
            
            print("[DEBUG]: Learning style", learning_style)
            print("[DEBUG]: Topic", topic)
            
            response = model.invoke(prompt.format(
                learning_style=learning_style,
                topic=topic
            ))
            
            print("[DEBUG]: Model response", response)
            cleaned_content = response.content.strip("```json\n").strip("```").strip()
            
            try:
                plan_dict = json.loads(cleaned_content)
                print("[DEBUG]: Parsed plan", plan_dict)
                new_state = state.copy()
                new_state["study_plan"] = plan_dict
                return new_state
            except json.JSONDecodeError as e:
                print(f"[DEBUG]: JSONDecodeError: {e}")
                print("[DEBUG]: Failed content", cleaned_content)
                raise
            except Exception as e:
                print(f"[DEBUG]: Error parsing plan: {e}")
                raise

        return generate_plan

    def create_review_node(self):
        REVIEW_PROMPT = """Você é um especialista em educação responsável por revisar e validar planos de estudo.
        O plano de estudos que você receberá será implementado por uma LLM para criar uma sessão de estudo eficaz e interativa.
        Sugira feedbacks que melhorem e foquem no ensino do tópico de forma eficaz e envolvente.
        
        Analise criticamente o seguinte plano:
        {plan}
        
        Considerando o estilo de aprendizagem do aluno:
        {learning_style}
        
        Verifique:
        1. Se os títulos são claros e bem estruturados
        2. Se as descrições são informativas e adequadas
        3. Se as atividades estão alinhadas com o estilo de aprendizagem
        4. Se a distribuição do tempo é apropriada
        5. Se os recursos são relevantes e bem escolhidos
        6. Se contém instruções de ensino claras e detalhadas
        7. Se os exemplos e exercícios de programação são apropriados
        
        Forneça um feedback detalhado no seguinte formato JSON:
        
            "status": "approved" ou "needs_revision",
            "feedback": "Seu feedback detalhado aqui",
            "suggestions": [
                "Sugestão 1",
                "Sugestão 2"
            ]
        """

        prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT)
        model = ChatOpenAI(model="gpt-4o", temperature=0.3)

        def review_plan(state: PlanningState) -> PlanningState:
            print("[DEBUG]: Reviewing plan", state["study_plan"])
            
            response = model.invoke(prompt.format(
                plan=state["study_plan"],
                learning_style=state["user_profile"].get("EstiloAprendizagem", {})
            ))
            
            cleaned_content = response.content.strip("```json\n").strip("```").strip()
            
            print("[DEBUG]: Review response", response)
            new_state = state.copy()
            
            try:
                feedback_dict = json.loads(cleaned_content)
                print("[DEBUG]: Parsed feedback", feedback_dict)
                new_state["review_feedback"] = feedback_dict
            except:
                print("[DEBUG]: Error parsing review response")
                new_state["review_feedback"] = {"status": "error", "feedback": cleaned_content}
            
            return new_state

        return review_plan

    def create_database_node(self):
        async def update_database(state: PlanningState) -> PlanningState:
            print("[DEBUG]: Updating database with state", state)
            new_state = state.copy()
            
            if not state.get("study_plan"):
                print("[DEBUG]: No study plan found, skipping update")
                new_state["update_status"] = False
                return new_state

            success = await self.db_tool.update_study_plan(
                state["id_sessao"],
                state["study_plan"]
            )
            
            print("[DEBUG]: Database update success", success)
            new_state["update_status"] = success
            return new_state
            
        return update_database

    async def create_session_plan(self, topic: str, student_profile: dict, id_sessao: str) -> dict:
        print("[DEBUG]: Creating session plan for topic:", topic)
        print("[DEBUG]: Student profile:", student_profile)
        print("[DEBUG]: Session ID:", id_sessao)
        
        if not topic:
            return {
                "error": "Tópico não fornecido",
                "plan": None,
                "feedback": None,
                "update_status": False,
                "id_sessao": id_sessao
            }

        initial_state = PlanningState(
            messages=[HumanMessage(content=topic)],
            study_plan={},
            user_profile=student_profile,
            next_step=None,
            review_feedback=None,
            id_sessao=id_sessao,
            update_status=None
        )

        try:
            result = await self.workflow.ainvoke(initial_state)
            print("[DEBUG]: Workflow result", result)

            final_plan = result["study_plan"]
            final_plan["created_at"] = datetime.now(timezone.utc).isoformat()
            print("[DEBUG]: Final plan", final_plan)

            return {
                "plan": final_plan,
                "feedback": result["review_feedback"],
                "update_status": result["update_status"],
                "id_sessao": id_sessao
            }
        except Exception as e:
            print(f"[DEBUG]: Error creating session plan: {e}")
            return {
                "error": f"Erro na criação do plano: {str(e)}",
                "plan": None,
                "feedback": None,
                "update_status": False,
                "id_sessao": id_sessao
            }