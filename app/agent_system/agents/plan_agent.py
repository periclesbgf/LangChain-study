from typing import TypedDict, List, Sequence, Dict, Any, Optional
from datetime import datetime, timezone
from langgraph.graph import END, Graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import motor.motor_asyncio
from pymongo import errors
import json
from agent.tools import DatabaseUpdateTool

class ActivityResource(BaseModel):
    tipo: str
    descricao: str
    url: str = Field(default="")

class Activity(BaseModel):
    descricao: str
    tipo: str
    formato: str

class PlanStep(BaseModel):
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[ActivityResource]
    atividade: Activity
    progresso: int

class StudyPlan(BaseModel):
    plano_execucao: List[PlanStep]
    duracao_total: str
    progresso_total: int
    created_at: datetime
    horario_sugerido: Dict[str, str]

class PlanningState(TypedDict):
    messages: Sequence[BaseMessage]
    study_plan: dict
    user_profile: dict
    next_step: str | None
    review_feedback: str | None
    id_sessao: str
    update_status: bool | None
    scheduled_time: Dict[str, str] | None

class SessionPlanWorkflow:
    def __init__(self, db_tool: DatabaseUpdateTool):
        self.db_tool = db_tool
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
        Objetivo Geral da Disciplina: {objetivo_geral}

        Estilo de Aprendizagem do Aluno:
        {learning_style}

        Informações de Horário:
        Data do Encontro: {encounter_date}
        Horário do Encontro: {encounter_start} até {encounter_end}
        Preferência de Horário do Aluno: {study_preference}

        Crie um plano de estudo detalhado que:
        1. Seja adaptado ao estilo de aprendizagem específico do aluno
        2. Divida o conteúdo em etapas claras e gerenciáveis
        3. Inclua atividades e recursos apropriados ao estilo do aluno
        4. Foque em ensinar o tópico de forma eficaz e envolvente
        5. Seja específico para o tema da programação, incluindo exemplos práticos e exercícios de código
        6. SEMPRE FOQUE EM EXPLICAR O ASSUNTO ANTES DE FAZER QUALQUER ATIVIDADE
        7. IMPORTANTE: Agende a sessão de estudo para o mesmo dia do encontro ou para um dia anterior ao encontro
        8. Na descrição, explique detalhadamente as tecnologias, por exemplo. 
        9. Utilize o objetivo geral da disciplina para saber o que é a disciplina, mas foque no tema da sessão.

        Regras ESTRITAS para Agendamento:
        1. SEMPRE use a data do encontro ({encounter_date}) como referência
        2. A sessão DEVE ocorrer no mesmo dia do encontro, NUNCA depois
        3. Se agendar para o mesmo dia, a sessão DEVE terminar pelo menos 1 hora antes do encontro ({encounter_start})
        5. Respeite a preferência de horário do aluno ({study_preference}) quando possível
        6. Garanta pelo menos 1 hora de intervalo entre o fim da sessão e o início do encontro
        7. A duração da sessão deve ser compatível com o plano de estudo

        O plano deve seguir EXATAMENTE esta estrutura JSON:

            "plano_execucao": [
                    "titulo": "Título da seção"
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

                    "progresso": 0
            ],
            "duracao_total": "60 minutos",
            "progresso_total": 0,
            "horario_sugerido":
                "data": "YYYY-MM-DD",
                "inicio": "HH:MM",
                "fim": "HH:MM",
                "justificativa": "Explique a escolha do horário em relação à data do encontro"

        Adapte as atividades e recursos ao estilo de aprendizagem:
        - Para aluno Visual: Priorize recursos visuais como videos ou busca no banco de dados vetorial de imagens
        - Para aluno Verbal: Foque em explicações escritas e discussões
        - Para aluno Ativo: Inclua exercícios práticos e experimentação
        - Para aluno Reflexivo: Adicione momentos de análise e reflexão
        - Para aluno Sequencial: Organize em passos pequenos e conectados
        - Para aluno Global: Forneça visão geral e conexões com outros temas
        - Para links use links reais

        NOTA: LIMITE o TITULO e DESCRICAO a 100 caracteres.

        Retorne APENAS o JSON do plano, sem explicações adicionais."""

        prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
        model = ChatOpenAI(model="gpt-4o")

        def generate_plan(state: PlanningState) -> PlanningState:
            #print("[DEBUG]: Generating plan with state", state)

            topic = state["messages"][0].content if state["messages"] else "Tema não especificado"
            learning_style = state["user_profile"].get("EstiloAprendizagem", {})

            # Extrair informações de horário de forma mais estruturada
            encounter_info = state["user_profile"].get("horarios", {}).get("encontro", {})
            encounter_date = encounter_info.get("data", "Data não especificada")
            encounter_start = encounter_info.get("inicio", "Horário não especificado")
            encounter_end = encounter_info.get("fim", "Horário não especificado")
            study_preference = state["user_profile"].get("horarios", {}).get("preferencia")
            objetivo_geral = state["objetivo_geral"]
            #print("[DEBUG]: Learning style", learning_style)
            #print("[DEBUG]: Topic", topic)
            #print("[DEBUG]: Encounter date", encounter_date)
            #print("[DEBUG]: Study preference", study_preference)

            response = model.invoke(prompt.format(
                learning_style=learning_style,
                topic=topic,
                encounter_date=encounter_date,
                encounter_start=encounter_start,
                encounter_end=encounter_end,
                study_preference=study_preference,
                objetivo_geral=objetivo_geral,
            ))

            #print("[DEBUG]: Model response", response)

            cleaned_content = response.content.strip("```json\n").strip("```").strip()

            try:
                plan_dict = json.loads(cleaned_content)
                print("[DEBUG]: Parsed plan", plan_dict)
                new_state = state.copy()
                new_state["study_plan"] = plan_dict
                new_state["scheduled_time"] = plan_dict.get("horario_sugerido")
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

        Analise criticamente o seguinte plano:
        {plan}

        Considerando:
        - Estilo de aprendizagem do aluno: {learning_style}
        - Horário do encontro: {encounter_time}
        - Preferência de horário: {study_preference}
        - Objetivo geral da disciplina: {objetivo_geral}

        Verifique:
        1. Se os títulos são claros e bem estruturados
        2. Se as descrições são informativas e adequadas ao conteudo
        3. Se as atividades estão alinhadas com o estilo de aprendizagem
        4. Se a distribuição do tempo é apropriada
        5. Se os recursos são relevantes e bem escolhidos
        6. Se contém instruções de ensino claras e detalhadas
        7. Se os exemplos e exercícios de programação são apropriados
        8. Se o horário sugerido é adequado e respeita as regras de agendamento
        9. Se o plano está profundo tecnicamente

        O feedback deve seguir EXATAMENTE esta estrutura JSON:
            "status": "approved" ou "needs_revision",
            "feedback": "Seu feedback detalhado aqui",
            "suggestions": [
                "Sugestão 1",
                "Sugestão 2"
            ],
            "horario_validado": 
                "adequado": true/false,
                "comentario": "Comentário sobre o horário sugerido"

        Retorne APENAS o JSON, sem explicações adicionais.

        NOTA: Use o objetivo geral da disciplina para saber o que é a disciplina, mas foque no tema da sessão.
        """

        prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT)
        model = ChatOpenAI(model="gpt-4o", temperature=0.3)

        def review_plan(state: PlanningState) -> PlanningState:
            print("[DEBUG]: Reviewing plan", state["study_plan"])

            encounter_time = state["user_profile"].get("horarios", {}).get("encontro", {})
            study_preference = state["user_profile"].get("horarios", {}).get("preferencia")
            objetivo_geral = state["objetivo_geral"]
            response = model.invoke(prompt.format(
                plan=state["study_plan"],
                learning_style=state["user_profile"].get("EstiloAprendizagem", {}),
                encounter_time=encounter_time,
                study_preference=study_preference,
                objetivo_geral=objetivo_geral
            ))

            cleaned_content = response.content.strip("```json\n").strip("```").strip()

            print("[DEBUG]: Review response", response)
            new_state = state.copy()

            try:
                feedback_dict = json.loads(cleaned_content)
                print("[DEBUG]: Parsed feedback", feedback_dict)
                new_state["review_feedback"] = feedback_dict

                # Atualizar horário agendado apenas se validado
                if feedback_dict.get("horario_validado", {}).get("adequado", False):
                    new_state["scheduled_time"] = state["study_plan"].get("horario_sugerido")
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

            plan_with_schedule = state["study_plan"].copy()
            if state.get("scheduled_time"):
                plan_with_schedule["horario_agendado"] = state["scheduled_time"]

            success = await self.db_tool.update_study_plan(
                state["id_sessao"],
                plan_with_schedule
            )

            print("[DEBUG]: Database update success", success)
            new_state["update_status"] = success
            return new_state

        return update_database

    async def create_session_plan(self, topic: str, student_profile: dict, id_sessao: str, objetivo_geral: list) -> dict:
        if not topic:
            return {
                "error": "Tópico não fornecido",
                "plan": None,
                "feedback": None,
                "update_status": False,
                "id_sessao": id_sessao,
                "scheduled_time": None
            }

        initial_state = PlanningState(
            messages=[HumanMessage(content=topic)],
            study_plan={},
            user_profile=student_profile,
            next_step=None,
            review_feedback=None,
            id_sessao=id_sessao,
            update_status=None,
            scheduled_time=None,
            objetivo_geral=objetivo_geral
        )

        try:
            result = await self.workflow.ainvoke(initial_state)
            print("[DEBUG]: Workflow result", result)

            final_plan = result["study_plan"]
            final_plan["created_at"] = datetime.now(timezone.utc).isoformat()

            return {
                "plan": final_plan,
                "feedback": result["review_feedback"],
                "update_status": result["update_status"],
                "id_sessao": id_sessao,
                "scheduled_time": result.get("scheduled_time"),
                "objetivo_geral": objetivo_geral
            }
        except Exception as e:
            print(f"[DEBUG]: Error creating session plan: {e}")
            return {
                "error": f"Erro na criação do plano: {str(e)}",
                "plan": None,
                "feedback": None,
                "update_status": False,
                "id_sessao": id_sessao,
                "scheduled_time": None,
                "objetivo_geral": objetivo_geral
            }