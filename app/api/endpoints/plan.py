from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
from typing import Dict, Any
from api.controllers.auth import get_current_user
from pydantic import BaseModel, conint
from api.controllers.plan_controller import PlanController
from api.endpoints.models import StudyPlan, AutomaticStudyPlanRequest, AutomaticStudyPlanResponse
from api.controllers.study_sessions_controller import StudySessionsController
from api.dispatchers.study_sessions_dispatcher import StudySessionsDispatcher
from api.controllers.discipline_controller import DisciplineController
from api.dispatchers.discipline_dispatcher import DisciplineDispatcher
from database.sql_database_manager import DatabaseManager, session, metadata
from agent.plan_agent import SessionPlanWorkflow
from database.mongo_database_manager import MongoDatabaseManager
from agent.tools import DatabaseUpdateTool
from api.controllers.calendar_controller import CalendarController
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
from logg import logger

router_study_plan = APIRouter()


@router_study_plan.post("/study_plan")
async def create_study_plan(
    plan: StudyPlan,
    current_user: dict = Depends(get_current_user)
):
    """
    Cria um novo plano de estudos para o estudante autenticado.
    """
    try:
        controller = PlanController()

        plan_data = plan.model_dump()
        plan_data["created_at"] = datetime.now(timezone.utc)

        result = await controller.create_study_plan(plan_data)

        if not result:
            raise HTTPException(status_code=500, detail="Erro ao criar plano de estudos.")

        logger.info(f"[STUDY_PLAN_CREATE] Usuário {current_user['sub']} criou um novo plano de estudos com ID {result}")

        return {"message": "Plano de estudos criado com sucesso", "id": result}

    except Exception as e:
        logger.error(f"Erro ao criar plano de estudos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar plano de estudos: {str(e)}")

@router_study_plan.get("/study_plan/{id_sessao}")
async def get_study_plan(
    id_sessao: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Recupera um plano de estudos específico pelo id_sessao.
    """
    try:
        controller = PlanController()  # Instancia o Controller
        plan = await controller.get_study_plan(id_sessao)  # Usa o Controller

        if not plan:
            raise HTTPException(status_code=404, detail="Plano de estudos não encontrado.")
        logger.info(f"[STUDY_PLAN] Usuário {current_user['sub']} acessou o plano de estudos com ID {id_sessao}")

        return plan

    except Exception as e:
        logger.error(f"Erro ao buscar plano de estudos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar plano de estudos: {str(e)}")

@router_study_plan.put("/study_plan/{id_sessao}")
async def update_study_plan(
    id_sessao: str,
    updated_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Atualiza um plano de estudos existente.
    """
    try:
        controller = PlanController()  # Instancia o Controller
        updated_data["updated_at"] = datetime.now(timezone.utc)
        success = await controller.update_study_plan(id_sessao, updated_data)  # Usa o Controller

        if not success:
            raise HTTPException(status_code=404, detail="Plano de estudos não encontrado.")

        logger.info(f"[STUDY_PLAN_UPDATE] Usuário {current_user['sub']} atualizou o plano de estudos com ID {id_sessao}")

        return {"message": "Plano de estudos atualizado com sucesso"}

    except Exception as e:
        logger.error(f"Erro ao atualizar plano de estudos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar plano de estudos: {str(e)}")

@router_study_plan.delete("/study_plan/{id_sessao}")
async def delete_study_plan(
    id_sessao: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Exclui um plano de estudos pelo id_sessao.
    """
    try:
        controller = PlanController()
        success = await controller.delete_study_plan(id_sessao) # possivel erro de segurança aqui

        if not success:
            raise HTTPException(status_code=404, detail="Plano de estudos não encontrado.")
        logger.info(f"[STUDY_PLAN_DELETE] Usuário {current_user['sub']} excluiu o plano de estudos com ID {id_sessao}")
        return {"message": "Plano de estudos excluído com sucesso"}

    except Exception as e:
        logger.error(f"Erro ao excluir plano de estudos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao excluir plano de estudos: {str(e)}")

@router_study_plan.get("/study_plan/sessions/without_plan")
async def get_sessions_without_plan(
    current_user: dict = Depends(get_current_user)
):
    """
    Recupera todas as sessões de estudo do usuário que não possuem plano de execução.
    """
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        study_sessions_dispatcher = StudySessionsDispatcher(sql_database_manager)
        study_sessions_controller = StudySessionsController(study_sessions_dispatcher)

        # Chamar o controlador para buscar as sessões de estudo
        study_sessions = study_sessions_controller.get_all_study_sessions(current_user['sub'])

        controller = PlanController()
        sessions = await controller.get_sessions_without_plan(current_user["sub"], study_sessions)

        if not sessions:
            return {"sessions": []}
        logger.info(f"[STUDY_PLAN] Usuário {current_user['sub']} acessou sessões sem plano")
        # Retorna as sessões sem plano
        return {
            "sessions": sessions
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao buscar sessões sem plano: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao buscar sessões sem plano: {str(e)}"
        )

@router_study_plan.get("/study_plan/sessions/without_plan/{discipline_id}")
async def get_sessions_without_plan_by_discipline(
    discipline_id: int,
    current_user: dict = Depends(get_current_user)
):
    """
    Recupera todas as sessões de estudo de uma disciplina específica que não possuem plano de execução.
    """
    try:
        # Inicializa os gerenciadores necessários
        sql_database_manager = DatabaseManager(session, metadata)
        study_sessions_dispatcher = StudySessionsDispatcher(sql_database_manager)
        study_sessions_controller = StudySessionsController(study_sessions_dispatcher)

        # Recupera sessões de estudo pela disciplina
        study_sessions = study_sessions_controller.get_study_session_from_discipline(
            discipline_id, current_user["sub"]
        )

        # Inicializa o controlador de planos
        controller = PlanController()

        # Usa o novo método para filtrar as sessões
        sessions = await controller.get_sessions_without_plan_by_discipline(
            current_user["sub"], discipline_id, study_sessions
        )

        if not sessions:
            return {"sessions": []}

        return {
            "sessions": sessions
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao buscar sessões sem plano por disciplina: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar sessões sem plano por disciplina: {str(e)}"
        )


@router_study_plan.get("/study_plan/verify/{id_sessao}")
async def verify_session_has_plan(
    id_sessao: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Verifica se uma sessão específica já possui um plano de estudo.
    """
    try:
        controller = PlanController()
        has_plan = await controller.verify_session_has_plan(id_sessao)
        return {"has_plan": has_plan}

    except Exception as e:
        print(f"Erro ao verificar plano da sessão: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao verificar plano da sessão: {str(e)}"
        )

@router_study_plan.post("/study_plan/auto", response_model=AutomaticStudyPlanResponse)
async def create_automatic_study_plan(
    planData: AutomaticStudyPlanRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"[PLAN_AUTO_GEN] Usuário {current_user['sub']} solicitou geração de plano automático para sessão {planData.session_id}")
        mongo_manager = MongoDatabaseManager()
        db_tool = DatabaseUpdateTool(mongo_manager)

        # Obter o perfil do estudante
        student_profile = await mongo_manager.get_student_profile(
            current_user["sub"], 
            'student_learn_preference'
        )
        print("Student profile: ", student_profile)

        if not student_profile:
            raise HTTPException(
                status_code=404,
                detail="Perfil do estudante não encontrado"
            )

        session_id = planData.session_id
        topic = planData.tema

        if not topic:
            raise HTTPException(
                status_code=400,
                detail="Tema da sessão não fornecido"
            )

        # Instanciar o DatabaseManager e buscar horários
        sql_manager = DatabaseManager(session, metadata)
        encontro_info = sql_manager.get_encontro_horarios(session_id)
        discipline_dispatcher = DisciplineDispatcher(sql_manager)
        discipline_controller = DisciplineController(discipline_dispatcher)

        discipline_data = discipline_controller.get_discipline_by_session_id(session_id)
        print("discipline_data:")
        print(discipline_data)

        objetivo_geral = discipline_data.get('Objetivos', [])

        print("objetivo_geral:")
        print(objetivo_geral)
        if not encontro_info:
            raise HTTPException(
                status_code=404,
                detail="Informações do encontro não encontradas"
            )

        # Buscar a sessão existente do MongoDB
        existing_session = await mongo_manager.get_study_plan(session_id)
        if not existing_session:
            raise HTTPException(
                status_code=404,
                detail="Sessão de estudo não encontrada"
            )

        # Adicionar informações de horário e período ao perfil do estudante
        student_profile["horarios"] = {
            "encontro": {
                "inicio": encontro_info["horario_inicio"].strftime("%H:%M"),
                "fim": encontro_info["horario_fim"].strftime("%H:%M"),
                "data": encontro_info["data_encontro"].strftime("%Y-%m-%d")
            },
            "preferencia": planData.periodo or encontro_info["preferencia_horario"]
        }

        print("student_profile", student_profile)

        # Criar o plano de sessão
        workflow = SessionPlanWorkflow(db_tool)
        result = await workflow.create_session_plan(
            topic=topic,
            student_profile=student_profile,
            id_sessao=session_id,
            objetivo_geral=objetivo_geral
        )

        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        # Atualizar os horários da sessão de estudo, se aplicável
        if result.get("scheduled_time"):
            data = result["scheduled_time"]["data"]
            inicio = result["scheduled_time"]["inicio"]
            fim = result["scheduled_time"]["fim"]
            
            start_datetime = datetime.strptime(f"{data} {inicio}", "%Y-%m-%d %H:%M")
            end_datetime = datetime.strptime(f"{data} {fim}", "%Y-%m-%d %H:%M")
            
            sql_manager.update_session_times(
                session_id=session_id,
                start_time=start_datetime,
                end_time=end_datetime
            )

            # Criar evento no calendário
            event_title = f"Sessão de Estudo - {topic}"
            dispatcher = CalendarDispatcher(sql_manager)
            controller = CalendarController(dispatcher)

            controller.create_event(
                title=event_title,
                description=result["plan"].get("descricao", ""),
                start_time=start_datetime,
                end_time=end_datetime,
                location="Online",
                current_user=current_user['sub'],
                categoria="estudo_individual",
                importancia="alta",
                #course_id=planData.disciplina_id
            )

        response = AutomaticStudyPlanResponse(
            message="Plano de estudos gerado com sucesso",
            plano=result["plan"]
        )
        logger.info(f"[PLAN_AUTO_GEN_SUCCESS] Plano gerado para sessão {planData.session_id} do usuário {current_user['sub']}")
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao criar plano automático: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao criar plano automático: {str(e)}"
        )

class StepProgressUpdate(BaseModel):
    """Schema para atualização do progresso de uma etapa"""
    step_index: conint(ge=0)
    progress: conint(ge=0, le=100)

@router_study_plan.put("/study_plan/{id_sessao}/progress", response_model=Dict[str, Any])
async def update_step_progress(
    id_sessao: str,
    progress_data: StepProgressUpdate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"[PLAN_PROGRESS] Usuário {current_user['sub']} atualizou progresso da etapa {progress_data.step_index} para {progress_data.progress}% na sessão {id_sessao}")
    """
    Atualiza o progresso de uma etapa específica do plano de estudos.

    Args:
        id_sessao: ID da sessão de estudo
        progress_data: Dados de progresso contendo step_index e progress
        current_user: Usuário autenticado

    Returns:
        Dict contendo mensagem de sucesso e dados atualizados do progresso
    """
    try:
        controller = PlanController()

        # Verifica se o plano existe
        existing_plan = await controller.get_study_plan(id_sessao)
        if not existing_plan:
            raise HTTPException(
                status_code=404,
                detail="Plano de estudos não encontrado"
            )

        # Verifica se o índice da etapa é válido
        plano_execucao = existing_plan.get("plano_execucao", [])
        if progress_data.step_index >= len(plano_execucao):
            raise HTTPException(
                status_code=400,
                detail=f"Índice de etapa inválido: {progress_data.step_index}"
            )

        # Atualiza o progresso
        success = await controller.update_step_progress(
            id_sessao,
            progress_data.step_index,
            progress_data.progress
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Erro ao atualizar progresso da etapa"
            )

        # Recupera os dados atualizados do progresso
        updated_progress = await controller.get_plan_progress(id_sessao)
        if not updated_progress:
            raise HTTPException(
                status_code=500,
                detail="Erro ao recuperar progresso atualizado"
            )

        return {
            "message": "Progresso atualizado com sucesso",
            "data": {
                "session_id": id_sessao,
                "step_index": progress_data.step_index,
                "new_progress": progress_data.progress,
                "total_progress": updated_progress["progresso_total"],
                "plano_execucao": updated_progress["plano_execucao"]
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao atualizar progresso: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao atualizar progresso: {str(e)}"
        )
@router_study_plan.get("/study_plan/{id_sessao}/progress")
async def get_plan_progress(
    id_sessao: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Recupera o progresso atual do plano de estudos.

    Args:
        id_sessao: ID da sessão de estudo
        current_user: Usuário autenticado

    Returns:
        Dict contendo os dados de progresso do plano
    """
    try:
        controller = PlanController()
        progress = await controller.get_plan_progress(id_sessao)

        if not progress:
            raise HTTPException(
                status_code=404,
                detail="Plano de estudos não encontrado"
            )

        return {
            "session_id": id_sessao,
            "total_progress": progress["progresso_total"],
            "plano_execucao": progress["plano_execucao"]
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao recuperar progresso: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao recuperar progresso: {str(e)}"
        )