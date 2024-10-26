from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
from typing import Dict, Any
from api.controllers.auth import get_current_user
from pydantic import BaseModel
from api.controllers.plan_controller import PlanController
from api.endpoints.models import StudyPlan
from api.controllers.study_sessions_controller import StudySessionsController
from api.dispatchers.study_sessions_dispatcher import StudySessionsDispatcher
from database.sql_database_manager import DatabaseManager, session, metadata
from agent.plan_agent import SessionPlanWorkflow
from database.mongo_database_manager import MongoDatabaseManager
from agent.tools import DatabaseUpdateTool

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
        controller = PlanController()  # Instancia o Controller

        plan_data = plan.model_dump()
        plan_data["created_at"] = datetime.now(timezone.utc)

        result = await controller.create_study_plan(plan_data)  # Usa o Controller

        if not result:
            raise HTTPException(status_code=500, detail="Erro ao criar plano de estudos.")

        return {"message": "Plano de estudos criado com sucesso", "id": result}

    except Exception as e:
        print(f"Erro ao criar plano de estudos: {e}")
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

        return plan

    except Exception as e:
        print(f"Erro ao buscar plano de estudos: {e}")
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

        return {"message": "Plano de estudos atualizado com sucesso"}

    except Exception as e:
        print(f"Erro ao atualizar plano de estudos: {e}")
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
        controller = PlanController()  # Instancia o Controller
        success = await controller.delete_study_plan(id_sessao)  # Usa o Controller

        if not success:
            raise HTTPException(status_code=404, detail="Plano de estudos não encontrado.")

        return {"message": "Plano de estudos excluído com sucesso"}

    except Exception as e:
        print(f"Erro ao excluir plano de estudos: {e}")
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
        print("Endpoint get_sessions_without_plan")
        print(sessions)
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

@router_study_plan.post("/study_plan/auto")
async def create_automatic_study_plan(
    session_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """
    Creates an automatic study plan using SessionPlanWorkflow.
    """
    try:
        print("session_data", session_data)
        mongo_manager = MongoDatabaseManager()
        db_tool = DatabaseUpdateTool(mongo_manager)
        # Get student profile
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

        # Get existing session data from MongoDB
        session_id = session_data.get('tema', {}).get('session_id')
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="ID da sessão não fornecido"
            )

        # Fetch existing session from MongoDB
        existing_session = await mongo_manager.get_study_plan(session_id)
        if not existing_session:
            raise HTTPException(
                status_code=404,
                detail="Sessão de estudo não encontrada"
            )

        print("initializing workflow")
        workflow = SessionPlanWorkflow(db_tool)
        print("workflow initialized")

        # Pass the session description from existing data
        result = await workflow.create_session_plan(
            topic=existing_session.get('descricao', ''),
            student_profile=student_profile,
            id_sessao=session_id
        )

        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        return {
            "message": "Plano de estudos gerado com sucesso",
            "plano": result["plan"],
            "feedback": result["feedback"]
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao criar plano automático: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao criar plano automático: {str(e)}"
        )