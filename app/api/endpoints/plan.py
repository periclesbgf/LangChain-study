from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
from typing import Dict, Any
from api.controllers.auth import get_current_user
from pydantic import BaseModel
from api.controllers.plan_controller import PlanController  # Importa o Controller
from api.endpoints.models import StudyPlan

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
