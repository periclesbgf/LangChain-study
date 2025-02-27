# app/api/endpoints/profiles.py

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict
from database.sql_database_manager import DatabaseManager, session
from api.controllers.auth import get_current_user
from sql_interface.sql_tables import tabela_perfil_aprendizado_aluno
from api.endpoints.models import EstiloAprendizagem, Feedback, PreferenciasAprendizado
from database.mongo_database_manager import MongoDatabaseManager
from datetime import datetime, timezone
from database.sql_database_manager import DatabaseManager, session, metadata

router_profiles = APIRouter()


@router_profiles.post("/profiles")
async def create_profile(
    estilo_aprendizagem: EstiloAprendizagem,
    current_user: dict = Depends(get_current_user)
):
    """
    Cria um novo perfil de aprendizado para o estudante autenticado.
    """
    try:
        mongo_manager = MongoDatabaseManager()
        sql_manager = DatabaseManager(session, metadata)

        student_name = sql_manager.get_user_name_by_email(current_user["sub"])

        profile_data = {
            "Nome": student_name,
            "Email": current_user["sub"],
            "EstiloAprendizagem": estilo_aprendizagem.model_dump(),
            "Feedback": None,
            "PreferenciaAprendizado": None,
            "created_at": datetime.now(timezone.utc)
        }

        profile_id = await mongo_manager.create_student_profile(
            email=current_user["sub"],
            profile_data=profile_data
        )

        if not profile_id:
            raise HTTPException(status_code=500, detail="Erro ao criar perfil.")

        return {"message": "Perfil criado com sucesso", "IdPerfil": profile_id}

    except Exception as e:
        print(f"Erro ao criar perfil: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar perfil: {str(e)}")

@router_profiles.get("/profiles")
async def get_profile(
    current_user: dict = Depends(get_current_user)
):
    """
    Recupera o perfil do estudante com base no email.
    """
    try:
        mongo_manager = MongoDatabaseManager()

        email = current_user["sub"]
        print(f"Buscando perfil para o email: {email}")

        profile = await mongo_manager.get_student_profile(
            email=email,
            collection_name="student_learn_preference"
        )

        if not profile:
            sql_manager = DatabaseManager(session, metadata)

            student_name = sql_manager.get_user_name_by_email(current_user["sub"])

            profile_data = {
                "Nome": student_name,
                "Email": current_user["sub"],
                "EstiloAprendizagem": None,
                "Feedback": None,
                "PreferenciaAprendizado": None,
                "created_at": datetime.now(timezone.utc)
            }

            profile_id = await mongo_manager.create_student_profile(
                email=current_user["sub"],
                profile_data=profile_data
            )
            return profile_id
        profile["_id"] = str(profile["_id"])
        return profile

    except Exception as e:
        print(f"Erro ao buscar perfil: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar perfil: {str(e)}")

@router_profiles.put("/profiles")
async def update_profile(
    estilo_aprendizagem: EstiloAprendizagem,
    current_user: dict = Depends(get_current_user)
):
    """
    Atualiza o perfil de aprendizado do estudante autenticado.
    """
    try:
        mongo_manager = MongoDatabaseManager()

        update_result = await mongo_manager.db['student_learn_preference'].update_one(
            {"Email": current_user["sub"]},
            {
                "$set": {
                    "EstiloAprendizagem": estilo_aprendizagem.dict(),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )

        if update_result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Perfil não encontrado.")

        return {"message": "Perfil atualizado com sucesso"}

    except Exception as e:
        print(f"Erro ao atualizar perfil: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar perfil: {str(e)}")

@router_profiles.delete("/profiles")
async def delete_profile(
    current_user: dict = Depends(get_current_user)
):
    """
    Exclui o perfil de aprendizado do estudante autenticado.
    """
    try:
        mongo_manager = MongoDatabaseManager()

        delete_result = await mongo_manager.db['student_learn_preference'].delete_one(
            {"Email": current_user['sub']}
        )

        if delete_result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Perfil não encontrado.")

        return {"message": "Perfil excluído com sucesso"}

    except Exception as e:
        print(f"Erro ao excluir perfil: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao excluir perfil: {str(e)}")