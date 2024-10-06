from fastapi import APIRouter, HTTPException, Depends
from fastapi.logger import logger
from sqlalchemy.orm import Session
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.controllers.educator_controller import EducatorController
from api.dispatchers.educator_dispatcher import EducatorDispatcher
from api.endpoints.models import EducatorCreate, EducatorUpdate

router_educator = APIRouter()


@router_educator.get("/educators")
async def get_all_educators(current_user: dict = Depends(get_current_user)):
    logger.info(f"Fetching all educators for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = EducatorDispatcher(sql_database_manager)
        controller = EducatorController(dispatcher)

        # Obter todos os educadores
        educators = controller.get_all_educators(current_user['sub'])

        logger.info(f"Educators fetched successfully for user: {current_user['sub']}")
        return {"educators": educators}
    except Exception as e:
        logger.error(f"Error fetching educators for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_educator.post("/educators")
async def create_educator(
    educator: EducatorCreate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Creating new educator for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = EducatorDispatcher(sql_database_manager)
        controller = EducatorController(dispatcher)

        # Criar um novo educador
        controller.create_educator(educator.name, educator.instituicao, educator.especializacao_disciplina, current_user['sub'])

        logger.info(f"New educator created successfully for user: {current_user['sub']}")
        return {"message": "Educator created successfully"}
    except Exception as e:
        logger.error(f"Error creating educator for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_educator.put("/educators/{educator_id}")
async def update_educator(
    educator_id: int,
    educator_data: EducatorUpdate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Updating educator {educator_id} for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = EducatorDispatcher(sql_database_manager)
        controller = EducatorController(dispatcher)

        # Atualizar o educador
        controller.update_educator(educator_id, educator_data.name, educator_data.instituicao, educator_data.especializacao_disciplina, current_user['sub'])

        logger.info(f"Educator {educator_id} updated successfully for user: {current_user['sub']}")
        return {"message": "Educator updated successfully"}
    except Exception as e:
        logger.error(f"Error updating educator {educator_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_educator.delete("/educators/{educator_id}")
async def delete_educator(educator_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"Deleting educator {educator_id} for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = EducatorDispatcher(sql_database_manager)
        controller = EducatorController(dispatcher)

        # Deletar o educador
        controller.delete_educator(educator_id, current_user['sub'])

        logger.info(f"Educator {educator_id} deleted successfully for user: {current_user['sub']}")
        return {"message": "Educator deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting educator {educator_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
