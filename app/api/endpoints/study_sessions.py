# app/api/endpoints/study_sessions.py
from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends
from fastapi.logger import logger
from sqlalchemy.orm import Session
from api.controllers.study_sessions_controller import StudySessionsController
from api.dispatchers.study_sessions_dispatcher import StudySessionsDispatcher
from database.sql_database_manager import DatabaseManager, session, metadata
from pdfminer.high_level import extract_text
from api.controllers.auth import get_current_user
from chains.chain_setup import DisciplinChain
import pdfplumber
from utils import OPENAI_API_KEY


router_study_sessions = APIRouter()

@router_study_sessions.get("/study_sessions")
async def get_study_sessions(current_user: dict = Depends(get_current_user)):
    logger.info(f"Fetching study sessions for user: {current_user['sub']}")
    try:
        # Instanciar o dispatcher e controlador
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Chamar o controlador para buscar as sessões de estudo
        study_sessions = controller.get_all_study_sessions(current_user['sub'])
        logger.info(f"Study sessions fetched successfully for user: {current_user['sub']}")
        return {"study_sessions": study_sessions}
    except Exception as e:
        logger.error(f"Error fetching study sessions for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.post("/study_sessions")
async def create_study_session(discipline_name: str, current_user: dict = Depends(get_current_user)):
    logger.info(f"Creating new study session for user: {current_user['sub']} and discipline: {discipline_name}")
    try:
        # Instanciar o dispatcher e controlador
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Chamar o controlador para criar uma nova sessão de estudo
        new_session = controller.create_study_session(current_user['sub'], discipline_name)
        logger.info(f"New study session created successfully for user: {current_user['sub']}, discipline: {discipline_name}")
        return {"new_session": new_session}
    except Exception as e:
        logger.error(f"Error creating study session for user: {current_user['sub']}, discipline: {discipline_name} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.put("/study_sessions/{session_id}")
async def update_study_session(session_id: int, session_data: dict, current_user: dict = Depends(get_current_user)):
    logger.info(f"Updating study session {session_id} for user: {current_user['sub']}")
    try:
        # Instanciar o dispatcher e controlador
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Chamar o controlador para atualizar a sessão de estudo
        updated_session = controller.update_study_session(session_id, session_data)
        logger.info(f"Study session {session_id} updated successfully for user: {current_user['sub']}")
        return {"updated_session": updated_session}
    except Exception as e:
        logger.error(f"Error updating study session {session_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.delete("/study_sessions/{session_id}")
async def delete_study_session(session_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"Deleting study session {session_id} for user: {current_user['sub']}")
    try:
        # Instanciar o dispatcher e controlador
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Chamar o controlador para deletar a sessão de estudo
        controller.delete_study_session(session_id)
        logger.info(f"Study session {session_id} deleted successfully for user: {current_user['sub']}")
        return {"message": "Study session deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting study session {session_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.get("/study_sessions/discipline")
async def get_study_session_from_discipline(discipline_name: str, current_user: dict = Depends(get_current_user)):
    logger.info(f"Fetching study sessions for user: {current_user['sub']} and discipline: {discipline_name}")
    try:
        # Instanciar o dispatcher e controlador
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Chamar o controlador para buscar sessões de estudo pela disciplina
        study_sessions = controller.get_study_session_from_discipline(discipline_name, current_user['sub'])
        print(study_sessions)
        return {"study_sessions": study_sessions}
    except Exception as e:
        logger.error(f"Error fetching study sessions for discipline '{discipline_name}' for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
