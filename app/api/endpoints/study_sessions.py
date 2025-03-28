# app/api/endpoints/study_sessions.py
from typing import List
from api.controllers.calendar_controller import CalendarController
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
from api.controllers.discipline_controller import DisciplineController
from api.dispatchers.discipline_dispatcher import DisciplineDispatcher
from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends, Query
from fastapi.logger import logger
from sqlalchemy.orm import Session
from api.controllers.study_sessions_controller import StudySessionsController
from api.dispatchers.study_sessions_dispatcher import StudySessionsDispatcher
from database.sql_database_manager import DatabaseManager, session, metadata
from pdfminer.high_level import extract_text
from api.controllers.auth import get_current_user
from chains.chain_setup import DisciplinChain
import pdfplumber
from api.endpoints.models import StudySessionCreate, StudySession
from datetime import datetime
from utils import OPENAI_API_KEY


router_study_sessions = APIRouter()

@router_study_sessions.get("/study_sessions", response_model=List[StudySession])
async def get_study_sessions(current_user: dict = Depends(get_current_user)):
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        study_sessions = controller.get_all_study_sessions(current_user['sub'])

        # Converter tuplas para objetos Pydantic
        study_sessions_data = [
            StudySession(
                id=session[0],
                course_id=session[1],
                user_id=session[2],
                title=session[3],
                start_time=session[4],
                end_time=session[5],
                status=session[6],
                notes=session[7],
                resources=session[8],
                period=session[9],
            )
            for session in study_sessions
        ]

        logger.info(f"[STUDY_SESSION_LIST] Usuário {current_user['sub']} acessou lista de sessões de estudo")

        return study_sessions_data
    except Exception as e:
        logger.error(f"Error fetching study sessions for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.post("/study_sessions")
async def create_study_session(
    study_session_model: StudySessionCreate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Instanciar os controladores e dispatchers
        sql_database_manager = DatabaseManager(session, metadata)
        discipline_dispatcher = DisciplineDispatcher(sql_database_manager)
        discipline_controller = DisciplineController(discipline_dispatcher)

        # Obter a disciplina pelo ID
        discipline = discipline_controller.get_discipline_by_id(
            study_session_model.discipline_id, current_user['sub']
        )

        # Instanciar o controlador de sessões de estudo
        study_sessions_dispatcher = StudySessionsDispatcher(sql_database_manager)
        study_sessions_controller = StudySessionsController(study_sessions_dispatcher)

        # Criar a sessão de estudo e obter o session_id
        session_result = await study_sessions_controller.create_study_session(
            user_email=current_user['sub'],
            discipline_id=study_session_model.discipline_id,
            subject=study_session_model.subject,
            start_time=study_session_model.start_time,
            end_time=study_session_model.end_time,
        )
        session_id = session_result['session_id']

        logger.info(f"[STUDY_SESSION_CREATE] Usuário {current_user['sub']} criou uma nova sessão de estudo com ID {session_id}")
        return {"new_session": session_id}
    except Exception as e:
        logger.error(f"[STUDY_SESSION_CREATE] Erro ao criar sessão de estudo para o usuário: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.put("/study_sessions/{session_id}")
async def update_study_session(session_id: int, session_data: dict, current_user: dict = Depends(get_current_user)):
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        updated_session = controller.update_study_session(session_id, session_data)
        logger.info(f"[STUDY_SESSION_UPDATE] Usuário {current_user['sub']} atualizou a sessão de estudo com ID {session_id}")
        return {"updated_session": updated_session}
    except Exception as e:
        logger.error(f"[STUDY_SESSION_UPDATE] Error updating study session {session_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.delete("/study_sessions/{session_id}")
async def delete_study_session(session_id: int, current_user: dict = Depends(get_current_user)):
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        controller.delete_study_session(session_id)
        logger.info(f"[STUDY_SESSION_DELETE] Usuário {current_user['sub']} deletou a sessão de estudo com ID {session_id}")
        return {"message": "Study session deleted successfully"}
    except Exception as e:
        logger.error(f"[STUDY_SESSION_DELETE] Error deleting study session {session_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.get("/study_sessions/discipline/{discipline_id}")
async def get_study_session_from_discipline(
    discipline_id: int,
    current_user: dict = Depends(get_current_user)
):
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        discipline_dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher, discipline_dispatcher=discipline_dispatcher)

        discipline_name = controller.get_discipline_name_from_id(discipline_id, current_user['sub'])

        study_sessions = controller.get_study_session_from_discipline(discipline_id, current_user['sub'])
        response = {
            "discipline_name": discipline_name,
            "study_sessions": study_sessions
        }
        logger.info(f"[STUDY_SESSION_DISCIPLINE] Usuário {current_user['sub']} acessou sessões de estudo da disciplina {discipline_name}")
        return response
    except Exception as e:
        logger.error(f"Error fetching study sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar sessões de estudo: {str(e)}")

@router_study_sessions.get("/study_sessions/session/{session_id}")
async def get_study_session_by_id(
    session_id: int,
    current_user: dict = Depends(get_current_user)
):
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Chamar o controlador para buscar a sessão de estudo pelo ID
        study_session = controller.get_study_session_by_id(session_id, current_user['sub'])
        if not study_session:
            raise HTTPException(status_code=404, detail="Sessão de estudo não encontrada.")
        logger.info(f"[STUDY_SESSION_FETCH] Usuário {current_user['sub']} acessou a sessão de estudo com ID {session_id}")
        return {"study_session": study_session}
    except Exception as e:
        logger.error(f"Error fetching study session ID '{session_id}' for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar sessão de estudo: {str(e)}")
