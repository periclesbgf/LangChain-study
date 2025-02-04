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
    logger.info(f"Fetching study sessions for user: {current_user['sub']}")
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

        logger.info(f"Study sessions fetched successfully for user: {current_user['sub']}")
        print(study_sessions_data)
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
        print(f"Creating new study session for user: {current_user['sub']} and discipline: {discipline['NomeCurso']}")

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

        logger.info(f"New study session and empty plan created successfully for user: {current_user['sub']}, discipline: {discipline['NomeCurso']}")
        return {"new_session": session_id}
    except Exception as e:
        logger.error(f"Error creating study session: {str(e)}")
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

@router_study_sessions.get("/study_sessions/discipline/{discipline_id}")
async def get_study_session_from_discipline(
    discipline_id: int,
    current_user: dict = Depends(get_current_user)
):
    print(f"Fetching study sessions for user: {current_user['sub']} and discipline: {discipline_id}")
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Buscar sessões de estudo pela disciplina
        study_sessions = controller.get_study_session_from_discipline(discipline_id, current_user['sub'])
        response = {"study_sessions": study_sessions}
        print(response)
        return response
    except Exception as e:
        logger.error(f"Error fetching study sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar sessões de estudo: {str(e)}")

@router_study_sessions.get("/study_sessions/session/{session_id}")
async def get_study_session_by_id(
    session_id: int,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Fetching study session with ID {session_id} for user: {current_user['sub']}")
    try:
        # Instanciar o dispatcher e controlador
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = StudySessionsDispatcher(sql_database_manager)
        controller = StudySessionsController(dispatcher)

        # Chamar o controlador para buscar a sessão de estudo pelo ID
        study_session = controller.get_study_session_by_id(session_id, current_user['sub'])
        if not study_session:
            raise HTTPException(status_code=404, detail="Sessão de estudo não encontrada.")
        # Retornar o resultado no formato correto
        print(study_session)
        return {"study_session": study_session}
    except Exception as e:
        logger.error(f"Error fetching study session ID '{session_id}' for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar sessão de estudo: {str(e)}")
