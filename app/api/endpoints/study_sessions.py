# app/api/endpoints/study_sessions.py
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
from api.endpoints.models import StudySessionCreate
from datetime import datetime
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
async def create_study_session(
    study_sessio_model: StudySessionCreate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Instanciando classes do controlador
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Obtendo disciplina pelo ID
        discipline = controller.get_discipline_by_id(study_sessio_model.discipline_id, current_user['sub'])
        print(f"Creating new study session for user: {current_user['sub']} and discipline: {discipline["NomeCurso"]}")

        # Controladores de calendário e sessões de estudo
        calendar_dispatcher = CalendarDispatcher(sql_database_manager)
        calendar_controller = CalendarController(calendar_dispatcher)

        study_sessions_dispatcher = StudySessionsDispatcher(sql_database_manager)
        study_sessions_controller = StudySessionsController(study_sessions_dispatcher)
        print("controllers and dispatchers created")
        # Criando a sessão de estudo
        new_session = study_sessions_controller.create_study_session(
            user_email=current_user['sub'],
            discipline_id=study_sessio_model.discipline_id,
            subject=study_sessio_model.subject,
        )

        # Criando um evento no calendário
        calendar_controller.create_event(
            title=f"{study_sessio_model.subject}",
            description=study_sessio_model.subject,
            start_time=study_sessio_model.start_time,
            end_time=study_sessio_model.end_time,
            location="Online",
            current_user=current_user['sub'],
            course_id=study_sessio_model.discipline_id
        )

        logger.info(f"New study session created successfully for user: {current_user['sub']}, discipline: {discipline["NomeCurso"]}")
        return {"new_session": new_session}
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
        print(f"Study sessions: {study_sessions}")
        return {"study_sessions": study_sessions}
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
