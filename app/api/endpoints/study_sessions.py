from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.controllers.auth import get_current_user
from api.controllers.study_sessions_controller import StudySessionsController
from database.sql_database_manager import session

router_study_sessions = APIRouter()

@router_study_sessions.get("/study_sessions")
async def get_study_sessions(current_user: dict = Depends(get_current_user)):
    try:
        controller = StudySessionsController(session)
        study_sessions = controller.get_all_study_sessions(current_user['sub'])
        return {"study_sessions": study_sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.post("/study_sessions")
async def create_study_session(discipline_name: str, current_user: dict = Depends(get_current_user)):
    try:
        controller = StudySessionsController(session)
        new_session = controller.create_study_session(current_user['sub'], discipline_name)
        return {"new_session": new_session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.put("/study_sessions/{session_id}")
async def update_study_session(session_id: int, session_data: dict, current_user: dict = Depends(get_current_user)):
    try:
        controller = StudySessionsController(session)
        updated_session = controller.update_study_session(session_id, session_data)
        return {"updated_session": updated_session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_study_sessions.delete("/study_sessions/{session_id}")
async def delete_study_session(session_id: int, current_user: dict = Depends(get_current_user)):
    try:
        controller = StudySessionsController(session)
        controller.delete_study_session(session_id)
        return {"message": "Study session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
