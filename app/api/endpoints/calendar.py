# app/api/endpoints/calendar.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.logger import logger
from sqlalchemy.orm import Session
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.controllers.calendar_controller import CalendarController
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
from datetime import datetime
from api.endpoints.models import CalendarEvent, CalendarEventUpdate


router_calendar = APIRouter()


@router_calendar.get("/calendar/events")
async def get_calendar_events(current_user: dict = Depends(get_current_user)):
    logger.info(f"[CALENDAR_LIST] Usuário {current_user['sub']} acessou lista de eventos do calendário")
    try:
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = CalendarDispatcher(sql_database_manager)
        controller = CalendarController(dispatcher)

        events = controller.get_all_events(current_user['sub'])

        logger.info(f"Calendar events fetched successfully for user: {current_user['sub']}")
        return {"events": events}
    except Exception as e:
        logger.error(f"Error fetching calendar events for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_calendar.post("/calendar/events")
async def create_calendar_event(
    event: CalendarEvent,
    course_id: int = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = CalendarDispatcher(sql_database_manager)
        controller = CalendarController(dispatcher)

        controller.create_event(
            event.title,
            event.description,
            event.start_time,
            event.end_time,
            event.location,
            current_user['sub'],
            course_id,
            event.categoria,
            event.importancia,
            event.material,
        )

        logger.info(f"[CALENDAR_CREATE] Usuário {current_user['sub']} criou um novo evento no calendário")
        return {"message": "Event created successfully"}
    except Exception as e:
        logger.error(f"Error creating calendar event for user: {current_user['sub']} - {str(e)}")
        if "End time must be after start time" in str(e):
            raise HTTPException(status_code=400, detail="End time must be after start time")
        if "duplicado" in str(e):
            raise HTTPException(status_code=400, detail="Event already exists")
        raise HTTPException(status_code=500, detail=str(e))



@router_calendar.put("/calendar/events/{event_id}")
async def update_calendar_event(
    event_id: int,
    event_data: CalendarEventUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        # Instanciar o Dispatcher e Controller
        dispatcher = CalendarDispatcher(sql_database_manager)
        controller = CalendarController(dispatcher)

        # Atualizar evento via controller, passando o e-mail do usuário (sub)
        controller.update_event(
            event_id,
            title=event_data.title,
            description=event_data.description,
            start_time=event_data.start_time,
            end_time=event_data.end_time,
            location=event_data.location,
            current_user=current_user['sub'],
            categoria=event_data.categoria,
            importancia=event_data.importancia,
            material=event_data.material,
        )

        logger.info(f"[CALENDAR_UPDATE] Usuário {current_user['sub']} atualizou o evento {event_id} no calendário")
        return {"message": "Event updated successfully"}
    except Exception as e:
        logger.error(f"Error updating calendar event {event_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_calendar.delete("/calendar/events/{event_id}")
async def delete_calendar_event(
    event_id: int,
    current_user: dict = Depends(get_current_user)
):
    try:
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = CalendarDispatcher(sql_database_manager)
        controller = CalendarController(dispatcher)

        controller.delete_event(event_id, current_user['sub'])

        logger.info(f"[CALENDAR_DELETE] Usuário {current_user['sub']} deletou o evento {event_id} do calendário")
        return {"message": "Event deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting calendar event {event_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

