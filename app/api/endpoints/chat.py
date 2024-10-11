# app/api/endpoints/chat.py

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime
from api.controllers.chat_controller import ChatController
#from api.dispatchers.chat_dispatcher import ChatDispatcher
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.endpoints.models import MessageRequest

router_chat = APIRouter()

# @router_chat.post("/chat/sessions/{session_id}/messages")
# async def send_message(
#     session_id: int,
#     message: dict,
#     current_user: dict = Depends(get_current_user)
# ):
#     try:
#         db_manager = DatabaseManager(session, metadata)
#         dispatcher = ChatDispatcher(db_manager)
#         controller = ChatController(dispatcher)

#         response = controller.handle_user_message(
#             session_id,
#             current_user['sub'],
#             message
#         )

#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router_chat.get("/chat/sessions/{session_id}/messages")
# async def get_chat_history(
#     session_id: int,
#     page: int = Query(1, ge=1),
#     page_size: int = Query(20, ge=1),
#     current_user: dict = Depends(get_current_user)
# ):
#     try:
#         db_manager = DatabaseManager(session, metadata)
#         dispatcher = ChatDispatcher(db_manager)
#         controller = ChatController(dispatcher)

#         messages = controller.get_chat_history(
#             session_id,
#             current_user['sub'],
#             page,
#             page_size
#         )

#         return {"messages": messages}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router_chat.post("/chat")
async def chat_endpoint(request: MessageRequest, current_user=Depends(get_current_user)):
    try:
        print("Handling user message")
        print(f"session_id={request.session_id} message='{request.message}'")
        controller = ChatController(session_id=str(request.session_id), student_email=current_user["sub"])
        response = await controller.handle_user_message(request.message)
        print(f"response='{response}'")
        return {"response": response}
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))