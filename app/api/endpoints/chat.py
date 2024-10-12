# app/api/endpoints/chat.py

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime
from api.controllers.chat_controller import ChatController
#from api.dispatchers.chat_dispatcher import ChatDispatcher
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.endpoints.models import MessageRequest
from fastapi import APIRouter, Depends, HTTPException
from pymongo import MongoClient
from bson.objectid import ObjectId
from utils import MONGO_URI, MONGO_DB_NAME
from typing import Optional
import json
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
        print(f"session_id={request.session_id}, discipline_id={request.discipline_id}, message='{request.message}'")
        controller = ChatController(
            session_id=str(request.session_id),
            student_email=current_user["sub"],
            disciplina=str(request.discipline_id)  # Passa discipline_id como string
        )
        response = await controller.handle_user_message(request.message)
        print(f"response='{response}'")
        return {"response": response}
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router_chat.get("/chat_history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 10,
    before: Optional[str] = None,
    current_user=Depends(get_current_user)
):
    try:
        # Conecte-se ao MongoDB
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db['chat_history']

        # Construir a consulta
        query = {
            'session_id': session_id,
            'user_email': current_user["sub"],
        }

        if before:
            # Converter a string before para datetime
            try:
                before_datetime = datetime.fromisoformat(before)
                query['timestamp'] = {'$lt': before_datetime}
            except ValueError:
                print(f"Invalid 'before' timestamp format: {before}")
                raise HTTPException(status_code=400, detail="Invalid 'before' timestamp format.")

        # Buscar mensagens ordenadas por timestamp descendente (mais recentes primeiro), limitado por 'limit'
        messages_cursor = collection.find(
            query
        ).sort("timestamp", -1).limit(limit)

        messages = []
        for message_doc in messages_cursor:
            history_data = json.loads(message_doc.get('history', '{}'))
            content = history_data.get('data', {}).get('content', '')
            role = history_data.get('type', 'unknown')  # 'human' ou 'ai'
            timestamp = message_doc.get('timestamp')
            messages.append({
                'role': 'user' if role == 'human' else 'assistant',
                'content': content,
                'timestamp': timestamp.isoformat()
            })

        # Como ordenamos descendente, invertemos para ter ordem ascendente
        messages.reverse()

        return {'messages': messages}
    except Exception as e:
        print(f"Error in get_chat_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))