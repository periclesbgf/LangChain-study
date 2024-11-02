# app/api/endpoints/chat.py

import traceback
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from api.controllers.chat_controller import ChatController
from api.controllers.auth import get_current_user
from api.endpoints.models import MessageRequest
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, Any
from datetime import datetime
import json
import os
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

from utils import MONGO_URI, MONGO_DB_NAME, QDRANT_URL, OPENAI_API_KEY
from database.vector_db import (
    QdrantHandler, 
    Embeddings, 
    TextSplitter, 
    Embeddings
)
from agent.image_handler import ImageHandler
from database.mongo_database_manager import MongoDatabaseManager
from agent.agent_test import TutorWorkflow

router_chat = APIRouter()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class ChatEndpointManager:
    def __init__(self):
        self.mongo_manager = None
        self.qdrant_handler = None
        self.image_handler = None
        self.embeddings = None
        self._cache = {}
        
    @asynccontextmanager
    async def get_mongo_manager(self):
        """Get or create MongoDB manager with connection pooling"""
        if not self.mongo_manager:
            self.mongo_manager = MongoDatabaseManager()
        try:
            yield self.mongo_manager
        except Exception as e:
            print(f"Error with MongoDB manager: {e}")
            raise
            
    @lru_cache(maxsize=100)
    def get_qdrant_handler(self) -> QdrantHandler:
        """Get or create QdrantHandler with caching"""
        if not self.qdrant_handler:
            if not self.image_handler:
                self.image_handler = ImageHandler(OPENAI_API_KEY)
            if not self.embeddings:
                # Inicializa os embeddings corretamente
                self.embeddings = Embeddings()
                
            self.qdrant_handler = QdrantHandler(
                url=QDRANT_URL,
                collection_name="student_documents",
                embeddings=self.embeddings.get_embeddings(),
                text_splitter=TextSplitter(),
                image_handler=self.image_handler,
                mongo_manager=self.mongo_manager
            )
        return self.qdrant_handler
        
    async def get_student_data(
        self, 
        user_email: str, 
        session_id: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Get student profile and study plan concurrently"""
        async with self.get_mongo_manager() as mongo_manager:
            # Execute requests concurrently
            profile_task = asyncio.create_task(
                mongo_manager.get_student_profile(
                    email=user_email,
                    collection_name="student_learn_preference"
                )
            )
            plan_task = asyncio.create_task(
                mongo_manager.get_study_plan(session_id)
            )
            
            # Wait for both tasks to complete
            student_profile, study_plan = await asyncio.gather(
                profile_task, 
                plan_task
            )
            print(f"Student profile: {student_profile}")
            print(f"Study plan: {study_plan}")
            
            if not student_profile:
                raise HTTPException(
                    status_code=404, 
                    detail="Perfil do estudante não encontrado."
                )
                
            if not study_plan:
                raise HTTPException(
                    status_code=404, 
                    detail="Plano de estudo não encontrado."
                )
                
            return student_profile, study_plan
            
    def get_cache_key(self, user_email: str, session_id: str) -> str:
        """Generate cache key for student data"""
        return f"{user_email}:{session_id}"
        
    async def get_cached_student_data(
        self, 
        user_email: str, 
        session_id: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Get student data with caching"""
        cache_key = self.get_cache_key(user_email, session_id)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        data = await self.get_student_data(user_email, session_id)
        self._cache[cache_key] = data
        return data

chat_manager = ChatEndpointManager()

@router_chat.post("/chat")
async def chat_endpoint(
    request: MessageRequest = Depends(MessageRequest.as_form),
    current_user=Depends(get_current_user),
    background_tasks: BackgroundTasks = None
):
    """Optimized chat endpoint with better error handling and performance"""
    try:
        # Get cached student data
        student_profile, study_plan_raw = await chat_manager.get_cached_student_data(
            current_user["sub"],
            str(request.session_id)
        )
        print(f"Student profile: {student_profile}")
        # Convert study plan to JSON
        study_plan = json.dumps(study_plan_raw, cls=DateTimeEncoder)
        
        # Get or create handlers
        qdrant_handler = chat_manager.get_qdrant_handler()
        
        # Initialize workflow first
        workflow = await initialize_workflow(
            qdrant_handler,
            current_user["sub"],
            request.discipline_id,
            str(request.session_id),
            chat_manager.mongo_manager
        )
        
        # Then initialize controller with the workflow
        controller = await initialize_controller(
            request,
            current_user["sub"],
            qdrant_handler,
            student_profile,
            study_plan,
            workflow  # Pass the workflow here
        )
        
        # Process message
        files = [request.file] if request.file else []
        response = await controller.handle_user_message(request.message, files)
        
        # Add background task for cleanup if needed
        if background_tasks:
            background_tasks.add_task(cleanup_resources, controller)
            
        return {"response": response}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

async def initialize_workflow(
    qdrant_handler,
    user_email: str,
    discipline_id: str,
    session_id: str,
    mongo_manager
) -> TutorWorkflow:
    """Initialize TutorWorkflow asynchronously"""
    return TutorWorkflow(
        qdrant_handler=qdrant_handler,
        student_email=user_email,
        disciplina=discipline_id,
        session_id=session_id,
        image_collection=mongo_manager.db.image_collection
    )

async def initialize_controller(
    request,
    user_email: str,
    qdrant_handler,
    student_profile: Dict[str, Any],
    study_plan: str,
    tutor_workflow: TutorWorkflow  # Add workflow parameter
) -> ChatController:
    """Initialize ChatController asynchronously"""
    return ChatController(
        session_id=str(request.session_id),
        student_email=user_email,
        disciplina=request.discipline_id,
        qdrant_handler=qdrant_handler,
        image_handler=chat_manager.image_handler,
        retrieval_agent=tutor_workflow,  # Pass the workflow here
        student_profile=student_profile,
        mongo_db_name=MONGO_DB_NAME,
        mongo_uri=MONGO_URI,
        plano_execucao=study_plan
    )

async def cleanup_resources(controller: ChatController):
    """Cleanup resources after request completion"""
    try:
        await controller.cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")

@router_chat.get("/chat_history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 10,
    before: Optional[str] = None,
    current_user=Depends(get_current_user)
):
    """Get chat history with optimized database queries"""
    try:
        async with chat_manager.get_mongo_manager() as mongo_manager:
            collection = mongo_manager.db['chat_history']
            
            # Build query
            query = {
                'session_id': session_id,
                'user_email': current_user["sub"],
            }
            
            if before:
                try:
                    before_datetime = datetime.fromisoformat(before)
                    query['timestamp'] = {'$lt': before_datetime}
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid 'before' timestamp format."
                    )
            
            # Execute query
            cursor = collection.find(query)
            cursor = cursor.sort("timestamp", -1).limit(limit)
            
            messages = []
            async for message_doc in cursor:
                try:
                    history_data = json.loads(message_doc.get('history', '{}'))
                    content = history_data.get('data', {}).get('content', '')
                    role = history_data.get('type', 'unknown')
                    timestamp = message_doc.get('timestamp')
                    
                    messages.append({
                        'role': 'user' if role == 'human' else 'assistant',
                        'content': content,
                        'timestamp': timestamp.isoformat()
                    })
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
            
            messages.reverse()
            return {'messages': messages}
            
    except Exception as e:
        print(f"Error in get_chat_history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@lru_cache(maxsize=100)
def _carregar_json(caminho_arquivo: str) -> Dict[str, Any]:
    """Load JSON file with caching"""
    caminho_absoluto = os.path.abspath(caminho_arquivo)
    try:
        with open(caminho_absoluto, 'r', encoding='utf-8') as arquivo:
            return json.load(arquivo)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erro ao carregar {caminho_arquivo}: {e}")
        return {}