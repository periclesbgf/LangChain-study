# app/api/endpoints/chat.py

import traceback
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from api.controllers.chat_controller import ChatController
from api.controllers.auth import get_current_user
from api.endpoints.models import MessageRequest
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import json
import os
import logging
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager
import time
from cachetools import TTLCache

from utils import MONGO_URI, MONGO_DB_NAME, QDRANT_URL, OPENAI_API_KEY
from database.vector_db import (
    QdrantHandler,
    Embeddings,
    TextSplitter
)
from agent.image_handler import ImageHandler
from database.mongo_database_manager import MongoDatabaseManager, MongoPDFHandler
from agent.agent_test import TutorWorkflow

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router_chat = APIRouter()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class ChatEndpointManager:
    """Singleton for managing shared resources with proper caching"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatEndpointManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.mongo_manager = None
        self.qdrant_handler = None
        self.image_handler = None
        self.embeddings = None
        # TTL cache with 15-minute expiration
        self._cache = TTLCache(maxsize=500, ttl=900)
        # Cache for workflow instances
        self._workflow_cache = TTLCache(maxsize=100, ttl=1800)
        # Cache for controller instances
        self._controller_cache = TTLCache(maxsize=100, ttl=900)
        self._initialized = True
        
        # Preload common dependencies
        self._preload_dependencies()
    
    def _preload_dependencies(self):
        """Preload common dependencies in background"""
        asyncio.create_task(self._async_preload())
    
    async def _async_preload(self):
        """Asynchronously initialize core components"""
        try:
            # Initialize MongoDB connection
            self.mongo_manager = MongoDatabaseManager()
            
            # Initialize image handler and embeddings
            self.image_handler = ImageHandler(OPENAI_API_KEY)
            self.embeddings = Embeddings()
            
            # Initialize Qdrant handler
            self._init_qdrant_handler()
            
            logger.info("Preloaded all dependencies")
        except Exception as e:
            logger.error(f"Error preloading dependencies: {e}")

    @asynccontextmanager
    async def get_mongo_manager(self):
        """Get or create MongoDB manager with connection pooling"""
        if not self.mongo_manager:
            self.mongo_manager = MongoDatabaseManager()
        try:
            yield self.mongo_manager
        except Exception as e:
            logger.error(f"Error with MongoDB manager: {e}")
            raise

    def _init_qdrant_handler(self) -> None:
        """Initialize QdrantHandler"""
        if not self.qdrant_handler:
            if not self.image_handler:
                self.image_handler = ImageHandler(OPENAI_API_KEY)
            if not self.embeddings:
                self.embeddings = Embeddings()

            self.qdrant_handler = QdrantHandler(
                url=QDRANT_URL,
                collection_name="student_documents",
                embeddings=self.embeddings.get_embeddings(),
                text_splitter=TextSplitter(),
                image_handler=self.image_handler,
                mongo_manager=self.mongo_manager
            )

    def get_qdrant_handler(self) -> QdrantHandler:
        """Get QdrantHandler (initialized only once)"""
        if not self.qdrant_handler:
            self._init_qdrant_handler()
        return self.qdrant_handler

    async def get_student_data(
        self,
        user_email: str,
        session_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get student profile and study plan concurrently"""
        start_time = time.time()
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
            
            query_time = time.time() - start_time
            logger.info(f"Student data retrieved in {query_time:.2f}s")

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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get student data with TTL caching"""
        cache_key = self.get_cache_key(user_email, session_id)

        try:
            if cache_key in self._cache:
                logger.info(f"Cache hit for {cache_key}")
                return self._cache[cache_key]
        except Exception:
            # Safely handle cache errors and continue
            logger.warning("Cache error, continuing with database query")

        data = await self.get_student_data(user_email, session_id)
        self._cache[cache_key] = data
        return data
        
    async def get_or_create_workflow(
        self,
        user_email: str,
        discipline_id: str,
        session_id: str
    ) -> TutorWorkflow:
        """Get or create TutorWorkflow with caching"""
        cache_key = f"workflow:{user_email}:{discipline_id}:{session_id}"
        
        try:
            if cache_key in self._workflow_cache:
                logger.info(f"Workflow cache hit for {cache_key}")
                return self._workflow_cache[cache_key]
        except Exception:
            logger.warning("Workflow cache error, creating new instance")
        
        # Initialize MongoDB if needed
        if not self.mongo_manager:
            self.mongo_manager = MongoDatabaseManager()
            
        # Get Qdrant handler
        qdrant_handler = self.get_qdrant_handler()
        
        # Create workflow
        workflow = TutorWorkflow(
            qdrant_handler=qdrant_handler,
            student_email=user_email,
            disciplina=discipline_id,
            session_id=session_id,
            image_collection=self.mongo_manager.db.image_collection
        )
        
        # Store in cache
        self._workflow_cache[cache_key] = workflow
        return workflow
        
    async def get_or_create_controller(
        self,
        request_data: dict,
        user_email: str,
        student_profile: Dict[str, Any],
        study_plan: str,
        workflow: TutorWorkflow,
    ) -> ChatController:
        """Get or create ChatController with caching"""
        session_id = str(request_data.get("session_id", ""))
        discipline_id = request_data.get("discipline_id", "")
        cache_key = f"controller:{user_email}:{discipline_id}:{session_id}"
        
        try:
            if cache_key in self._controller_cache:
                logger.info(f"Controller cache hit for {cache_key}")
                return self._controller_cache[cache_key]
        except Exception:
            logger.warning("Controller cache error, creating new instance")
        
        # Get resources
        qdrant_handler = self.get_qdrant_handler()
        
        # Initialize PDF handler
        if not self.mongo_manager:
            self.mongo_manager = MongoDatabaseManager()
        pdf_handler = MongoPDFHandler(self.mongo_manager)
        
        # Create controller
        controller = ChatController(
            session_id=session_id,
            student_email=user_email,
            disciplina=discipline_id,
            qdrant_handler=qdrant_handler,
            image_handler=self.image_handler,
            retrieval_agent=workflow,
            student_profile=student_profile,
            mongo_db_name=MONGO_DB_NAME,
            mongo_uri=MONGO_URI,
            plano_execucao=study_plan,
            pdf_handler=pdf_handler
        )
        
        # Store in cache
        self._controller_cache[cache_key] = controller
        return controller

# Initialize the singleton manager
chat_manager = ChatEndpointManager()

@router_chat.post("/chat")
async def chat_endpoint(
    request: MessageRequest = Depends(MessageRequest.as_form),
    current_user=Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """High-performance chat endpoint with resource pooling and background processing"""
    start_time = time.time()
    
    try:
        # Convert request to dict for caching
        request_dict = {
            "session_id": request.session_id,
            "discipline_id": request.discipline_id
        }
        
        # Process all setup tasks concurrently
        student_data_task = asyncio.create_task(
            chat_manager.get_cached_student_data(
                current_user["sub"],
                str(request.session_id)
            )
        )
        
        workflow_task = asyncio.create_task(
            chat_manager.get_or_create_workflow(
                current_user["sub"],
                request.discipline_id,
                str(request.session_id)
            )
        )
        
        # Wait for student data and workflow concurrently
        (student_profile, study_plan_raw), workflow = await asyncio.gather(
            student_data_task,
            workflow_task
        )
        
        # Convert study plan to JSON
        study_plan = json.dumps(study_plan_raw, cls=DateTimeEncoder)
        
        # Create or get controller
        controller = await chat_manager.get_or_create_controller(
            request_dict,
            current_user["sub"],
            student_profile,
            study_plan,
            workflow
        )
        
        # Process files in background if present
        files = [request.file] if request.file else []
        if files:
            background_tasks.add_task(controller._process_files, files)
            
        # Process message and get response
        response = await controller.handle_user_message(request.message, files)
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(f"Chat request processed in {total_time:.2f}s")
        
        return {"response": response}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router_chat.get("/chat_history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 10,
    before: Optional[str] = None,
    current_user=Depends(get_current_user)
):
    """Get chat history with optimized database queries and projection"""
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

            # Use projection to limit fields returned
            projection = {
                "timestamp": 1,
                "history": 1,
                "_id": 0
            }
            
            # Execute optimized query
            cursor = collection.find(
                query, 
                projection
            ).sort("timestamp", -1).limit(limit)

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
                    logger.warning(f"Error processing message: {e}")
                    continue

            messages.reverse()
            return {'messages': messages}

    except Exception as e:
        logger.error(f"Error in get_chat_history: {e}", exc_info=True)
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
        logger.error(f"Erro ao carregar {caminho_arquivo}: {e}")
        return {}