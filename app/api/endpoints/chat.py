# app/api/endpoints/chat.py

import traceback
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from api.controllers.chat_controller import ChatController
from api.controllers.file_controller import FileController
from api.controllers.auth import get_current_user
from api.endpoints.models import MessageRequest, AccessLevel
from typing import Optional, Dict, Any, Tuple, AsyncGenerator
from datetime import datetime
import json
import os
import logging
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager
import time
from cachetools import TTLCache

# Custom JSON serialization for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def json_serialize(obj):
    """Serialize an object to JSON string, handling datetime objects."""
    return json.dumps(obj, cls=DateTimeEncoder)


def make_json_serializable(obj):
    """Recursively convert an object to be JSON serializable."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict
        return make_json_serializable(obj.__dict__)
    else:
        # Try to convert to string as last resort
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

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

            self.qdrant_handler = QdrantHandler( #ponto de atenção
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

        # Create TutorWorkflow from agent_test.py
        print(f"ENDPOINT: Creating TutorWorkflow for session {session_id}")
        try:
            agent = TutorWorkflow(
                qdrant_handler=qdrant_handler,
                student_email=user_email,
                disciplina=discipline_id,
                session_id=session_id,
                image_collection=self.mongo_manager.db.image_collection
            )
            print(f"ENDPOINT: Successfully created TutorWorkflow: {type(agent)}")
            # Verificar se o método invoke com streaming está disponível
            if hasattr(agent, 'invoke'):
                print(f"ENDPOINT: invoke method exists in agent")
            else:
                print(f"ENDPOINT ERROR: invoke method NOT FOUND in agent")
                print(f"ENDPOINT: Available methods: {dir(agent)}")
        except Exception as e:
            print(f"ENDPOINT ERROR: Failed to create TutorWorkflow: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Store in cache
        self._workflow_cache[cache_key] = agent
        return agent

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
        print("Session ID: ", session_id)
        print("Discipline ID: ", discipline_id)

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
            tutor_workflow=workflow,
            student_profile=student_profile,
            plano_execucao=study_plan
        )

        # Store in cache
        self._controller_cache[cache_key] = controller
        return controller

# Initialize the singleton manager
chat_manager = ChatEndpointManager()

@router_chat.post("/chat")
async def chat_endpoint(
    request: MessageRequest = Depends(MessageRequest.as_form),
    current_user=Depends(get_current_user)
):
    try:
        # Verifica se há arquivos e/ou mensagem de texto
        has_files = bool(request.file)
        has_message = bool(request.message and request.message.strip())
        files = None

        print(f"ENDPOINT: Request contains files: {has_files}, contains message: {has_message}")

        request_dict = {
            "session_id": request.session_id,
            "discipline_id": request.discipline_id
        }
        
        # Se tiver arquivo, processa com o FileController antes de continuar
        if has_files:
            # Inicializa recursos necessários para o FileController
            mongo_manager = MongoDatabaseManager()
            pdf_handler = MongoPDFHandler(mongo_manager)
            qdrant_handler = chat_manager.get_qdrant_handler()
            
            # Cria instância do FileController
            file_controller = FileController(
                mongo_db=mongo_manager,
                pdf_handler=pdf_handler,
                qdrant_handler=qdrant_handler,
                student_email=current_user["sub"],
                disciplina=request.discipline_id,
                session_id=str(request.session_id),
                access_level=AccessLevel.SESSION.value
            )
            
            # Processa o arquivo e obtém seu conteúdo antes de ser consumido pelo controller
            file = request.file
            file_content = await file.read()
            await file.seek(0)
            # Processa o arquivo com o FileController
            file_results = await file_controller.process_files([file])
            files = [file]
            
            # Se não tiver mensagem, cria uma resposta streaming informando sobre o processamento do arquivo
            if not has_message:
                async def file_response_stream():
                    if any(result["status"] == "error" for result in file_results):
                        error_messages = [result["message"] for result in file_results if result["status"] == "error"]
                        message = "Erro ao processar arquivos: " + "; ".join(error_messages)
                        yield json.dumps({"type": "error", "content": message}) + "\n"
                    elif any(result["status"] == "rejected" for result in file_results):
                        reject_messages = [result["message"] for result in file_results if result["status"] == "rejected"]
                        message = "; ".join(reject_messages)
                        yield json.dumps({"type": "warning", "content": message}) + "\n"
                    else:
                        yield json.dumps({"type": "success", "content": "Arquivo processado com sucesso. Você pode fazer perguntas sobre ele agora."}) + "\n"
                        yield json.dumps({"type": "success", "content": "Você pode ver o PDF clicando no botão de mídia (ele fica localizado encima a direita)"}) + "\n"

                
                return StreamingResponse(
                    file_response_stream(),
                    media_type="application/x-ndjson"
                )

        # Se não tem arquivo nem mensagem, retorna erro em formato de stream
        if not has_files and not has_message:
            async def error_stream():
                yield json.dumps({"type": "error", "content": "Nenhuma entrada fornecida. Envie uma mensagem ou um arquivo."}) + "\n"

            return StreamingResponse(
                error_stream(),
                media_type="application/x-ndjson"
            )

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

        (student_profile, study_plan_raw), workflow = await asyncio.gather(
            student_data_task,
            workflow_task
        )

        study_plan = json.dumps(study_plan_raw, cls=DateTimeEncoder)

        controller = await chat_manager.get_or_create_controller(
            request_dict,
            current_user["sub"],
            student_profile,
            study_plan,
            workflow
        )

        # Sempre usa streaming para resposta
        async def response_stream():
            try:
                # Notifica que está processando a mensagem
                yield json.dumps({"type": "processing", "content": "Pensando..."}) + "\n"

                async_generator = await controller.handle_user_message(
                    user_input=request.message,
                    files=files,
                    stream=True
                )

                message_content = []
                chunks_received = 0

                async for chunk in async_generator:
                    chunks_received += 1

                    if chunk.get("type") == "chunk":
                        message_content.append(chunk.get("content", ""))
                    else:
                        print(f"ENDPOINT: Received and forwarding non-text chunk type: {chunk.get('type')}")

                    try:
                        yield json_serialize(chunk) + "\n"
                    except TypeError as e:
                        safe_chunk = make_json_serializable(chunk)
                        yield json.dumps(safe_chunk) + "\n"
            except Exception as stream_error:
                import traceback
                traceback.print_exc()
                yield json.dumps({"type": "error", "content": f"Streaming error: {str(stream_error)}"}) + "\n"

        return StreamingResponse(
            response_stream(),
            media_type="application/x-ndjson"
        )

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
        logger.error(f"Erro ao carregar {caminho_arquivo}: {e}")
        return {}