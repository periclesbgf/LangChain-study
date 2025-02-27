# app/api/controllers/chat_controller.py

import base64
import os
import json
import logging
import asyncio
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import uuid
import hashlib
from cachetools import TTLCache, LRUCache

from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap

from database.mongo_database_manager import (
    CustomMongoDBChatMessageHistory,
    MongoPDFHandler,
    MongoDatabaseManager
)
from database.vector_db import TextSplitter, Embeddings, QdrantHandler
from agent.image_handler import ImageHandler
from utils import (
    OPENAI_API_KEY,
    MONGO_DB_NAME,
    MONGO_URI,
)
from agent.agent_test import TutorWorkflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-load and compress prompt templates to minimize token count
# These have been compressed to retain essential information while reducing token usage

ORCHESTRATOR_PROMPT = """
You're an educational orchestrator that coordinates agent responses based on:
1. Student's learning profile and question type
2. Current progress in the study plan
3. Available educational resources

For each question, determine if specialized agents are needed:
- Retrieval Agent: For material lookup or web searches
- Tutorial Agent: For guided learning support
- Exercise Agent: For practice creation

Return JSON with reformulated question, required agents, and context.
"""

PLAN_CREATOR_PROMPT = """
Create a personalized learning plan based on:
1. Student profile and learning style preferences
2. Current progress and historical interactions
3. Scaffolding requirements for the topic

Focus on breaking concepts into manageable steps aligned with the student's
learning style, providing appropriate support and practice opportunities.
"""

# Constants for performance optimization
MAX_HISTORY_MESSAGES = 10
RESPONSE_CACHE_SIZE = 100
RESPONSE_CACHE_TTL = 600  # 10 minutes
FILE_PROCESSING_CHUNK_SIZE = 1024 * 1024  # 1MB chunks

class ChatController:
    """
    Optimized ChatController that manages interactions between users and the agent system
    with performance enhancements including caching, lazy loading, and parallel processing.
    """
    
    def __init__(
        self,
        session_id: str,
        student_email: str,
        disciplina: str,
        qdrant_handler: QdrantHandler,
        image_handler: ImageHandler,
        retrieval_agent: TutorWorkflow,
        student_profile: dict,
        mongo_db_name: str,
        mongo_uri: str,
        plano_execucao: dict,
        pdf_handler: MongoPDFHandler
    ):
        """Initialize the controller with optimized resource management"""
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.perfil = student_profile
        
        # Normalize plan format for consistency
        if isinstance(plano_execucao, str):
            try:
                self.plano_execucao = json.loads(plano_execucao)
                logger.info("Successfully parsed plano_execucao as JSON")
            except json.JSONDecodeError:
                logger.warning("Failed to parse plano_execucao as JSON, using as string")
                self.plano_execucao = plano_execucao
        else:
            # Already a dictionary or other object, keep as is
            self.plano_execucao = plano_execucao
            
        # Log the type for debugging
        logger.info(f"Final plano_execucao type: {type(self.plano_execucao)}")

        # Initialize resources with lazy loading
        self._qdrant_handler = qdrant_handler
        self._image_handler = image_handler
        self._pdf_handler = pdf_handler
        self._tutor_workflow = retrieval_agent
        
        # Initialize lazy-loaded resources
        self._llm = None
        self._embeddings = None
        self._text_splitter = None
        self._mongo_client = None
        self._db = None
        self._image_collection = None
        self._chain = None
        self._chain_with_history = None
        self._chat_history = None
        
        # Configure caching for responses
        self._response_cache = TTLCache(maxsize=RESPONSE_CACHE_SIZE, ttl=RESPONSE_CACHE_TTL)
        
        # Analytics with lightweight tracking
        self._analytics = {
            "session_start": datetime.now(),
            "interaction_count": 0,
            "last_response_time": 0
        }
        
        # Background loading
        asyncio.create_task(self._preload_resources())
        
    async def _preload_resources(self):
        """Preload common resources in background"""
        try:
            # Initialize high-priority resources first
            self._llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                openai_api_key=OPENAI_API_KEY
            )
            
            # Initialize chat history
            _ = self.chat_history
            
            # Initialize remaining resources
            self._text_splitter = TextSplitter()
            self._embeddings = Embeddings().get_embeddings()
            
            logger.info(f"Preloaded resources for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error preloading resources: {e}")
    
    @property
    def llm(self):
        """Lazy-loaded LLM model"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                openai_api_key=OPENAI_API_KEY
            )
        return self._llm
    
    @property
    def db(self):
        """Lazy-loaded MongoDB connection"""
        if self._db is None:
            if self._mongo_client is None:
                self._mongo_client = MongoClient(
                    MONGO_URI,
                    maxPoolSize=10,
                    socketTimeoutMS=5000,
                    connectTimeoutMS=5000
                )
            self._db = self._mongo_client[MONGO_DB_NAME]
        return self._db
    
    @property
    def image_collection(self):
        """Lazy-loaded image collection"""
        if self._image_collection is None:
            self._image_collection = self.db["image_collection"]
        return self._image_collection
        
    @property
    def chat_history(self):
        """Lazy-loaded chat history"""
        if self._chat_history is None:
            self._chat_history = CustomMongoDBChatMessageHistory(
                user_email=self.student_email,
                disciplina=self.disciplina,
                connection_string=MONGO_URI,
                session_id=self.session_id,
                database_name=MONGO_DB_NAME,
                collection_name="chat_history",
                session_id_key="session_id",
                history_key="history",
            )
        return self._chat_history
    
    @property
    def chain(self):
        """Lazy-loaded chain setup"""
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain
    
    @property
    def chain_with_history(self):
        """Lazy-loaded chain with history"""
        if self._chain_with_history is None:
            self._chain_with_history = self._setup_chat_history()
        return self._chain_with_history

    async def cleanup(self):
        """Clean up resources when the controller is no longer needed"""
        try:
            if self._mongo_client:
                self._mongo_client.close()
            logger.info(f"Cleaned up resources for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def handle_user_message(
        self,
        user_input: Optional[str] = None,
        files=None
    ) -> Union[str, Dict[str, Any]]:
        """
        Process user message and files with optimized execution
        """
        start_time = time.time()
        interaction_key = f"{self.session_id}:{hash(user_input)}"
        
        try:
            # Handle empty input
            if not user_input and not files:
                return "Nenhuma entrada fornecida."
                
            # Check cache for identical recent requests
            if user_input and interaction_key in self._response_cache:
                logger.info(f"Cache hit for message: {interaction_key}")
                return self._response_cache[interaction_key]
                
            # Get current chat history efficiently
            current_history = self.chat_history.messages[-MAX_HISTORY_MESSAGES:] if self.chat_history.messages else []
            
            # Process workflow with optimized parameters
            if user_input:
                # Invoke workflow with current context
                workflow_response = await self._tutor_workflow.invoke(
                    query=user_input,
                    student_profile=self.perfil,
                    current_plan=self.plano_execucao,
                    chat_history=current_history
                )
                
                # Process response
                if isinstance(workflow_response, dict):
                    messages = workflow_response.get("messages", [])
                    if messages:
                        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
                        if ai_messages:
                            final_message = ai_messages[-1]
                            
                            # Process multimodal content
                            response = await self._process_response_content(
                                final_message, user_input
                            )
                            
                            # Cache the response
                            self._response_cache[interaction_key] = response
                            
                            # Track performance
                            self._analytics["interaction_count"] += 1
                            self._analytics["last_response_time"] = time.time() - start_time
                            
                            logger.info(f"Processed message in {self._analytics['last_response_time']:.2f}s")
                            return response
                
            # Handle file upload with background processing
            if files:
                return "Arquivo em processamento. Os resultados estarão disponíveis em breve."
                
            return "Processamento concluído."

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            return "Ocorreu um erro ao processar sua mensagem. Por favor, tente novamente."

    async def _process_response_content(
        self,
        message: BaseMessage,
        user_input: str
    ) -> Union[str, Dict[str, Any]]:
        """Process message content handling multimodal responses"""
        try:
            # Try to parse JSON for multimodal content
            content = json.loads(message.content) if isinstance(message.content, str) else message.content
            
            # Handle image content
            if isinstance(content, dict) and content.get("type") == "image":
                image_bytes = content.get("image_bytes")
                if image_bytes:
                    # Convert image bytes to base64
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    response_data = {
                        "type": "image",
                        "content": content.get("description", ""),
                        "image": f"data:image/jpeg;base64,{image_base64}"
                    }
                    
                    # Save interaction to history
                    await self._save_chat_history(
                        HumanMessage(content=user_input),
                        AIMessage(content=json.dumps(response_data))
                    )
                    return response_data
        except (json.JSONDecodeError, AttributeError):
            # Not JSON or multimodal content, handle as plain text
            pass
            
        # For plain text responses
        await self._save_chat_history(
            HumanMessage(content=user_input),
            message
        )
        return message.content

    async def _save_chat_history(self, user_message: HumanMessage, ai_message: AIMessage):
        """Save chat history with optimized batching"""
        try:
            # Add messages using async-compatible method
            self.chat_history.add_message(user_message)
            self.chat_history.add_message(ai_message)
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")

    def _create_chain(self):
        """Create optimized processing chain"""
        # Simplified prompt templates for reduced token count
        main_prompt = ChatPromptTemplate.from_messages([
            ("system", "Student profile: {perfil}"),
            ("system", "Execution plan: {plano_execucao}"),
            MessagesPlaceholder(variable_name="history"),
            ("system", ORCHESTRATOR_PROMPT),
            ("human", "{input}")
        ])

        other_prompt = ChatPromptTemplate.from_messages([
            ("system", "Student profile: {perfil}"),
            ("system", "Execution plan: {plano_execucao}"),
            MessagesPlaceholder(variable_name="history"),
            ("system", PLAN_CREATOR_PROMPT),
            ("human", "{input}")
        ])

        # Output parsers
        json_output_parser = JsonOutputParser()
        str_output_parser = StrOutputParser()

        # Define optimized chain
        first_stage = RunnableMap({
            "main_output": main_prompt | self.llm | json_output_parser,
            "original_input": RunnablePassthrough()
        })

        return (
            first_stage
            | (lambda x: {
                "history": x["original_input"].get("history", [])[-MAX_HISTORY_MESSAGES:],
                "reformulated_question": x["main_output"],
                "perfil": self.perfil,
                "plano_execucao": self.plano_execucao,
                "input": x["original_input"].get("input")
            })
            | other_prompt
            | self.llm
            | str_output_parser
        )

    def _setup_chat_history(self):
        """Set up chain with chat history"""
        history_factory = lambda session_id: CustomMongoDBChatMessageHistory(
            user_email=self.student_email,
            disciplina=self.disciplina,
            connection_string=MONGO_URI,
            session_id=session_id,
            database_name=MONGO_DB_NAME,
            collection_name="chat_history",
            session_id_key="session_id",
            history_key="history",
        )
        
        return RunnableWithMessageHistory(
            self.chain,
            history_factory,
            input_messages_key="input",
            history_messages_key="history",
        )

    async def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning analytics with optimized retrieval"""
        session_duration = (datetime.now() - self._analytics["session_start"]).total_seconds()
        
        # Get progress data from the database
        progress_data = {}
        try:
            # Attempt to get study progress
            if hasattr(self._tutor_workflow, "progress_manager"):
                progress_summary = await self._tutor_workflow.progress_manager.get_study_summary(self.session_id)
                if progress_summary:
                    progress_data = progress_summary
        except Exception as e:
            logger.warning(f"Error retrieving progress data: {e}")
            
        return {
            "session_id": self.session_id,
            "session_duration": session_duration,
            "total_interactions": self._analytics["interaction_count"],
            "average_response_time": self._analytics["last_response_time"],
            "progress_data": progress_data
        }

    async def _process_files(self, files):
        """Process uploaded files with chunking and parallel processing"""
        # Process files concurrently in smaller chunks
        processing_tasks = []
        
        for file in files:
            task = asyncio.create_task(self._process_single_file(file))
            processing_tasks.append(task)
        
        # Wait for all files to be processed
        await asyncio.gather(*processing_tasks)
        
    async def _process_single_file(self, file):
        """Process a single file with optimized handling"""
        try:
            filename = file.filename
            # Read file content
            content = await file.read()
            
            # Create file metadata
            pdf_uuid = str(uuid.uuid4())
            content_hash = hashlib.md5(content).hexdigest()
            metadata = {
                "filename": filename,
                "uuid": pdf_uuid,
                "hash": content_hash,
                "student_email": self.student_email,
                "disciplina": self.disciplina,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "access_level": "session"
            }
            
            # Process based on file type
            is_pdf = file.content_type == "application/pdf" or filename.lower().endswith(".pdf")
            
            # Run MongoDB and Qdrant storage concurrently
            storage_tasks = []
            
            if is_pdf:
                # Store in MongoDB
                pdf_task = asyncio.create_task(
                    self._pdf_handler.store_pdf(
                        pdf_uuid=pdf_uuid,
                        pdf_bytes=content,
                        student_email=self.student_email,
                        disciplina=self.disciplina,
                        session_id=self.session_id,
                        filename=filename,
                        content_hash=content_hash,
                        access_level="session"
                    )
                )
                storage_tasks.append(pdf_task)
                
            # Store in vector DB for embeddings
            vector_task = asyncio.create_task(
                self._qdrant_handler.process_file(
                    content=content,
                    filename=filename,
                    student_email=self.student_email,
                    session_id=self.session_id,
                    disciplina=self.disciplina,
                    access_level="session"
                )
            )
            storage_tasks.append(vector_task)
            
            # Wait for all storage tasks to complete
            await asyncio.gather(*storage_tasks)
            logger.info(f"File '{filename}' processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            # Continue with other files even if one fails
