# app/api/controllers/chat_controller.py

import base64
import os
import json
import logging
import asyncio
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from logg import logger
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
        tutor_workflow: TutorWorkflow,
        student_profile: dict,
        plano_execucao: dict,
    ):
        """Initialize the controller with optimized resource management"""
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.perfil = student_profile

        if isinstance(plano_execucao, str):
            try:
                self.plano_execucao = json.loads(plano_execucao)
            except json.JSONDecodeError:
                self.plano_execucao = plano_execucao
        else:
            self.plano_execucao = plano_execucao

        logger.info(f"Final plano_execucao type: {type(self.plano_execucao)}")

        self._qdrant_handler = qdrant_handler
        self._tutor_workflow = tutor_workflow

        self._llm = None
        self._embeddings = None
        self._text_splitter = None
        self._mongo_client = None
        self._db = None
        self._image_collection = None
        self._chain = None
        self._chain_with_history = None
        self._chat_history = None

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
            #print("Chat history:", self._chat_history)
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
        files=None,
        stream: bool = False
    ) -> Union[str, Dict[str, Any], AsyncGenerator[Dict[str, str], None]]:
        """
        Process user message and files with optimized execution.
        When stream=True, returns an async generator that yields response chunks.
        This method is already async and must be awaited.
        """
        start_time = time.time()
        logger.info(f"[USER_MESSAGE] {self.student_email} | disciplina={self.disciplina} | session={self.session_id}")
        
        try:
            if not user_input:
                if stream:
                    return self._create_simple_generator({"type": "error", "content": "Nenhuma entrada fornecida."})
                return "Nenhuma entrada fornecida."

            interaction_key = f"{self.session_id}:{hash(user_input)}"

            if not stream and user_input and interaction_key in self._response_cache:
                logger.info(f"Cache hit for message: {interaction_key}")
                return self._response_cache[interaction_key]

            try:
                current_history = self.chat_history.messages[-MAX_HISTORY_MESSAGES:] if self.chat_history.messages else []
                print(f"[CHAT_CONTROLLER] Retrieved chat history: {len(current_history)} messages")
                for i, msg in enumerate(current_history):
                    print(f"[CHAT_CONTROLLER] Message {i}: type={type(msg).__name__}, content preview: {str(msg.content)[:50]}...")
            except Exception as hist_error:
                print(f"[CHAT_CONTROLLER] ERROR retrieving chat history: {str(hist_error)}")
                import traceback
                traceback.print_exc()
                current_history = []

            if user_input:
                if stream:
                    generator = self._create_streaming_response(user_input, current_history)
                    return generator

                workflow_response = await self._tutor_workflow.invoke(
                    query=user_input,
                    student_profile=self.perfil,
                    current_plan=self.plano_execucao,
                    chat_history=current_history
                )

                if isinstance(workflow_response, dict):
                    messages = workflow_response.get("messages", [])
                    if messages:
                        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
                        if ai_messages:
                            final_message = ai_messages[-1]

                            response = await self._process_response_content(
                                final_message, user_input
                            )

                            self._response_cache[interaction_key] = response

                            self._analytics["interaction_count"] += 1
                            self._analytics["last_response_time"] = time.time() - start_time
                            
                            logger.info(f"[CHAT_RESPONSE] {self.student_email} | session={self.session_id} | time={self._analytics['last_response_time']:.2f}s")
                            return response

            if stream:
                return self._create_simple_generator({"type": "complete", "content": "Processamento concluído."})
            return "Processamento concluído."

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            if stream:
                return self._create_simple_generator({"type": "error", "content": f"Ocorreu um erro ao processar sua mensagem: {str(e)}"})
            return "Ocorreu um erro ao processar sua mensagem. Por favor, tente novamente."

    def _create_simple_generator(self, message: Dict[str, str]):
        """Helper to create a simple async generator with a single message"""
        async def generator():
            yield message
        return generator()

    async def _create_streaming_response(self, user_input: str, current_history: List[BaseMessage]) -> AsyncGenerator[Dict[str, str], None]:
        """
        Creates a streaming response for the frontend with chunked delivery.
        Uses TutorWorkflow's invoke method with stream=True to generate streaming responses.
        """
        start_time = time.time()

        try:
            yield {"type": "processing", "content": "Entendendo a pergunta..."}

            full_text = ""
            has_image = False
            image_data = None

            stream_generator = await self._tutor_workflow.invoke(
                query=user_input,
                student_profile=self.perfil,
                current_plan=self.plano_execucao,
                chat_history=current_history,
                stream=True
            )

            chunks_count = 0

            async for chunk in stream_generator:
                chunks_count += 1
                chunk_type = chunk.get("type", "unknown")

                if chunk_type == "chunk":
                    chunk_content = chunk.get("content", "")
                    full_text += chunk_content
                elif chunk_type == "image":
                    has_image = True
                    full_text = chunk.get("content", "")  # Texto associado à imagem
                    image_data = chunk.get("image")
                    print(f"CHAT_CONTROLLER: Received image chunk")
                elif chunk_type == "error":
                    yield chunk
                    break
                yield chunk


            processing_time = time.time() - start_time
            logger.info(f"[CHAT_STREAM] {self.student_email} | session={self.session_id} | time={processing_time:.2f}s")

            if full_text:
                print(f"CHAT_CONTROLLER: Salvando mensagem completa no histórico (tamanho: {len(full_text)})")
                if has_image and image_data:
                    multimodal_content = {
                        "type": "multimodal",
                        "content": full_text,
                        "image": image_data
                    }
                    await self._save_chat_history(
                        HumanMessage(content=user_input),
                        AIMessage(content=json.dumps(multimodal_content))
                    )
                else:
                    await self._save_chat_history(
                        HumanMessage(content=user_input),
                        AIMessage(content=full_text)
                    )

        except Exception as e:
            logger.error(f"Error in streaming generator: {e}")
            import traceback
            traceback.print_exc()
            yield {"type": "error", "content": f"Ocorreu um erro ao processar sua mensagem: {str(e)}"}

    async def _process_response_content(
        self,
        message: BaseMessage,
        user_input: str
    ) -> Union[str, Dict[str, Any]]:
        """Process message content handling multimodal responses"""
        try:
            content = json.loads(message.content) if isinstance(message.content, str) else message.content

            if isinstance(content, dict) and content.get("type") == "image":
                image_bytes = content.get("image_bytes")
                if image_bytes:
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    response_data = {
                        "type": "image",
                        "content": content.get("description", ""),
                        "image": f"data:image/jpeg;base64,{image_base64}"
                    }

                    await self._save_chat_history(
                        HumanMessage(content=user_input),
                        AIMessage(content=json.dumps(response_data))
                    )
                    return response_data
        except (json.JSONDecodeError, AttributeError):
            raise ValueError("Invalid response content format")

        await self._save_chat_history(
            HumanMessage(content=user_input),
            message
        )
        return message.content

    async def _save_chat_history(self, user_message: HumanMessage, ai_message: AIMessage):
        """Save chat history with optimized batching and proper format validation"""
        try:

            if hasattr(ai_message, 'content') and isinstance(ai_message.content, str):
                try:
                    if ai_message.content.startswith('{"type":"multimodal"'):
                        multimodal_data = json.loads(ai_message.content)
                        if multimodal_data.get("type") == "multimodal":
                            text_content = multimodal_data.get("content", "")
                            ai_message = AIMessage(
                                content=text_content,
                                additional_kwargs={"image_data": multimodal_data.get("image")}
                            )
                            raise ValueError("Multimodal content detected, splitting into separate messages")
                except json.JSONDecodeError:
                    raise ValueError("Invalid multimodal content format")
                except Exception as norm_error:
                    raise ValueError(f"Error normalizing multimodal content: {norm_error}")
            self.chat_history.add_message(user_message)
            self.chat_history.add_message(ai_message)
            print(f"[CHAT_CONTROLLER]   Messages successfully added to history")

        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
            import traceback
            traceback.print_exc()

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
