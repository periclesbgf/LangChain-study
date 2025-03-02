# websocket_manager.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Set, Optional
import asyncio
import json
import base64
import io
import numpy as np
import wave
import struct
from datetime import datetime
from openai import AsyncOpenAI
import tempfile
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router_websocket = APIRouter()
client = AsyncOpenAI()

def process_audio_data(audio_base64: str) -> bytes:
    """
    Processa os dados de áudio do formato base64 para PCM16.
    """
    try:
        # Decodifica o áudio de base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Converte para array numpy com o formato correto
        try:
            # Primeiro, tenta converter assumindo que é um array float32
            float32_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Garante que os valores estão no intervalo [-1, 1]
            float32_array = np.clip(float32_array, -1.0, 1.0)
            
            # Converte para int16 com normalização adequada
            int16_array = (float32_array * 32767.0).astype(np.int16)
            
            return int16_array.tobytes()
            
        except ValueError as ve:
            logger.error(f"Erro ao converter áudio: {ve}")
            raise
            
    except Exception as e:
        logger.error(f"Erro no processamento do áudio: {e}")
        raise

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.test_connections: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket, session_id: str = None):
        try:
            await websocket.accept()
            
            if session_id:
                if session_id not in self.active_connections:
                    self.active_connections[session_id] = set()
                self.active_connections[session_id].add(websocket)
                logger.info(f"Session connection established: session {session_id}")
            else:
                self.test_connections.add(websocket)
                logger.info("Test connection established")
            
            logger.info("Connection open")
            
        except Exception as e:
            logger.error(f"Error in connect: {str(e)}")
            raise

    def disconnect(self, websocket: WebSocket, session_id: str = None):
        try:
            if session_id:
                if session_id in self.active_connections:
                    self.active_connections[session_id].remove(websocket)
                    if not self.active_connections[session_id]:
                        del self.active_connections[session_id]
                    logger.info(f"Session disconnected: {session_id}")
            else:
                self.test_connections.remove(websocket)
                logger.info("Test connection disconnected")
        except Exception as e:
            logger.error(f"Error in disconnect: {str(e)}")

    async def process_audio(self, audio_data: str, session_id: Optional[str] = None, websocket: Optional[WebSocket] = None):
        try:
            # Processa o áudio
            pcm16_data = process_audio_data(audio_data)
            
            # Cria arquivo WAV temporário
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_filename = temp_wav.name
                
                with wave.open(temp_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 2 bytes por amostra (16 bits)
                    wav_file.setframerate(44100)  # taxa de amostragem
                    wav_file.writeframes(pcm16_data)

            try:
                # Envia para a OpenAI
                with open(temp_filename, 'rb') as audio_file:
                    transcription = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="pt",
                        response_format="text"
                    )

                logger.info(f"Transcription: {transcription}")

                # Gera resposta via chat
                chat_response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Você é um assistente prestativo e amigável que responde em português."},
                        {"role": "user", "content": transcription}
                    ]
                )

                # Prepara resposta para o frontend
                response_message = {
                    "type": "response",
                    "transcription": transcription,
                    "response": chat_response.choices[0].message.content,
                    "timestamp": datetime.now().isoformat()
                }

                # Envia resposta
                if session_id:
                    await self.broadcast_to_session(session_id, response_message)
                else:
                    await websocket.send_text(json.dumps(response_message))

            finally:
                # Limpa arquivo temporário
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)

            return True
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            error_message = {
                "type": "error",
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
            if session_id:
                await self.broadcast_to_session(session_id, error_message)
            else:
                await websocket.send_text(json.dumps(error_message))
            return None

    async def broadcast_to_session(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            failed_connections = set()
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting message: {str(e)}")
                    failed_connections.add(connection)
            
            for failed in failed_connections:
                self.disconnect(failed, session_id)

manager = ConnectionManager()

@router_websocket.websocket("/ws/test-voice")
async def test_websocket_endpoint(websocket: WebSocket):
    logger.info("Test connection initiated")
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["type"] == "audio":
                await manager.process_audio(data["data"], websocket=websocket)
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in test websocket endpoint: {str(e)}")
        manager.disconnect(websocket)

@router_websocket.websocket("/ws/voice/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["type"] == "audio":
                await manager.process_audio(data["data"], session_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {str(e)}")
        manager.disconnect(websocket, session_id)

@router_websocket.websocket("/ws/chat/{session_id}")
async def chat_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming chat responses.
    This allows for immediate feedback and chunk-based responses from the tutor.
    """
    from api.controllers.chat_controller import ChatController
    from utils import MONGO_URI, MONGO_DB_NAME
    from database.mongo_database_manager import MongoDatabaseManager, MongoPDFHandler
    from database.vector_db import QdrantHandler, Embeddings, TextSplitter
    from agent.image_handler import ImageHandler
    from agent.agent_test import TutorWorkflow
    
    # Setup needed controllers and dependencies
    mongo_manager = MongoDatabaseManager()
    embeddings = Embeddings()
    image_handler = ImageHandler(os.environ.get("OPENAI_API_KEY"))
    text_splitter = TextSplitter()
    
    # Connect websocket
    await manager.connect(websocket, session_id)
    
    try:
        # Accept initial connection
        logger.info(f"Chat WebSocket connected for session: {session_id}")
        
        # Send initial confirmation
        await websocket.send_text(json.dumps({
            "type": "connection", 
            "status": "connected",
            "session_id": session_id
        }))
        
        # Main message loop
        while True:
            # Receive message
            message_text = await websocket.receive_text()
            data = json.loads(message_text)
            
            # Handle chat message
            if data["type"] == "chat":
                logger.info(f"Received chat message from session {session_id}")
                
                # Extract data
                user_email = data.get("user_email")
                discipline_id = data.get("discipline_id") 
                user_message = data.get("message")
                
                if not user_email or not discipline_id or not user_message:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": "Missing required fields (user_email, discipline_id, message)"
                    }))
                    continue
                
                try:
                    # Initialize required components
                    qdrant_handler = QdrantHandler(
                        embeddings=embeddings.get_embeddings(),
                        text_splitter=text_splitter,
                        image_handler=image_handler,
                        mongo_manager=mongo_manager
                    )
                    
                    pdf_handler = MongoPDFHandler(mongo_manager)
                    
                    # Get student profile and study plan
                    student_profile = await mongo_manager.get_student_profile(
                        email=user_email,
                        collection_name="student_learn_preference"
                    )
                    
                    study_plan = await mongo_manager.get_study_plan(session_id)
                    
                    if not student_profile or not study_plan:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "content": "Could not load student profile or study plan"
                        }))
                        continue
                    
                    # Initialize tutor workflow
                    workflow = TutorWorkflow(
                        qdrant_handler=qdrant_handler,
                        student_email=user_email,
                        disciplina=discipline_id,
                        session_id=session_id,
                        image_collection=mongo_manager.db.image_collection
                    )
                    
                    # Create controller
                    controller = ChatController(
                        session_id=session_id,
                        student_email=user_email,
                        disciplina=discipline_id,
                        qdrant_handler=qdrant_handler,
                        image_handler=image_handler,
                        retrieval_agent=workflow,
                        student_profile=student_profile,
                        mongo_db_name=MONGO_DB_NAME,
                        mongo_uri=MONGO_URI,
                        plano_execucao=study_plan,
                        pdf_handler=pdf_handler
                    )
                    
                    # Get response generator
                    response_generator = controller.handle_user_message(
                        user_input=user_message, 
                        stream=True
                    )
                    
                    # Stream chunks to client
                    async for chunk in response_generator:
                        # Send formatted chunk
                        await websocket.send_text(json.dumps(chunk))
                        
                except Exception as e:
                    logger.error(f"Error processing chat message: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": f"Error processing message: {str(e)}"
                    }))
                    
    except WebSocketDisconnect:
        logger.info(f"Chat WebSocket disconnected: {session_id}")
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"Error in chat websocket endpoint: {str(e)}")
        manager.disconnect(websocket, session_id)