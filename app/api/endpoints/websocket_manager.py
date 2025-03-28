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

# Importar logger do sistema principal
from logg import logger

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
                logger.info(f"[WEBSOCKET_CONNECT] Nova conexão para sessão {session_id}")
            else:
                self.test_connections.add(websocket)
                logger.info("[WEBSOCKET_TEST_CONNECT] Nova conexão de teste")
            
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
                    logger.info(f"[WEBSOCKET_DISCONNECT] Conexão encerrada para sessão {session_id}")
            else:
                self.test_connections.remove(websocket)
                logger.info("[WEBSOCKET_TEST_DISCONNECT] Conexão de teste encerrada")
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

                logger.info(f"[AUDIO_TRANSCRIPTION] Sessão {session_id or 'teste'} - Áudio transcrito com sucesso")

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