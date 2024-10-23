# app/api/endpoints/chat.py

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from api.controllers.chat_controller import ChatController
from api.controllers.auth import get_current_user
from api.endpoints.models import MessageRequest
from pymongo import MongoClient
from utils import MONGO_URI, MONGO_DB_NAME, QDRANT_URL
from typing import Optional
from database.vector_db import QdrantHandler
from agent.image_handler import ImageHandler
from database.vector_db import Embeddings
from datetime import datetime
from utils import OPENAI_API_KEY
from agent.agents import RetrievalAgent, ChatAgent
from database.mongo_database_manager import MongoDatabaseManager
from agent.agents import ChatAgent
#from agent.agent_test import TutorWorkflow

import json
import os

router_chat = APIRouter()

# Update the chat endpoint to include TutorWorkflow
@router_chat.post("/chat")
async def chat_endpoint(
    request: MessageRequest = Depends(MessageRequest.as_form),
    current_user=Depends(get_current_user)
):
    try:
        # Existing initialization code remains the same
        embeddings = Embeddings().get_embeddings()
        qdrant_handler = QdrantHandler(
            url=QDRANT_URL,
            collection_name="student_documents",
            embeddings=embeddings
        )
        image_handler = ImageHandler(OPENAI_API_KEY)

        # Get student profile
        mongo_manager = MongoDatabaseManager()
        student_profile = await mongo_manager.get_student_profile(
            email=current_user["sub"],
            collection_name="student_learn_preference"
        )

        if not student_profile:
            raise HTTPException(status_code=404, detail="Perfil do estudante não encontrado.")
        print(f"Student profile: {student_profile}")

        # # Initialize TutorWorkflow
        # tutor_workflow = TutorWorkflow(
        #     qdrant_handler=qdrant_handler,
        #     student_email=current_user["sub"],
        #     disciplina=request.discipline_id
        # )

        # Initialize other agents as before
        chat_agent = ChatAgent(
            student_profile=student_profile,
            execution_plan=_carregar_json("resources/plano_acao.json"),
            mongo_uri=MONGO_URI,
            database_name=MONGO_DB_NAME,
            session_id=str(request.session_id),
            user_email=current_user["sub"],
            disciplina=request.discipline_id
        )

        retrieval_agent = RetrievalAgent(
            qdrant_handler=qdrant_handler,
            embeddings=embeddings,
            disciplina=request.discipline_id,
            session_id=str(request.session_id),
            student_email=current_user["sub"]
        )

        # Initialize ChatController with all components
        controller = ChatController(
            session_id=str(request.session_id),
            student_email=current_user["sub"],
            disciplina=request.discipline_id,
            qdrant_handler=qdrant_handler,
            image_handler=image_handler,
            retrieval_agent=retrieval_agent,
            student_profile=student_profile,
            mongo_db_name=MONGO_DB_NAME,
            mongo_uri=MONGO_URI,
        )

        print(f"Received message: {request.message}")
        print(f"Received file: {request.file}")

        # Process user message
        files = [request.file] if request.file else []
        response = await controller.handle_user_message(request.message, files)
        print(f"Response: {response}")

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
    

def _carregar_json(caminho_arquivo: str):
    """Carrega dados de um arquivo JSON."""
    caminho_absoluto = os.path.abspath(caminho_arquivo)
    try:
        with open(caminho_absoluto, 'r', encoding='utf-8') as arquivo:
            return json.load(arquivo)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erro ao carregar {caminho_arquivo}: {e}")
        return {}