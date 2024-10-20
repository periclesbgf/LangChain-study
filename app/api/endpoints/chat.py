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
import json

router_chat = APIRouter()

@router_chat.post("/chat")
async def chat_endpoint(
    request: MessageRequest = Depends(MessageRequest.as_form),
    current_user=Depends(get_current_user)
):
    try:
        # Inicializa o QdrantHandler e o ImageHandler
        embeddings = Embeddings().get_embeddings()
        qdrant_handler = QdrantHandler(
            url=QDRANT_URL,
            collection_name="student_documents",
            embeddings=embeddings
        )
        image_handler = ImageHandler(OPENAI_API_KEY)
        retrieval_agent = RetrievalAgent(
            qdrant_handler=qdrant_handler,
            embeddings=embeddings,
            disciplina=request.discipline_id,
            student_email=current_user["sub"]
        )
        # Proximo passo: Criar o FORMS para o Perfil do estudante, Criar o GET do perfil do estudante e carregar aqui. inicializar o ChatAgent
        # chat_agent = ChatAgent(
        #     student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina
        controller = ChatController(
            session_id=str(request.session_id),
            student_email=current_user["sub"],
            disciplina=str(request.discipline_id),
            qdrant_handler=qdrant_handler,
            image_handler=image_handler,
            retrieval_agent=retrieval_agent
        )

        print(f"Received message: {request.message}")
        print(f"Received file: {request.file}")

        # Se um arquivo foi enviado, processa-o
        files = []
        if request.file:
            files.append(request.file)

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