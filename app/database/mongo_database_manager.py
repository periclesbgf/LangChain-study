# app/database/mongo_database_manager.py

import motor.motor_asyncio
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from utils import MONGO_DB_NAME, MONGO_URI
import os

class MongoDatabaseManager:
    def __init__(self):
        mongo_uri = MONGO_URI
        db_name = MONGO_DB_NAME
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db['mensagens']

    async def save_message(self, session_id: str, role: str, content: str):
        document = {
            'session_id': session_id,
            'role': role,
            'content': content
        }
        await self.collection.insert_one(document)

    async def get_chat_history(self, session_id: str):
        cursor = self.collection.find({'session_id': session_id}).sort('_id', 1)
        chat_history = []
        async for doc in cursor:
            role = doc['role']
            content = doc['content']
            if role == 'user':
                chat_history.append(HumanMessage(content=content))
            elif role == 'assistant':
                chat_history.append(AIMessage(content=content))
            elif role == 'system':
                chat_history.append(SystemMessage(content=content))
        return chat_history
    
