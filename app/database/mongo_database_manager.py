# app/database/mongo_database_manager.py

import motor.motor_asyncio
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from utils import MONGO_DB_NAME, MONGO_URI
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from langchain.schema import BaseMessage, message_to_dict, messages_from_dict
from pymongo import errors
from datetime import datetime, timezone
import json
import os
from typing import List, Optional


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
    

class CustomMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    def __init__(self, user_email: str, disciplina: str, *args, **kwargs):
        self.user_email = user_email
        self.disciplina = disciplina
        super().__init__(*args, **kwargs)
        # Cria um índice em (session_id, user_email, timestamp) para otimizar consultas
        self.collection.create_index(
            [
                (self.session_id_key, 1),
                ("user_email", 1),
                ("timestamp", 1)
            ],
            name="session_user_timestamp_index",
            unique=False
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to MongoDB with user_email, disciplina, and timestamp."""
        try:
            self.collection.insert_one(
                {
                    self.session_id_key: self.session_id,
                    self.history_key: json.dumps(message_to_dict(message)),
                    "user_email": self.user_email,
                    "disciplina": self.disciplina,
                    "timestamp": datetime.now(timezone.utc)
                }
            )
        except errors.WriteError as err:
            print(f"Error adding message: {err}")

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Append multiple messages to MongoDB with user_email, disciplina, and timestamp."""
        try:
            documents = [
                {
                    self.session_id_key: self.session_id,
                    self.history_key: json.dumps(message_to_dict(message)),
                    "user_email": self.user_email,
                    "disciplina": self.disciplina,
                    "timestamp": datetime.now(timezone.utc)
                }
                for message in messages
            ]
            self.collection.insert_many(documents)
        except errors.WriteError as err:
            print(f"Error adding messages: {err}")

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages filtered by session_id and user_email, sorted by timestamp."""
        try:
            cursor = self.collection.find(
                {
                    self.session_id_key: self.session_id,
                    "user_email": self.user_email
                }
            ).sort("timestamp", 1)
        except errors.OperationFailure as error:
            print(f"Error retrieving messages: {error}")
            return []

        items = [json.loads(document[self.history_key]) for document in cursor]
        messages = messages_from_dict(items)
        return messages