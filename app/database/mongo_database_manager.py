# app/database/mongo_database_manager.py

import motor.motor_asyncio
from utils import MONGO_DB_NAME, MONGO_URI
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.schema import BaseMessage, message_to_dict, messages_from_dict
from pymongo import errors
from datetime import datetime, timezone
import json
import os
from typing import List, Optional, Dict, Any


class MongoDatabaseManager:
    def __init__(self):
        mongo_uri = MONGO_URI
        db_name = MONGO_DB_NAME
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]

    async def create_student_profile(self, email: str, profile_data: Dict[str, Any]) -> Optional[str]:
        """
        Cria um novo perfil de estudante no MongoDB.
        """
        collection = self.db['student_learn_preference']
        try:
            # Verifica se o perfil já existe
            existing_profile = await collection.find_one({"user_email": email})
            if existing_profile:
                print(f"Perfil já existe para o email: {email}")
                return None

            # Insere o novo perfil
            result = await collection.insert_one(profile_data)
            print(f"Perfil criado com sucesso para o email: {email}")
            return str(result.inserted_id)

        except errors.PyMongoError as e:
            print(f"Erro ao criar perfil: {e}")
            return None

    async def get_student_profile(self, email: str, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Busca o perfil de um estudante pelo e-mail.
        """
        collection = self.db[collection_name]
        try:
            profile = await collection.find_one({"Email": email})  # Correção aqui
            if not profile:
                print(f"Perfil não encontrado para o email: {email}")
                return None
            return profile
        except errors.PyMongoError as e:
            print(f"Erro ao buscar perfil: {e}")
            return None

    async def update_student_profile(self, email: str, profile_data: Dict[str, Any], collection_name: str):
        """
        Atualiza o perfil de um estudante pelo e-mail.
        """
        collection = self.db[collection_name]
        try:
            result = await collection.update_one(
                {"user_email": email},
                {"$set": {"profile_data": profile_data, "updated_at": datetime.now(timezone.utc)}}
            )
            if result.matched_count == 0:
                print(f"Perfil não encontrado para o email: {email}")
                return False
            print(f"Perfil atualizado com sucesso para o email: {email}")
            return True
        except errors.PyMongoError as e:
            print(f"Erro ao atualizar perfil: {e}")
            return False

    async def delete_student_profile(self, email: str, collection_name: str):
        """
        Exclui o perfil de um estudante pelo e-mail.
        """
        collection = self.db[collection_name]
        try:
            result = await collection.delete_one({"user_email": email})
            if result.deleted_count == 0:
                print(f"Perfil não encontrado para o email: {email}")
                return False
            print(f"Perfil excluído com sucesso para o email: {email}")
            return True
        except errors.PyMongoError as e:
            print(f"Erro ao excluir perfil: {e}")
            return False

class CustomMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    def __init__(self, user_email: str, disciplina: str, *args, **kwargs):
        self.user_email = user_email
        self.disciplina = disciplina
        super().__init__(*args, **kwargs)

        # Verifica e cria a coleção se necessário
        self.ensure_collection_exists()

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

    def ensure_collection_exists(self):
        """
        Verifica se a coleção existe no banco de dados.
        Se não existir, cria a coleção.
        """
        try:
            db = self.collection.database
            if self.collection.name not in db.list_collection_names():
                db.create_collection(self.collection.name)
                print(f"Collection '{self.collection.name}' created successfully.")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise

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
