# app/database/mongo_database_manager.py

import traceback
from bson import Binary
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
        """Inicializa a conexão com o banco de dados MongoDB."""
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]

    async def get_collection(self, collection_name: str):
        """
        Obtém uma coleção do MongoDB.
        
        Args:
            collection_name: Nome da coleção
            
        Returns:
            AsyncIOMotorCollection: Coleção do MongoDB
        """
        try:
            # Verifica se a coleção existe
            collections = await self.db.list_collection_names()
            if collection_name not in collections:
                print(f"[MONGO] Criando nova coleção: {collection_name}")
                # Cria a coleção se não existir
                await self.db.create_collection(collection_name)
            
            return self.db[collection_name]
        except Exception as e:
            print(f"[MONGO] Erro ao obter coleção {collection_name}: {e}")
            raise
    async def close(self):
        """Fecha a conexão com o MongoDB."""
        try:
            self.client.close()
            print("[MONGO] Conexão fechada")
        except Exception as e:
            print(f"[MONGO] Erro ao fechar conexão: {e}")
            raise

    ### MÉTODOS DE PERFIL DE ESTUDANTE (JÁ EXISTENTES) ###
    async def create_student_profile(self, email: str, profile_data: Dict[str, Any]) -> Optional[str]:
        """Cria um novo perfil de estudante no MongoDB."""
        collection = self.db['student_learn_preference']
        try:
            existing_profile = await collection.find_one({"user_email": email})
            if existing_profile:
                print(f"Perfil já existe para o email: {email}")
                return None

            result = await collection.insert_one(profile_data)
            print(f"Perfil criado com sucesso para o email: {email}")
            return str(result.inserted_id)
        except errors.PyMongoError as e:
            print(f"Erro ao criar perfil: {e}")
            return None

    async def get_student_profile(self, email: str, collection_name: str) -> Optional[Dict[str, Any]]:
        """Busca o perfil de um estudante pelo e-mail."""
        collection = self.db[collection_name]
        try:
            profile = await collection.find_one({"Email": email})
            if not profile:
                print(f"Perfil não encontrado para o email: {email}")
                return None
            return profile
        except errors.PyMongoError as e:
            print(f"Erro ao buscar perfil: {e}")
            return None

    async def update_student_profile(self, email: str, profile_data: Dict[str, Any], collection_name: str) -> bool:
        """Atualiza o perfil de um estudante pelo e-mail."""
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

    async def delete_student_profile(self, email: str, collection_name: str) -> bool:
        """Exclui o perfil de um estudante pelo e-mail."""
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

    ### MÉTODOS DO PLANO DE ESTUDOS ###
    async def create_study_plan(self, plan_data: Dict[str, Any]) -> Optional[str]:
        """Cria um novo plano de estudos no MongoDB."""
        collection = self.db['study_plans']
        try:
            result = await collection.insert_one(plan_data)
            return str(result.inserted_id)
        except errors.PyMongoError as e:
            print(f"Erro ao criar plano de estudos: {e}")
            return None

    async def get_study_plan(self, id_sessao: str) -> Optional[Dict[str, Any]]:
        """Busca um plano de estudos pelo id_sessao."""
        collection = self.db['study_plans']
        try:
            print(f"[DEBUG] Buscando plano para sessão: {id_sessao}")
            plan = await collection.find_one({"id_sessao": str(id_sessao)})
            if plan:
                plan["_id"] = str(plan["_id"])  # Converte ObjectId para string
                print(f"[DEBUG] Plano encontrado: {plan}")
                return plan
            print(f"[DEBUG] Nenhum plano encontrado para sessão: {id_sessao}")
            return None
        except errors.PyMongoError as e:
            print(f"[DEBUG] Erro ao buscar plano de estudos: {e}")
            return None

    async def update_study_plan(self, id_sessao: str, updated_data: Dict[str, Any]) -> bool:
        """Atualiza um plano de estudos existente."""
        collection = self.db['study_plans']
        try:
            result = await collection.update_one(
                {"id_sessao": id_sessao},
                {"$set": updated_data}
            )
            return result.matched_count > 0
        except errors.PyMongoError as e:
            print(f"Erro ao atualizar plano de estudos: {e}")
            return False

    async def delete_study_plan(self, id_sessao: str) -> bool:
        """Exclui um plano de estudos pelo id_sessao."""
        collection = self.db['study_plans']
        try:
            result = await collection.delete_one({"id_sessao": id_sessao})
            return result.deleted_count > 0
        except errors.PyMongoError as e:
            print(f"Erro ao excluir plano de estudos: {e}")
            return False

    async def get_sessions_without_plan(self, student_email: str, study_sessions) -> List[Dict[str, Any]]:
        """
        Recupera as sessões de estudo do estudante que não possuem plano de execução.
        """
        try:
            if not study_sessions:
                print(f"Nenhuma sessão encontrada para o estudante: {student_email}")
                return []

            # Convert tuple data to dictionaries
            study_sessions_list = [
                {
                    'IdSessao': session[0],  # Accessing tuple elements by index
                    'IdEstudante': session[1],
                    'IdCurso': session[2],
                    'Assunto': session[3],
                    'Inicio': session[4],
                    'Fim': session[5],
                    'Produtividade': session[6],
                    'FeedbackDoAluno': session[7]
                }
                for session in study_sessions
            ]

            # Extract session IDs as strings
            session_ids = [str(session['IdSessao']) for session in study_sessions_list]

            # Access the MongoDB collection for study plans
            plans_collection = self.db['study_plans']

            # Find all plans for the given session IDs
            plans = await plans_collection.find(
                {
                    "id_sessao": {"$in": session_ids}
                },
                {
                    "_id": 0,
                    "id_sessao": 1,
                    "plano_execucao": 1
                }
            ).to_list(length=None)

            # Create a set of session IDs that have a non-empty plan
            sessions_with_plan_ids = {
                plan['id_sessao'] 
                for plan in plans 
                if plan.get('plano_execucao') and len(plan['plano_execucao']) > 0
            }

            # Find sessions that don't have a plan or have an empty plan
            sessions_without_plan = [
                {
                    "id_sessao": str(session['IdSessao']),
                    "Assunto": session['Assunto']
                }
                for session in study_sessions_list
                if str(session['IdSessao']) not in sessions_with_plan_ids
            ]

            print(f"Sessões sem plano para o estudante {student_email}: {sessions_without_plan}")
            return sessions_without_plan

        except Exception as e:
            print(f"Erro ao buscar sessões sem plano: {e}")
            return []

    async def create_automatic_study_plan(self, student_email: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates and saves an automatic study plan.
        """
        collection = self.db['study_plans']
        try:
            # Check if plan exists
            existing_plan = await collection.find_one({"id_sessao": session_data['session_id']})
            
            if existing_plan:
                # Update existing plan
                result = await collection.update_one(
                    {"id_sessao": session_data['session_id']},
                    {"$set": {
                        "plano_execucao": session_data.get('plano_execucao', []),
                        "duracao_total": session_data.get('duracao', "60 minutos"),
                        "updated_at": datetime.now(timezone.utc)
                    }}
                )
                return str(existing_plan['_id'])
            else:
                # Create new plan
                plan_data = {
                    "id_sessao": session_data['session_id'],
                    "duracao_total": session_data.get('duracao', "60 minutos"),
                    "plano_execucao": session_data.get('plano_execucao', []),
                    "progresso_total": 0,
                    "created_at": datetime.now(timezone.utc)
                }
                result = await collection.insert_one(plan_data)
                return str(result.inserted_id)

        except Exception as e:
            print(f"Error creating automatic study plan: {e}")
            raise


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

class MongoImageHandler:
    def __init__(self, mongo_manager):
        self.mongo_manager = mongo_manager
        self.image_collection = self.mongo_manager.db['image_collection']

    async def store_image(
        self,
        image_uuid: str,
        image_bytes: bytes,
        student_email: str,
        disciplina: str,
        session_id: str,
        filename: str,
        content_hash: str,
        access_level: str = "session"
    ) -> Dict[str, Any]:
        """
        Stores an image in MongoDB with verification.
        Returns the stored document data or raises an exception.
        """
        try:
            # Convert to BSON Binary
            binary_content = Binary(image_bytes)
            
            # Prepare document
            image_document = {
                "_id": image_uuid,
                "image_data": binary_content,
                "student_email": student_email,
                "disciplina": disciplina,
                "session_id": session_id,
                "filename": filename,
                "content_type": "image/jpeg",
                "file_size": len(image_bytes),
                "content_hash": content_hash,
                "access_level": access_level,
                "created_at": datetime.now(timezone.utc)
            }

            # Ensure collection exists
            collections = await self.mongo_manager.db.list_collection_names()
            if 'image_collection' not in collections:
                print("[DEBUG] 'image_collection' não existe, criando coleção e índices.")
                await self.mongo_manager.db.create_collection('image_collection')
                await self.image_collection.create_index("student_email")
                await self.image_collection.create_index("disciplina")
                await self.image_collection.create_index([("created_at", -1)])
                print("[MONGO] Created image_collection and indices")
            else:
                print("[DEBUG] 'image_collection' já existe.")

            # Insert document
            print(f"[DEBUG] Inserindo documento da imagem com _id={image_uuid}")
            result = await self.image_collection.insert_one(image_document)
            if not result.acknowledged:
                raise Exception("MongoDB insert not acknowledged")
            print(f"[DEBUG] Documento inserido com sucesso: {result.inserted_id}")

            # Verify insertion
            stored_doc = await self.image_collection.find_one({"_id": image_uuid})
            if not stored_doc:
                raise Exception("Image document not found after insertion")

            # Verify data integrity
            stored_size = len(stored_doc["image_data"])
            if stored_size != len(image_bytes):
                raise Exception(f"Size mismatch: stored={stored_size}, original={len(image_bytes)}")

            print(f"[MONGO] Successfully stored image {image_uuid}")
            print(f"[MONGO] File size: {stored_size} bytes")
            return stored_doc

        except Exception as e:
            print(f"[MONGO] Error storing image: {str(e)}")
            traceback.print_exc()
            raise


    async def verify_image_storage(self, image_uuid: str) -> bool:
        """
        Verifies if an image is properly stored in MongoDB.
        """
        try:
            stored_doc = await self.image_collection.find_one({"_id": image_uuid})
            return bool(stored_doc and stored_doc.get("image_data"))
        except Exception as e:
            print(f"[MONGO] Error verifying image {image_uuid}: {str(e)}")
            return False