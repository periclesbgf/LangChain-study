# app/database/workspace_handler.py
from typing import Optional, List
import uuid
from datetime import datetime, timezone
from bson import ObjectId
from api.endpoints.models import Material, AccessLevel
from agent.image_handler import ImageHandler
from mongo_database_manager import MongoDatabaseManager
from vector_db import QdrantHandler
from pydantic import BaseModel
import io
from PIL import Image
import hashlib
from enum import Enum


class WorkspaceHandler:
    def __init__(self, mongo_manager: MongoDatabaseManager, qdrant_handler: QdrantHandler, image_handler: ImageHandler):
        self.mongo_manager = mongo_manager
        self.qdrant_handler = qdrant_handler
        self.image_handler = image_handler
        self.materials_collection = self.mongo_manager.client[self.mongo_manager.db_name]["materials"]

    async def add_material(
        self,
        file_content: bytes,
        filename: str,
        student_email: str,
        access_level: AccessLevel,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Material:
        """Adiciona um novo material com nível de acesso especificado."""
        
        # Gera ID único e hash do conteúdo
        material_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(file_content).hexdigest()
        
        # Cria documento do material
        material = Material(
            id=material_id,
            name=filename,
            type=self._get_file_type(filename),
            access_level=access_level,
            discipline_id=discipline_id,
            session_id=session_id,
            student_email=student_email,
            content_hash=content_hash,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Armazena conteúdo do arquivo e metadados no MongoDB
        await self._store_in_mongodb(material, file_content)

        # Processa e armazena no Qdrant baseado no tipo do arquivo
        await self._process_and_store_in_qdrant(material, file_content)

        return material

    def _get_file_type(self, filename: str) -> str:
        """Determina o tipo do arquivo pelo nome."""
        lower_filename = filename.lower()
        if lower_filename.endswith('.pdf'):
            return 'pdf'
        elif lower_filename.endswith(('.jpg', '.jpeg', '.png')):
            return 'image'
        elif lower_filename.endswith(('.doc', '.docx')):
            return 'doc'
        else:
            raise ValueError("Tipo de arquivo não suportado")

    async def _store_in_mongodb(self, material: Material, content: bytes):
        """Armazena metadados do material e conteúdo no MongoDB."""
        document = {
            "_id": material.id,
            "name": material.name,
            "type": material.type,
            "access_level": material.access_level,
            "discipline_id": material.discipline_id,
            "session_id": material.session_id,
            "student_email": material.student_email,
            "content_hash": material.content_hash,
            "content": content,
            "created_at": material.created_at,
            "updated_at": material.updated_at
        }
        await self.materials_collection.insert_one(document)

    async def _process_and_store_in_qdrant(self, material: Material, content: bytes):
        """Processa o conteúdo do arquivo e armazena no Qdrant baseado no tipo."""
        metadata_extra = {
            "file_id": material.id,
            "name": material.name,
            "access_level": material.access_level,
            "session_id": material.session_id
        }

        if material.type == 'image':
            # Processa imagem usando o ImageHandler existente
            img_base64 = self.image_handler.encode_image_bytes(content)
            description = self.image_handler.image_summarize(img_base64)
            
            self.qdrant_handler.add_document(
                student_email=material.student_email,
                disciplina=material.discipline_id,
                content=description,
                metadata_extra=metadata_extra
            )
        else:
            # Processa documentos baseados em texto (PDF, DOC)
            text_content = await self._extract_text_content(content, material.type)
            
            self.qdrant_handler.add_document(
                student_email=material.student_email,
                disciplina=material.discipline_id,
                content=text_content,
                metadata_extra=metadata_extra
            )

    async def get_materials(
        self,
        student_email: str,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Material]:
        """
        Recupera materiais baseado no nível de acesso e contexto.
        Retorna materiais que são:
        1. Globais (acessíveis a todas as disciplinas)
        2. Específicos da disciplina (se discipline_id for fornecido)
        3. Específicos da sessão (se session_id for fornecido)
        """
        query = {"student_email": student_email}
        
        if session_id:
            # Para contexto de sessão, obtém:
            # - Materiais globais
            # - Materiais da disciplina específica
            # - Materiais da sessão específica
            query["$or"] = [
                {"access_level": AccessLevel.GLOBAL},
                {
                    "access_level": AccessLevel.DISCIPLINE,
                    "discipline_id": discipline_id
                },
                {
                    "access_level": AccessLevel.SESSION,
                    "session_id": session_id
                }
            ]
        elif discipline_id:
            # Para contexto de disciplina, obtém:
            # - Materiais globais
            # - Materiais da disciplina específica
            query["$or"] = [
                {"access_level": AccessLevel.GLOBAL},
                {
                    "access_level": AccessLevel.DISCIPLINE,
                    "discipline_id": discipline_id
                }
            ]
        else:
            # Apenas materiais globais
            query["access_level"] = AccessLevel.GLOBAL

        cursor = self.materials_collection.find(query)
        materials = []
        async for doc in cursor:
            materials.append(Material(**doc))
        return materials

    async def get_material_content(self, material_id: str) -> Optional[bytes]:
        """Recupera o conteúdo do material do MongoDB."""
        doc = await self.materials_collection.find_one({"_id": material_id})
        if doc:
            return doc.get("content")
        return None

    async def update_material_access(
        self,
        material_id: str,
        new_access_level: AccessLevel,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Atualiza o nível de acesso do material e contexto."""
        # Atualiza no MongoDB
        update_data = {
            "access_level": new_access_level,
            "discipline_id": discipline_id,
            "session_id": session_id,
            "updated_at": datetime.now(timezone.utc)
        }
        
        await self.materials_collection.update_one(
            {"_id": material_id},
            {"$set": update_data}
        )

        # Atualiza no Qdrant
        metadata_update = {
            "access_level": new_access_level,
            "discipline_id": discipline_id,
            "session_id": session_id
        }
        
        # Recupera o documento atual do MongoDB para obter todos os dados necessários
        doc = await self.materials_collection.find_one({"_id": material_id})
        if doc:
            self.qdrant_handler.add_document(
                student_email=doc["student_email"],
                disciplina=discipline_id or doc["discipline_id"],
                content=doc["content"],
                metadata_extra=metadata_update
            )

    async def delete_material(self, material_id: str):
        """Remove o material do MongoDB e suas referências do Qdrant."""
        # Primeiro, obtém os dados do material para poder remover do Qdrant
        doc = await self.materials_collection.find_one({"_id": material_id})
        if doc:
            # Remove do MongoDB
            await self.materials_collection.delete_one({"_id": material_id})
            
            # Remove do Qdrant usando o content_hash como identificador
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.file_id",
                        match=MatchValue(value=material_id)
                    )
                ]
            )
            self.qdrant_handler.client.delete(
                collection_name=self.qdrant_handler.collection_name,
                points_selector=query_filter
            )

    async def _extract_text_content(self, content: bytes, file_type: str) -> str:
        """
        Extrai texto do conteúdo do arquivo baseado no tipo.
        Você precisará implementar a lógica específica para cada tipo de arquivo.
        """
        # TODO: Implementar extração de texto para diferentes tipos de arquivo
        return "Texto extraído do documento"  # Placeholder