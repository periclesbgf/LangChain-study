# app/database/vector_db.py

import traceback
from typing import Optional, List
import uuid
from datetime import datetime, timezone
import hashlib
import io
import fitz
from PIL import Image
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
    Distance,
)
from utils import OPENAI_API_KEY
from langchain_qdrant import QdrantVectorStore
from agent.image_handler import ImageHandler
from pydantic import BaseModel
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from database.mongo_database_manager import MongoDatabaseManager, MongoImageHandler
from logg import logger


class Material(BaseModel):
    """Data model for materials stored in Qdrant."""
    id: str
    name: str
    type: str
    access_level: str
    discipline_id: Optional[str]
    session_id: Optional[str]
    student_email: str
    content_hash: str
    size: Optional[int]
    created_at: datetime
    updated_at: datetime


class OptimizedImageProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.image_cache = {}
        
    @lru_cache(maxsize=1000)
    def _calculate_image_hash(self, image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()
    
    async def optimize_image(self, image_bytes: bytes, max_size: int = 1024) -> bytes:
        """Optimize image size and quality"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._optimize_image_sync,
            image_bytes,
            max_size
        )
    
    def _optimize_image_sync(self, image_bytes: bytes, max_size: int) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        return buffer.getvalue()


class QdrantHandler:
    def __init__(self, url: str, collection_name: str, embeddings, image_handler: ImageHandler, text_splitter, mongo_manager: MongoDatabaseManager):
        """Initialize QdrantHandler with necessary components."""
        print(f"Initializing QdrantHandler for collection '{collection_name}'...")
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(url=url, prefer_grpc=True)  # Using gRPC for better performance
        self.image_handler = image_handler
        self.text_splitter = text_splitter
        self.mongo_manager = mongo_manager
        self.image_processor = OptimizedImageProcessor()
        self.batch_size = 100
        self._setup_client()
        self.ensure_collection_exists()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            validate_collection_config=True
        )
        print(f"QdrantHandler initialized successfully for collection '{collection_name}'")

    def _setup_client(self):
        """Setup optimized client configuration"""
        self.client._client.options = [
            ('grpc.max_send_message_length', 512 * 1024 * 1024),
            ('grpc.max_receive_message_length', 512 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
        ]

    def ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant, create if it doesn't."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            print(f"Available collections: {collection_names}")

            if self.collection_name not in collection_names:
                vector_size = len(self.embeddings.embed_query("test query"))
                print(f"Creating collection '{self.collection_name}' with vector size {vector_size}")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"Collection '{self.collection_name}' created successfully")
            else:
                print(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            print(f"Error ensuring collection exists: {str(e)}")
            traceback.print_exc()
            raise

    @lru_cache(maxsize=1000)
    def _get_document_embedding(self, content: str):
        """Cache document embeddings"""
        return self.embeddings.embed_query(content)

    async def add_documents_batch(self, documents: List[Document]):
        """Add multiple documents in batch"""
        try:
            batched_docs = [documents[i:i + self.batch_size] 
                           for i in range(0, len(documents), self.batch_size)]
            
            for batch in batched_docs:
                points = []
                for doc in batch:
                    vector = self._get_document_embedding(doc.page_content)
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"metadata": doc.metadata, "content": doc.page_content}
                    )
                    points.append(point)
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            print(f"Successfully added {len(documents)} documents in batches")
        except Exception as e:
            print(f"Error adding documents batch: {str(e)}")
            traceback.print_exc()
            raise

    async def add_document(
        self,
        student_email: str,
        session_id: str,
        content: str,
        access_level: str,
        disciplina_id: str,
        specific_file_id: Optional[str] = None,
        metadata_extra: Optional[dict] = None,
        file_content: Optional[bytes] = None,
        filename: Optional[str] = None
    ):
        """Add a document with optimized metadata handling"""
        try:
            metadata = {
                "student_email": student_email,
                "session_id": session_id,
                "access_level": access_level,
                "disciplina": disciplina_id,
                "file_id": specific_file_id,
                "filename": filename,
                "timestamp": str(datetime.now(timezone.utc).timestamp())
            }

            if metadata_extra:
                metadata.update(metadata_extra)

            if metadata.get("type") == "image" and "image_uuid" not in metadata:
                metadata["image_uuid"] = str(uuid.uuid4())

            metadata = {k: str(v) if v is not None else "" for k, v in metadata.items()}
            
            doc = Document(page_content=content, metadata=metadata)
            await self.add_documents_batch([doc])
            
            print(f"Document added successfully for student {student_email}")
            if metadata.get("type") == "image":
                print(f"Image UUID: {metadata.get('image_uuid', 'not found')}")

        except Exception as e:
            print(f"Error adding document: {str(e)}")
            traceback.print_exc()
            raise

    async def process_file(
        self,
        content: bytes,
        filename: str,
        student_email: str,
        session_id: str,
        disciplina: str,
        access_level: str = "session",
        specific_file_id: Optional[str] = None
    ):
        """Process and store file content with optimizations"""
        try:
            print(f"Processing file: {filename}")
            content_hash = hashlib.sha256(content).hexdigest()

            if self.document_exists(content_hash, student_email, disciplina):
                print("Document already exists in Qdrant. Skipping insertion.")
                return

            if filename.lower().endswith('.pdf'):
                await self._process_pdf_file(
                    content=content,
                    student_email=student_email,
                    session_id=session_id,
                    disciplina=disciplina,
                    access_level=access_level,
                    content_hash=content_hash,
                    filename=filename,
                    specific_file_id=specific_file_id
                )
            elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                await self._process_image_file(
                    content=content,
                    student_email=student_email,
                    session_id=session_id,
                    disciplina=disciplina,
                    access_level=access_level,
                    content_hash=content_hash,
                    filename=filename,
                    specific_file_id=specific_file_id
                )
            else:
                raise ValueError(f"Unsupported file type: {filename}")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            traceback.print_exc()
            raise

    async def _process_pdf_file(
        self,
        content: bytes,
        student_email: str,
        session_id: str,
        disciplina: str,
        access_level: str,
        content_hash: str,
        filename: str,
        specific_file_id: Optional[str] = None
    ):
        """Process PDF files with optimizations"""
        try:
            mongo_image_handler = MongoImageHandler(self.mongo_manager)
            pdf_document = fitz.open(stream=content, filetype="pdf")
            
            # Process pages concurrently
            async with asyncio.TaskGroup() as group:
                tasks = []
                for page_num in range(len(pdf_document)):
                    task = group.create_task(
                        self._process_pdf_page(
                            pdf_document=pdf_document,
                            page_num=page_num,
                            mongo_image_handler=mongo_image_handler,
                            student_email=student_email,
                            session_id=session_id,
                            disciplina=disciplina,
                            access_level=access_level,
                            content_hash=content_hash,
                            filename=filename,
                            specific_file_id=specific_file_id
                        )
                    )
                    tasks.append(task)
                
            results = [task.result() for task in tasks]
            
            # Process all extracted text
            full_text = "".join([result.get("text", "") for result in results])
            if full_text.strip():
                chunks = self.text_splitter.split_text(full_text)
                text_documents = []
                for chunk_index, chunk in enumerate(chunks):
                    text_metadata = {
                        "type": "text",
                        "chunk_index": str(chunk_index + 1),
                        "total_chunks": str(len(chunks)),
                        "content_hash": content_hash,
                        "filename": filename,
                        "file_id": specific_file_id,
                        "source": "pdf_extraction"
                    }
                    
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "student_email": student_email,
                            "session_id": session_id,
                            "access_level": access_level,
                            "disciplina": disciplina,
                            **text_metadata
                        }
                    )
                    text_documents.append(doc)
                
                await self.add_documents_batch(text_documents)

            pdf_document.close()

        except Exception as e:
            print(f"Error processing PDF file: {str(e)}")
            traceback.print_exc()
            raise

    async def _process_pdf_page(self, pdf_document, page_num, mongo_image_handler, **kwargs):
        """Process a single PDF page"""
        page = pdf_document[page_num]
        page_text = page.get_text()
        
        # Process images in page
        image_list = page.get_images(full=True)
        image_results = []
        
        for image_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Optimize image
                optimized_bytes = await self.image_processor.optimize_image(image_bytes)
                
                image_result = await self._process_single_image(
                    optimized_bytes,
                    mongo_image_handler,
                    page_num,
                    image_index,
                    **kwargs
                )
                image_results.append(image_result)
                
            except Exception as e:
                print(f"Error processing image {image_index} on page {page_num}: {str(e)}")
                continue
        
        return {
            "text": page_text,
            "images": image_results
        }

    async def _process_single_image(
        self,
        image_bytes: bytes,
        mongo_image_handler: MongoImageHandler,
        page_num: int,
        image_index: int,
        **kwargs
    ):
        """Process a single image with optimizations"""
        image_uuid = str(uuid.uuid4())
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Generate image description concurrently with storage
        img_base64 = self.image_handler.encode_image_bytes(image_bytes)
        description_task = asyncio.create_task(
            asyncio.to_thread(self.image_handler.image_summarize, img_base64)
        )
        
        # Store image
        storage_task = asyncio.create_task(
            mongo_image_handler.store_image(
                image_uuid=image_uuid,
                image_bytes=image_bytes,
                student_email=kwargs["student_email"],
                disciplina=kwargs["disciplina"],
                session_id=kwargs["session_id"],
                filename=f"page_{page_num + 1}_image_{image_index + 1}.jpg",
                content_hash=image_hash,
                access_level=kwargs["access_level"],
            )
        )
        
        # Wait for both tasks to complete
        description, stored_doc = await asyncio.gather(description_task, storage_task)

# Verify storage
        if not await mongo_image_handler.verify_image_storage(image_uuid):
            raise Exception(f"Image storage verification failed for image {image_index + 1} on page {page_num + 1}")

        # Add to Qdrant with optimized metadata
        image_metadata = {
            "type": "image",
            "page_number": str(page_num + 1),
            "image_number": str(image_index + 1),
            "content_hash": image_hash,
            "parent_document": kwargs.get("content_hash", ""),
            "filename": kwargs.get("filename", ""),
            "file_id": kwargs.get("specific_file_id", ""),
            "image_uuid": image_uuid,
            "file_size": len(image_bytes),
            "content_type": "image/jpeg",
            "source": "pdf_extraction"
        }

        await self.add_document(
            student_email=kwargs["student_email"],
            session_id=kwargs["session_id"],
            content=description,
            access_level=kwargs["access_level"],
            disciplina_id=kwargs["disciplina"],
            specific_file_id=kwargs.get("specific_file_id"),
            metadata_extra=image_metadata
        )

        return {
            "image_uuid": image_uuid,
            "description": description,
            "metadata": image_metadata
        }

    async def _process_image_file(
        self,
        content: bytes,
        student_email: str,
        session_id: str,
        disciplina: str,
        access_level: str,
        content_hash: str,
        filename: str,
        specific_file_id: Optional[str] = None
    ):
        """Process image files with optimizations and parallel processing"""
        try:
            # Optimize image
            optimized_content = await self.image_processor.optimize_image(content)
            
            # Generate image description concurrently with storage
            img_base64 = self.image_handler.encode_image_bytes(optimized_content)
            image_uuid = str(uuid.uuid4())
            
            # Create concurrent tasks
            description_task = asyncio.create_task(
                asyncio.to_thread(self.image_handler.image_summarize, img_base64)
            )
            
            mongo_image_handler = MongoImageHandler(self.mongo_manager)
            storage_task = asyncio.create_task(
                mongo_image_handler.store_image(
                    image_uuid=image_uuid,
                    image_bytes=optimized_content,
                    student_email=student_email,
                    disciplina=disciplina,
                    session_id=session_id,
                    filename=filename,
                    content_hash=content_hash,
                    access_level=access_level
                )
            )
            
            # Wait for both tasks to complete
            description, stored_doc = await asyncio.gather(description_task, storage_task)
            
            # Verify storage
            if not await mongo_image_handler.verify_image_storage(image_uuid):
                raise Exception("Image storage verification failed")

            # Add to Qdrant with optimized metadata
            metadata = {
                "type": "image",
                "content_hash": content_hash,
                "filename": filename,
                "file_id": specific_file_id,
                "image_uuid": image_uuid,
                "file_size": stored_doc["file_size"],
                "content_type": stored_doc["content_type"]
            }

            await self.add_document(
                student_email=student_email,
                session_id=session_id,
                content=description,
                access_level=access_level,
                disciplina_id=disciplina,
                specific_file_id=specific_file_id,
                metadata_extra=metadata
            )

            return image_uuid

        except Exception as e:
            print(f"[ERROR] Failed to process image: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def document_exists(self, content_hash: str, student_email: str, disciplina: str) -> bool:
        """Check if a document already exists in the collection with caching."""
        try:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.content_hash",
                        match=MatchValue(value=str(content_hash))
                    ),
                    FieldCondition(
                        key="metadata.student_email",
                        match=MatchValue(value=str(student_email))
                    ),
                    FieldCondition(
                        key="metadata.disciplina",
                        match=MatchValue(value=str(disciplina))
                    )
                ]
            )
            
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1
            )
            exists = len(results) > 0
            print(f"Document exists check: {exists}")
            return exists
            
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")
            traceback.print_exc()
            return False

    async def get_materials(
            self,
            student_email: str,
            discipline_id: Optional[str] = None,
            session_id: Optional[str] = None
        ) -> List[Material]:
        """Retrieve materials with optimized filtering and caching."""
        try:
            cache_key = f"{student_email}:{discipline_id}:{session_id}"
            
            # Try to get from cache first
            if hasattr(self, '_materials_cache') and cache_key in self._materials_cache:
                return self._materials_cache[cache_key]
            
            filter_conditions = [{"access_level": "global"}]
            
            if discipline_id:
                filter_conditions.append({
                    "access_level": "discipline",
                    "discipline_id": discipline_id
                })
                
            if session_id:
                filter_conditions.append({
                    "access_level": "session",
                    "session_id": session_id
                })

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.student_email",
                        match=models.MatchValue(value=student_email)
                    )
                ],
                should=[
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=str(value))
                            ) for key, value in condition.items()
                        ]
                    ) for condition in filter_conditions
                ]
            )

            # Use batch processing for better performance
            batch_size = 100
            materials = []
            seen_ids = set()
            offset = 0
            
            while True:
                results, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=batch_size,
                    offset=offset
                )
                
                if not results:
                    break
                    
                for result in results:
                    metadata = result.payload.get("metadata", {})
                    if metadata.get("id") not in seen_ids:
                        try:
                            materials.append(Material(
                                id=metadata["id"],
                                name=metadata["name"],
                                type=metadata["type"],
                                access_level=metadata["access_level"],
                                discipline_id=metadata.get("discipline_id"),
                                session_id=metadata.get("session_id"),
                                student_email=metadata["student_email"],
                                content_hash=metadata["content_hash"],
                                size=metadata.get("size"),
                                created_at=datetime.fromtimestamp(metadata["created_at"]),
                                updated_at=datetime.fromtimestamp(metadata["updated_at"])
                            ))
                            seen_ids.add(metadata["id"])
                        except KeyError as ke:
                            print(f"[WARNING] Missing required field in metadata: {ke}")
                            print(f"Metadata content: {metadata}")
                            continue

                if not next_page_offset:
                    break
                offset = next_page_offset

            # Cache the results
            if not hasattr(self, '_materials_cache'):
                self._materials_cache = {}
            self._materials_cache[cache_key] = materials

            return materials

        except Exception as e:
            print(f"[ERROR] Error retrieving materials: {str(e)}")
            traceback.print_exc()
            raise

    def similarity_search_with_filter(
        self,
        query: str,
        student_email: str,
        session_id: Optional[str] = None,
        disciplina_id: Optional[str] = None,
        k: int = 5,
        use_global: bool = False,
        use_discipline: bool = True,
        use_session: bool = True,
        specific_file_id: Optional[str] = None,
        specific_metadata: Optional[dict] = None
    ) -> List[Document]:
        """Executes a similarity search with filtering."""
        try:
            access_conditions = []
            
            if use_global:
                access_conditions.append(
                    models.FieldCondition(
                        key="metadata.access_level",
                        match=models.MatchValue(value="global")
                    )
                )
                
            if use_discipline and disciplina_id:
                access_conditions.append(
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.access_level",
                                match=models.MatchValue(value="discipline")
                            ),
                            models.FieldCondition(
                                key="metadata.discipline_id",
                                match=models.MatchValue(value=disciplina_id)
                            )
                        ]
                    )
                )

            if use_session and session_id:
                access_conditions.append(
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.access_level",
                                match=models.MatchValue(value="session")
                            ),
                            models.FieldCondition(
                                key="metadata.session_id",
                                match=models.MatchValue(value=session_id)
                            )
                        ]
                    )
                )

            # Depois, construir as outras condições de filtro
            other_conditions = [
                models.FieldCondition(
                    key="metadata.student_email",
                    match=models.MatchValue(value=student_email)
                )
            ]

            # Adicionar metadados específicos
            if specific_metadata:
                for key, value in specific_metadata.items():
                    other_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=str(value))
                        )
                    )

            # Adicionar busca por arquivo específico
            if specific_file_id:
                other_conditions.append(
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=specific_file_id)
                    )
                )

            search_filter = models.Filter(
                must=other_conditions,
                should=access_conditions if access_conditions else None
            )

            return self._execute_search(query, search_filter, k)

        except Exception as e:
            print(f"[ERROR] Error during search: {str(e)}")
            return []

    def _cache_search_results(self, cache_key: str, results: List[Document]):
        """Cache search results with LRU cache management."""
        if not hasattr(self, '_search_cache'):
            self._search_cache = {}
        if len(self._search_cache) > 1000:  # Limit cache size
            self._search_cache.pop(next(iter(self._search_cache)))
        self._search_cache[cache_key] = results

    def _execute_search(self, query: str, search_filter: models.Filter, k: int) -> List[Document]:
        """Executes the search with the given filter and retrieves complete payloads."""
        try:

            # Executa a busca principal
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=search_filter
            )

            # Recupera os payloads completos
            scroll_results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=k,
                with_payload=True
            )

            # Atualiza o conteúdo dos documentos com os payloads completos
            for doc in results:
                for scroll_result in scroll_results:
                    if scroll_result.payload["metadata"].get("image_uuid") == doc.metadata.get("image_uuid"):
                        doc.page_content = scroll_result.payload.get("content", "")
                        break

            return results

        except Exception as e:
            print(f"[ERROR] Error executing search: {str(e)}")
            raise
    async def delete_material(self, material_id: str):
        """Delete a material with optimized cleanup."""
        try:
            # Clear relevant caches
            if hasattr(self, '_materials_cache'):
                self._materials_cache.clear()
            if hasattr(self, '_search_cache'):
                self._search_cache.clear()

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.id",
                        match=models.MatchValue(value=material_id)
                    )
                ]
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_filter=query_filter
            )
            logger.info(f"[DELETE] Material deleted: {material_id}")

        except Exception as e:
            logger.error(f"[ERROR] Error deleting material {material_id}: {str(e)}")
            raise

    async def update_material_access(
        self,
        material_id: str,
        new_access_level: str,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Update material access with cache management."""
        try:
            # Clear relevant caches
            if hasattr(self, '_materials_cache'):
                self._materials_cache.clear()
            if hasattr(self, '_search_cache'):
                self._search_cache.clear()
                
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.id",
                        match=models.MatchValue(value=material_id)
                    )
                ]
            )
            update_data = {
                            "metadata": {
                                "access_level": new_access_level,
                                "discipline_id": discipline_id,
                                "session_id": session_id,
                                "updated_at": str(datetime.now().timestamp())
                            }
                        }
            
            self.client.update(
                collection_name=self.collection_name,
                points_filter=query_filter,
                payload=update_data
            )
            logger.info(f"[UPDATE] Material updated: {material_id}")
            
        except Exception as e:
            logger.error(f"[VECTOR_DB] Error updating material {material_id}: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def debug_metadata(self):
        """Debug metadata with caching for repeated calls."""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10
            )

            metadata_issues = []
            for result in results:
                metadata = result.payload.get("metadata", {})

                if not metadata:
                    metadata_issues.append(f"Empty metadata for document {result.id}")
                    continue

                # Check critical fields
                critical_fields = ["student_email", "disciplina", "access_level"]
                for field in critical_fields:
                    value = metadata.get(field)
                    if value is None:
                        metadata_issues.append(f"Missing critical field: {field} in document {result.id}")
                    elif not isinstance(value, str):
                        metadata_issues.append(f"Field {field} is not a string in document {result.id}")

            return metadata_issues

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return [f"Error during debug: {str(e)}"]

    async def fix_document_metadata(self):
        """Fix metadata issues with batch processing."""
        try:
            batch_size = 100
            offset = 0
            total_fixed = 0
            
            while True:
                results, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset
                )
                
                if not results:
                    break

                update_operations = []
                for result in results:
                    try:
                        metadata = result.payload.get("metadata", {})
                        needs_update = False
                        
                        # Ensure all values are strings
                        for key, value in metadata.items():
                            if value is None:
                                metadata[key] = ""
                                needs_update = True
                            elif not isinstance(value, str):
                                metadata[key] = str(value)
                                needs_update = True

                        # Add missing required fields
                        required_fields = {
                            "access_level": "session",
                            "disciplina": "1",
                            "student_email": "",
                            "session_id": "",
                            "created_at": str(datetime.now().timestamp()),
                            "updated_at": str(datetime.now().timestamp())
                        }

                        for field, default_value in required_fields.items():
                            if field not in metadata:
                                metadata[field] = default_value
                                needs_update = True

                        if needs_update:
                            update_operations.append({
                                "id": result.id,
                                "metadata": metadata
                            })

                    except Exception as e:
                        logger.error(f"Error processing document {result.id}: {e}")
                        continue

                # Batch update documents
                if update_operations:
                    try:
                        async with asyncio.TaskGroup() as group:
                            for operation in update_operations:
                                group.create_task(
                                    self._update_document_metadata(operation)
                                )
                        total_fixed += len(update_operations)
                    except Exception as e:
                        logger.error(f"Error during batch update: {e}")

                if not next_page_offset:
                    break
                offset = next_page_offset

            if hasattr(self, '_materials_cache'):
                self._materials_cache.clear()
            if hasattr(self, '_search_cache'):
                self._search_cache.clear()

        except Exception as e:
            logger.error(f"Error fixing metadata: {e}")
            traceback.print_exc()

    async def _update_document_metadata(self, operation: dict):
        """Helper method for updating document metadata."""
        try:
            self.client.update_point(
                collection_name=self.collection_name,
                id=operation["id"],
                payload={"metadata": operation["metadata"]}
            )
        except Exception as e:
            logger.error(f"Error updating document {operation['id']}: {e}")
            raise


class TextSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._cache = {}

    @lru_cache(maxsize=1000)
    def split_text(self, text: str) -> List[str]:
        """Split text with caching for repeated calls."""
        return self.text_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with batch processing."""
        # Process documents in batches for better memory management
        batch_size = 10
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_results = self.text_splitter.split_documents(batch)
            results.extend(batch_results)
            
        return results


class Embeddings:
    def __init__(self, cache_size: int = 1000):
        """Initialize with caching capabilities."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self._cache = {}
        self.cache_size = cache_size
        self.executor = ThreadPoolExecutor(max_workers=4)

    def get_embeddings(self):
        """Return the embeddings instance."""
        return self.embeddings

    @lru_cache(maxsize=1000)
    def embed_query(self, text: str):
        """Embed query with caching."""
        return self.embeddings.embed_query(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with parallel processing."""
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                self.executor,
                self.embeddings.embed_documents,
                batch
            )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def clear_cache(self):
        """Clear embedding caches."""
        self._cache.clear()
        self.embed_query.cache_clear()