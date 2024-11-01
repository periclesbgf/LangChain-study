# app/database/vector_db.py

import traceback
from typing import Optional, List
import uuid
from datetime import datetime, timezone
import hashlib
import io
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
import docx
import traceback
from typing import Optional, List
import uuid
from datetime import datetime, timezone
import hashlib
import io
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
import docx
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
    MatchAny
)
from utils import OPENAI_API_KEY, QDRANT_URL
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, VectorParams, Distance
)
from agent.image_handler import ImageHandler

from pydantic import BaseModel
from typing import Optional, List
from database.mongo_database_manager import MongoDatabaseManager, MongoImageHandler


class Material(BaseModel):
    """Data model for materials stored in Qdrant."""
    id: str  # content_hash or image_uuid
    name: str  # filename
    type: str  # 'pdf', 'doc', 'image'
    access_level: str
    discipline_id: Optional[str]
    session_id: Optional[str]
    student_email: str
    content_hash: str
    size: Optional[int]  # file size
    created_at: datetime
    updated_at: datetime


class QdrantHandler:
    def __init__(self, url: str, collection_name: str, embeddings, image_handler: ImageHandler, text_splitter, mongo_manager: MongoDatabaseManager):
        """Initialize QdrantHandler with necessary components."""
        print(f"Initializing QdrantHandler for collection '{collection_name}'...")
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(url=url, prefer_grpc=False)
        self.image_handler = image_handler
        self.text_splitter = text_splitter
        self.mongo_manager = mongo_manager
        self.ensure_collection_exists()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            validate_collection_config=True
        )
        print(f"QdrantHandler initialized successfully for collection '{collection_name}'")

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

    def add_document(
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
        """
        Adiciona um documento ao vector store com metadados apropriados.
        
        Args:
            student_email: Email do aluno
            session_id: ID da sessão
            content: Conteúdo textual (descrição para imagens)
            access_level: Nível de acesso do documento
            disciplina_id: ID da disciplina
            specific_file_id: ID específico do arquivo (opcional)
            metadata_extra: Metadados adicionais (opcional)
            file_content: Conteúdo binário do arquivo (opcional)
            filename: Nome do arquivo (opcional)
        """
        try:
            # Prepare metadata base
            metadata = {
                "student_email": student_email,
                "session_id": session_id,
                "access_level": access_level,
                "disciplina": disciplina_id,
                "file_id": specific_file_id,
                "filename": filename,
                "timestamp": datetime.now(timezone.utc).timestamp()
            }

            # Adiciona metadados extras se fornecidos
            if metadata_extra:
                metadata.update(metadata_extra)

            # Para documentos tipo imagem, verifica se há referência ao UUID
            if metadata.get("type") == "image" and "image_uuid" not in metadata:
                # Se não houver UUID, gera um novo
                image_uuid = str(uuid.uuid4())
                metadata["image_uuid"] = image_uuid

            # Converte todos os valores de metadados para string
            metadata = {k: str(v) if v is not None else "" for k, v in metadata.items()}

            # Cria e adiciona o documento
            doc = Document(page_content=content, metadata=metadata)
            self.vector_store.add_documents([doc])
            print(f"Document added successfully for student {student_email}")
            print(f"Document type: {metadata.get('type', 'not specified')}")
            
            if metadata.get("type") == "image":
                print(f"Image UUID: {metadata.get('image_uuid', 'not found')}")

        except Exception as e:
            print(f"Error adding document: {str(e)}")
            traceback.print_exc()
            raise

    def document_exists(self, content_hash: str, student_email: str, disciplina: str) -> bool:
        """Check if a document already exists in the collection."""
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
        """Process and store file content based on file type."""
        try:
            print(f"Processing file: {filename}")
            content_hash = hashlib.sha256(content).hexdigest()

            # Check for existing document
            if self.document_exists(content_hash, student_email, disciplina):
                print("Document already exists in Qdrant. Skipping insertion.")
                return

            # Process based on file type
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
        """Process PDF files: extract text, images, and tables with proper image storage in MongoDB."""
        try:
            # Initialize MongoImageHandler
            mongo_image_handler = MongoImageHandler(self.mongo_manager)
            
            # Process with PyMuPDF
            pdf_document = fitz.open(stream=content, filetype="pdf")
            full_text = ""

            for page_num in range(len(pdf_document)):
                # Extract text
                page = pdf_document[page_num]
                page_text = page.get_text()
                full_text += page_text

                # Process images
                image_list = page.get_images(full=True)
                for image_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_format = base_image["ext"]  # Get image format (e.g., 'jpeg', 'png')
                        
                        # Generate image UUID
                        image_uuid = str(uuid.uuid4())
                        
                        # Generate image description
                        img_base64 = self.image_handler.encode_image_bytes(image_bytes)
                        description = self.image_handler.image_summarize(img_base64)
                        
                        # Create image hash
                        image_hash = hashlib.sha256(image_bytes).hexdigest()
                        
                        try:
                            # Store image in MongoDB
                            stored_doc = await mongo_image_handler.store_image(
                                image_uuid=image_uuid,
                                image_bytes=image_bytes,
                                student_email=student_email,
                                disciplina=disciplina,
                                session_id=session_id,
                                filename=f"page_{page_num + 1}_image_{image_index + 1}.{image_format}",
                                content_hash=image_hash,
                                access_level=access_level,
                            )

                            # Verify storage
                            if not await mongo_image_handler.verify_image_storage(image_uuid):
                                raise Exception(f"Image storage verification failed for image {image_index + 1} on page {page_num + 1}")

                            # Store reference in Qdrant
                            image_metadata = {
                                "type": "image",
                                "page_number": str(page_num + 1),
                                "image_number": str(image_index + 1),
                                "content_hash": image_hash,
                                "parent_document": content_hash,
                                "filename": filename,
                                "file_id": specific_file_id,
                                "image_uuid": image_uuid,
                                "file_size": len(image_bytes),
                                "content_type": f"image/{image_format}",
                                "source": "pdf_extraction"
                            }

                            self.add_document(
                                student_email=student_email,
                                session_id=session_id,
                                content=description,
                                access_level=access_level,
                                disciplina_id=disciplina,
                                specific_file_id=specific_file_id,
                                metadata_extra=image_metadata
                            )

                            print(f"[PDF] Successfully processed and stored image {image_index + 1} from page {page_num + 1}")
                            print(f"[PDF] Image UUID: {image_uuid}")
                            print(f"[PDF] Image size: {len(image_bytes)} bytes")

                        except Exception as e:
                            print(f"[PDF] Error storing image in MongoDB: {str(e)}")
                            traceback.print_exc()
                            continue

                    except Exception as e:
                        print(f"[PDF] Error extracting image {image_index + 1} from page {page_num + 1}: {str(e)}")
                        continue

            pdf_document.close()

            # Process text content
            if full_text.strip():
                chunks = self.text_splitter.split_text(full_text)
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
                    
                    self.add_document(
                        student_email=student_email,
                        session_id=session_id,
                        content=chunk,
                        access_level=access_level,
                        disciplina_id=disciplina,
                        specific_file_id=specific_file_id,
                        metadata_extra=text_metadata
                    )

            # Process tables with pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        tables = page.extract_tables()
                        for table_index, table in enumerate(tables):
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_text = df.to_csv(index=False)
                                table_hash = hashlib.sha256(table_text.encode('utf-8')).hexdigest()
                                
                                table_metadata = {
                                    "type": "table",
                                    "page_number": str(page_num + 1),
                                    "table_number": str(table_index + 1),
                                    "content_hash": table_hash,
                                    "parent_document": content_hash,
                                    "filename": filename,
                                    "file_id": specific_file_id,
                                    "source": "pdf_extraction"
                                }
                                
                                self.add_document(
                                    student_email=student_email,
                                    session_id=session_id,
                                    content=table_text,
                                    access_level=access_level,
                                    disciplina_id=disciplina,
                                    specific_file_id=specific_file_id,
                                    metadata_extra=table_metadata
                                )
                    except Exception as e:
                        print(f"[PDF] Error processing tables on page {page_num + 1}: {str(e)}")
                        continue

        except Exception as e:
            print(f"[PDF] Error processing PDF file: {str(e)}")
            traceback.print_exc()
            raise

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
        """Process image files with verification."""
        try:
            # Generate image description
            img_base64 = self.image_handler.encode_image_bytes(content)
            description = self.image_handler.image_summarize(img_base64)
            
            # Generate UUID for image
            image_uuid = str(uuid.uuid4())

            # Initialize MongoImageHandler
            mongo_image_handler = MongoImageHandler(self.mongo_manager)
            
            # Store and verify image
            stored_doc = await mongo_image_handler.store_image(
                image_uuid=image_uuid,
                image_bytes=content,
                student_email=student_email,
                disciplina=disciplina,
                session_id=session_id,
                filename=filename,
                content_hash=content_hash,
                access_level=access_level
            )

            # Verify storage
            if not await mongo_image_handler.verify_image_storage(image_uuid):
                raise Exception("Image storage verification failed")

            # Prepare metadata for Qdrant
            metadata = {
                "type": "image",
                "content_hash": content_hash,
                "filename": filename,
                "file_id": specific_file_id,
                "image_uuid": image_uuid,
                "file_size": stored_doc["file_size"],
                "content_type": stored_doc["content_type"]
            }

            # Add document reference to Qdrant
            self.add_document(
                student_email=student_email,
                session_id=session_id,
                content=description,
                access_level=access_level,
                disciplina_id=disciplina,
                specific_file_id=specific_file_id,
                metadata_extra=metadata
            )

            print(f"[PROCESS] Image processed and stored successfully: {image_uuid}")
            return image_uuid

        except Exception as e:
            print(f"[ERROR] Failed to process image: {str(e)}")
            traceback.print_exc()
            raise

    async def get_materials(
            self,
            student_email: str,
            discipline_id: Optional[str] = None,
            session_id: Optional[str] = None
        ) -> List[Material]:
            """Retrieve materials based on access level and context."""
            try:
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

                results, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=100
                )
                
                materials = []
                seen_ids = set()
                
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
        use_global: bool = True,
        use_discipline: bool = True,
        use_session: bool = True,
        specific_file_id: Optional[str] = None,
        specific_metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Realiza busca por similaridade com filtros flexíveis usando a estrutura correta do Qdrant.
        """
        print(f"\n[SEARCH] Iniciando busca com filtros:")
        print(f"[SEARCH] Query: {query}")
        print(f"[SEARCH] Student: {student_email}")
        print(f"[SEARCH] Config: global={use_global}, discipline={use_discipline}, session={use_session}")

        try:
            # Busca por ID específico
            if specific_file_id:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.student_email",
                            match=models.MatchValue(value=student_email)
                        ),
                        models.FieldCondition(
                            key="metadata.file_id",
                            match=models.MatchValue(value=specific_file_id)
                        )
                    ]
                )
                return self._execute_search(query, search_filter, k)

            # Busca por metadados específicos
            if specific_metadata:
                must_conditions = [
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=str(value))
                    )
                    for key, value in specific_metadata.items()
                ]
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.student_email",
                        match=models.MatchValue(value=student_email)
                    )
                )
                search_filter = models.Filter(must=must_conditions)
                return self._execute_search(query, search_filter, k)

            # Busca por níveis de acesso
            should_conditions = []

            # Global access
            if use_global:
                should_conditions.append(
                    models.FieldCondition(
                        key="metadata.access_level",
                        match=models.MatchValue(value="global")
                    )
                )

            # Discipline access
            if use_discipline and disciplina_id:
                should_conditions.append(
                    models.FieldCondition(
                        key="metadata.access_level",
                        match=models.MatchValue(value="discipline")
                    )
                )

            # Session access
            if use_session and session_id:
                should_conditions.append(
                    models.FieldCondition(
                        key="metadata.access_level",
                        match=models.MatchValue(value="session")
                    )
                )

            # Construir filtro final
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.student_email",
                        match=models.MatchValue(value=student_email)
                    )
                ],
                should=should_conditions
            )

            print(f"[SEARCH] Filtro construído: {search_filter}")
            return self._execute_search(query, search_filter, k)

        except Exception as e:
            print(f"[ERROR] Erro durante a busca: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _execute_search(self, query: str, search_filter: models.Filter, k: int) -> List[Document]:
        """Execute the search with the constructed filter."""
        try:
            print(f"[SEARCH] Executing search with filter: {search_filter}")
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=search_filter
            )

            print(f"[SEARCH] Found {len(results)} results")
            for i, doc in enumerate(results, 1):
                print(f"[SEARCH] Result {i}:")
                print(f"  - Access level: {doc.metadata.get('access_level')}")
                print(f"  - Preview: {doc.page_content[:100]}...")

            return results

        except Exception as e:
            print(f"[ERROR] Error executing search: {str(e)}")
            traceback.print_exc()
            raise

    async def delete_material(self, material_id: str):
        """Delete a material from Qdrant."""
        try:
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
            print(f"[DELETE] Material deleted: {material_id}")
            
        except Exception as e:
            print(f"[ERROR] Error deleting material {material_id}: {str(e)}")
            raise

    async def update_material_access(
        self,
        material_id: str,
        new_access_level: str,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Update material access level and context."""
        try:
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
                    "updated_at": datetime.now().timestamp()
                }
            }
            
            self.client.update(
                collection_name=self.collection_name,
                points_filter=query_filter,
                payload=update_data
            )
            print(f"[UPDATE] Material updated: {material_id}")
            
        except Exception as e:
            print(f"[ERROR] Error updating material {material_id}: {str(e)}")
            raise

    def debug_metadata(self):
        """Debug metadata in the collection."""
        print("Debugging metadata...")
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10
            )
            print(f"{len(results)} documents found.")

            for result in results:
                metadata = result.payload.get("metadata", {})
                print(f"\nMetadata Retrieved: {metadata}")

                if not metadata:
                    print("⚠️ Empty or missing metadata")
                    continue

                # Check critical fields
                critical_fields = ["student_email", "disciplina", "access_level"]
                for field in critical_fields:
                    value = metadata.get(field)
                    if value is None:
                        print(f"⚠️ Missing critical field: {field}")
                    elif not isinstance(value, str):
                        print(f"⚠️ Field {field} is not a string: {type(value)}")

        except Exception as e:
            print(f"Error listing documents: {e}")
            traceback.print_exc()

    def fix_document_metadata(self):
        """Fix metadata issues in documents."""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=100  # Adjust based on your needs
            )
            
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
                        self.client.update_point(
                            collection_name=self.collection_name,
                            id=result.id,
                            payload={"metadata": metadata}
                        )
                        print(f"Metadata fixed for document {result.id}")

                except Exception as e:
                    print(f"Error processing document {result.id}: {e}")
                    continue

        except Exception as e:
            print(f"Error fixing metadata: {e}")
            traceback.print_exc()

class TextSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 50):
        """
        Initializes the TextSplitter with chunk size and overlap.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into smaller chunks.
        
        Args:
            text: Text to be split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller chunks.
        
        Args:
            documents: List of documents to be split
            
        Returns:
            List of split documents
        """
        return self.text_splitter.split_documents(documents)


class Embeddings:
    def __init__(self):
        """
        Inicializa a classe de embeddings.
        """
        self.embeddings = self.load_embeddings()

    def load_embeddings(self) -> OpenAIEmbeddings:
        """
        Carrega as embeddings do OpenAI.
        
        :return: Instância de OpenAIEmbeddings
        """
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def get_embeddings(self) -> OpenAIEmbeddings:
        """
        Retorna a instância de embeddings.
        
        :return: Instância de OpenAIEmbeddings
        """
        return self.embeddings