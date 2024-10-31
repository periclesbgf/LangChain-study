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
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
import uuid
from qdrant_client.http import models

from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
)
from utils import OPENAI_API_KEY, QDRANT_URL
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, VectorParams, Distance
)
from pydantic import BaseModel
from langchain_core.documents import Document
from typing import List, Optional
import hashlib


class Material(BaseModel):
    id: str  # content_hash ou image_uuid
    name: str  # nome do arquivo
    type: str  # 'pdf', 'doc', 'image'
    access_level: str
    discipline_id: Optional[str]
    session_id: Optional[str]
    student_email: str
    content_hash: str
    size: Optional[int]  # tamanho do arquivo
    created_at: datetime
    updated_at: datetime

class QdrantHandler:
    def __init__(self, url: str, collection_name: str, embeddings, image_handler, text_splitter):
        print(f"Inicializando QdrantHandler para a coleção '{collection_name}'...")
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(url=url, prefer_grpc=False)
        self.image_handler = image_handler
        self.text_splitter = text_splitter
        print(f"QdrantClient inicializado para a URL '{url}'.")

        self.ensure_collection_exists()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            validate_collection_config=True
        )
        print(f"QdrantHandler inicializado com sucesso para a coleção '{collection_name}'.")

    def ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        print(f"Coleções disponíveis: {[col.name for col in collections]}")

        if self.collection_name not in [col.name for col in collections]:
            vector_size = len(self.embeddings.embed_query("test query"))
            print(f"Recriando coleção '{self.collection_name}' com tamanho de vetor {vector_size}.")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Coleção '{self.collection_name}' criada com sucesso.")
        else:
            print(f"Coleção '{self.collection_name}' já existe.")

    async def add_material(
        self,
        file_content: bytes,
        filename: str,
        student_email: str,
        access_level: str,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Material:
        """Adiciona um novo material com nível de acesso especificado."""
        
        # Gera IDs únicos e hash
        content_hash = hashlib.sha256(file_content).hexdigest()
        file_type = self._get_file_type(filename)
        image_uuid = str(uuid.uuid4()) if file_type == 'image' else None

        # Processa o conteúdo baseado no tipo
        if file_type == 'image':
            # Processa imagem
            img_base64 = self.image_handler.encode_image_bytes(file_content)
            description = self.image_handler.image_summarize(img_base64)
            
            # Metadata para imagem
            metadata = {
                "id": image_uuid,
                "name": filename,
                "student_email": student_email,
                "session_id": session_id,
                "content_hash": content_hash,
                "access_level": access_level.lower(),
                "type": file_type,
                "size": len(file_content),
                "created_at": datetime.now().timestamp(),
                "updated_at": datetime.now().timestamp(),
                "disciplina": discipline_id
            }
            
            # Adiciona ao Qdrant
            self.add_document(
                student_email=student_email,
                session_id=session_id,
                content=description,
                metadata_extra=metadata
            )
        else:
            # Processa documento texto
            text_content = await self._extract_text_content(file_content, file_type, content_hash)
            
            # Metadata para documento
            metadata = {
                "id": content_hash,
                "name": filename,
                "student_email": student_email,
                "session_id": session_id,
                "content_hash": content_hash,
                "access_level": access_level.lower(),
                "type": file_type,
                "size": len(file_content),
                "created_at": datetime.now().timestamp(),
                "updated_at": datetime.now().timestamp(),
                "disciplina": discipline_id
            }
            
            # Adiciona ao Qdrant
            if self.text_splitter and len(text_content) > 1000:
                chunks = self.text_splitter.split_text(text_content)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": i + 1,
                        "total_chunks": len(chunks)
                    }
                    self.add_document(
                        student_email=student_email,
                        session_id=session_id,
                        content=chunk,
                        metadata_extra=chunk_metadata
                    )
            else:
                self.add_document(
                    student_email=student_email,
                    session_id=session_id,
                    content=text_content,
                    metadata_extra=metadata
                )

        # Retorna Material
        return Material(
            id=image_uuid or content_hash,
            name=filename,
            type=file_type,
            access_level=access_level,
            discipline_id=discipline_id,
            session_id=session_id,
            student_email=student_email,
            content_hash=content_hash,
            size=len(file_content),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

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

    async def _extract_text_content(self, content: bytes, file_type: str, material_id: str) -> str:
        """Extrai texto do conteúdo do arquivo baseado no tipo."""
        try:
            extracted_content = ""
            
            if file_type == 'pdf':
                # Processa PDF
                pdf_document = fitz.open(stream=content, filetype="pdf")
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    extracted_content += page.get_text() + "\n"
                    
                    # Processa imagens da página
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        img_base64 = self.image_handler.encode_image_bytes(image_bytes)
                        description = self.image_handler.image_summarize(img_base64)
                        extracted_content += f"\nImagem {img_index + 1} da página {page_num + 1}: {description}\n"
                
                pdf_document.close()
                
                # Processa tabelas
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for table_index, table in enumerate(tables):
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_text = df.to_csv(index=False)
                                extracted_content += f"\nTabela {table_index + 1} da página {page_num + 1}:\n{table_text}\n"
                
            elif file_type == 'doc':
                # Processa DOC/DOCX
                doc = docx.Document(io.BytesIO(content))
                
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        extracted_content += paragraph.text + "\n"
                
                for table_index, table in enumerate(doc.tables):
                    table_data = []
                    for row in table.rows:
                        table_data.append([cell.text.strip() for cell in row.cells])
                    
                    if table_data:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        table_text = df.to_csv(index=False)
                        extracted_content += f"\nTabela {table_index + 1}:\n{table_text}\n"
                
            elif file_type == 'image':
                # Processa imagem
                image = Image.open(io.BytesIO(content))
                
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                
                ocr_text = pytesseract.image_to_string(image, lang='por+eng')
                if ocr_text.strip():
                    extracted_content += f"Texto detectado na imagem:\n{ocr_text}\n"
                
                img_base64 = self.image_handler.encode_image_bytes(content)
                description = self.image_handler.image_summarize(img_base64)
                extracted_content += f"\nDescrição da imagem:\n{description}\n"
            
            return self._clean_text(extracted_content)
            
        except Exception as e:
            print(f"[ERROR] Erro na extração de texto: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto extraído."""
        if not text:
            return ""
        
        # Remove caracteres especiais e espaços extras
        cleaned = " ".join(text.split())
        
        # Remove linhas vazias extras
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
        
        return cleaned

    async def get_materials(
        self,
        student_email: str,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Material]:
        """Recupera materiais do Qdrant baseado no nível de acesso e contexto."""
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
                                match=models.MatchValue(value=value)
                            ) for key, value in condition.items()
                        ]
                    ) for condition in filter_conditions
                ]
            )

            # Use scroll instead of search since we're not doing similarity search
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=100  # Adjust this value based on your needs
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
            print(f"[ERROR] Erro ao recuperar materiais: {str(e)}")
            traceback.print_exc()
            raise

    async def delete_material(self, material_id: str):
        """Remove o material do Qdrant."""
        try:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=material_id)
                    )
                ]
            )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_filter=query_filter
            )
            print(f"[DELETE] Material removido: {material_id}")
            
        except Exception as e:
            print(f"[ERROR] Erro ao remover material {material_id}: {str(e)}")
            raise

    async def update_material_access(
        self,
        material_id: str,
        new_access_level: str,
        discipline_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Atualiza o nível de acesso do material e contexto."""
        try:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=material_id)
                    )
                ]
            )
            
            update_data = {
                "access_level": new_access_level,
                "discipline_id": discipline_id,
                "session_id": session_id,
                "updated_at": datetime.now().timestamp()
            }
            
            self.client.update(
                collection_name=self.collection_name,
                points_filter=query_filter,
                payload=update_data
            )
            print(f"[UPDATE] Material atualizado: {material_id}")
            
        except Exception as e:
            print(f"[ERROR] Erro ao atualizar material {material_id}: {str(e)}")
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
        """Realiza busca por similaridade com filtros flexíveis."""
        print(f"\n[SEARCH] Iniciando busca com filtros:")
        print(f"[SEARCH] Query: {query}")
        print(f"[SEARCH] Student: {student_email}")
        print(f"[SEARCH] Config: global={use_global}, discipline={use_discipline}, session={use_session}")
        print(f"[SEARCH] Disciplina: {disciplina_id}, Session: {session_id}")

        try:
            must_conditions = [
                models.FieldCondition(
                    key="metadata.student_email",
                    match=models.MatchValue(value=student_email)
                )
            ]

            if specific_file_id:
                print(f"[SEARCH] Buscando por ID específico: {specific_file_id}")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=specific_file_id)
                    )
                )
            
            if specific_metadata:
                print(f"[SEARCH] Adicionando metadados específicos: {specific_metadata}")
                for key, value in specific_metadata.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=str(value))
                        )
                    )

            allowed_access_levels = []
            
            if use_global:
                print("[SEARCH] Adicionando acesso global aos níveis permitidos")
                allowed_access_levels.append("global")

            if use_discipline and disciplina_id:
                print(f"[SEARCH] Adicionando acesso de disciplina aos níveis permitidos: {disciplina_id}")
                allowed_access_levels.append("discipline")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.discipline_id",
                        match=models.MatchValue(value=disciplina_id)
                    )
                )

            if use_session and session_id:
                print(f"[SEARCH] Adicionando acesso de sessão aos níveis permitidos: {session_id}")
                allowed_access_levels.append("session")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id)
                    )
                )

            if allowed_access_levels:
                print(f"[SEARCH] Níveis de acesso permitidos: {allowed_access_levels}")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.access_level",
                        match=models.MatchAny(any=allowed_access_levels)
                    )
                )

            search_filter = models.Filter(must=must_conditions)
            print(f"[SEARCH] Filtro final construído: {search_filter}")
            
            return self._execute_search(query, search_filter, k)

        except Exception as e:
            print(f"[ERROR] Erro durante a busca: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _execute_search(self, query: str, search_filter: models.Filter, k: int) -> List[Document]:
        """Executa a busca com o filtro construído."""
        try:
            print(f"[SEARCH] Executando busca com filtro: {search_filter}")
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=search_filter
            )

            print(f"[SEARCH] Encontrados {len(results)} resultados")
            for i, doc in enumerate(results, 1):
                print(f"[SEARCH] Resultado {i}:")
                print(f"  - Nível de acesso: {doc.metadata.get('access_level')}")
                print(f"  - Preview: {doc.page_content[:100]}...")

            return results

        except Exception as e:
            print(f"[ERROR] Erro na execução da busca: {str(e)}")
            raise

    def similarity_search_without_filter(self, query: str, k: int = 5):
        """Realiza busca sem filtros."""
        print(f"Realizando busca sem filtro: query={query}")

        try:
            results = self.vector_store.similarity_search(query=query, k=k)
            print(f"{len(results)} documentos encontrados sem filtro.")
            return results
        except Exception as e:
            print(f"Erro na busca sem filtro: {e}")
            return []

    def document_exists(self, content_hash: str, student_email: str, disciplina: str) -> bool:
        """Verifica se um documento já existe na coleção."""
        query_filter = Filter(
            must=[
                FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                FieldCondition(key="student_email", match=MatchValue(value=student_email)),
                FieldCondition(key="disciplina", match=MatchValue(value=disciplina)),
            ]
        )
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1
            )
            exists = len(results) > 0
            print(f"Documento encontrado: {exists}")
            return exists
        except Exception as e:
            print(f"Erro ao verificar existência do documento: {e}")
            return False

    def debug_metadata(self):
        """Depura os metadados dos documentos na coleção."""
        print("Depurando metadados...")

        try:
            results, _ = self.client.scroll(collection_name=self.collection_name, limit=10)
            print(f"{len(results)} documentos encontrados.")

            for result in results:
                metadata = result.payload
                print(f"Metadados Recuperados: {metadata}")

                if 'metadata' in metadata:
                    disciplina_value = metadata['metadata'].get('disciplina')
                    print(f"Disciplina: {disciplina_value} (Tipo: {type(disciplina_value)})")
                else:
                    print("⚠️ O campo 'metadata' está ausente ou mal formatado.")

                if disciplina_value is None:
                    print("⚠️ O campo 'disciplina' está ausente ou vazio.")
                elif not isinstance(disciplina_value, str):
                    print("⚠️ O campo 'disciplina' não é uma string.")

        except Exception as e:
            print(f"Erro ao listar documentos: {e}")

    def fix_document_metadata(self):
        """Corrige os metadados dos documentos com 'disciplina' None."""
        try:
            results, _ = self.client.scroll(collection_name=self.collection_name, limit=10)
            for result in results:
                metadata = result.payload
                if metadata.get("disciplina") is None:
                    metadata["disciplina"] = "1"  # Corrige para string

                    self.client.update_point(
                        collection_name=self.collection_name,
                        id=result.id,
                        payload=metadata
                    )
                    print(f"Metadados corrigidos para o documento {result.id}")
        except Exception as e:
            print(f"Erro ao corrigir metadados: {e}")

    def compare_search_results(self, query: str, student_email: str, disciplina: str, k: int = 5):
        """Compara resultados de busca com e sem filtros."""
        print("🔍 Comparando resultados de busca...")

        print("🔍 Buscando sem filtro...")
        no_filter_results = self.similarity_search_without_filter(query, k)

        print("🔍 Buscando com filtro...")
        filter_results = self.similarity_search_with_filter(query, student_email, disciplina_id=disciplina, k=k)

        print(f"Sem filtro: {len(no_filter_results)} | Com filtro: {len(filter_results)}")
        if len(filter_results) == 0:
            print("⚠️ Nenhum resultado encontrado com filtro. Verifique os metadados e filtros.")

    def compare_search_results(self, query: str, student_email: str, disciplina: str, k: int = 5):
        print("🔍 Comparando resultados de busca...")

        print("🔍 Buscando sem filtro...")
        no_filter_results = self.similarity_search_without_filter(query, k)

        print("🔍 Buscando com filtro...")
        filter_results = self.similarity_search_with_filter(query, student_email, disciplina, k)

        print(f"Sem filtro: {len(no_filter_results)} | Com filtro: {len(filter_results)}")
        if len(filter_results) == 0:
            print("⚠️ Nenhum resultado encontrado com filtro. Verifique os metadados e filtros.")

    def document_exists(self, content_hash: str, student_email: str, disciplina: int) -> bool:
        query_filter = Filter(
            must=[
                FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                FieldCondition(key="student_email", match=MatchValue(value=student_email)),
                FieldCondition(key="disciplina", match=MatchValue(value=disciplina)),
            ]
        )
        try:
            results, _ = self.client.scroll(collection_name=self.collection_name, scroll_filter=query_filter, limit=1)
            exists = len(results) > 0
            print(f"Documento encontrado: {exists}")
            return exists
        except Exception as e:
            print(f"Erro ao verificar existência do documento: {e}")
            return False
class TextSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 50):
        """
        Inicializa o TextSplitter com tamanho de chunk e sobreposição.
        
        :param chunk_size: Tamanho máximo de cada pedaço de texto
        :param chunk_overlap: Sobreposição entre pedaços
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos em pedaços menores.
        
        :param documents: Lista de documentos a serem divididos
        :return: Lista de documentos divididos
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)


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