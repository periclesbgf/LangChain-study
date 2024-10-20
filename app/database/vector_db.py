# app/database/vector_db.py

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
from langchain_core.documents import Document
from typing import List, Optional
import hashlib

class QdrantHandler:
    def __init__(self, url: str, collection_name: str, embeddings):
        print(f"Inicializando QdrantHandler para a coleção '{collection_name}'...")
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(url=url, prefer_grpc=False)
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

    def add_document(self, student_email: str, disciplina: str, content: str, metadata_extra: Optional[dict] = None):
        print(f"Adicionando documento: student_email={student_email}, disciplina={disciplina}")
        print(f"Tipo de 'disciplina' antes do armazenamento: {type(disciplina)}")

        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        metadata = {
            "student_email": student_email,
            "disciplina": str(disciplina),  # Forçando para string
            "content_hash": content_hash
        }

        if metadata_extra:
            metadata.update(metadata_extra)

        document = Document(page_content=content, metadata=metadata)
        self.vector_store.add_documents([document])
        print("Documento adicionado com sucesso.")

        # Recuperar e exibir os metadados logo após a inserção
        self.debug_metadata()


    def similarity_search_with_filter(self, query: str, student_email: str, disciplina: str, k: int = 5):
        """
        Realiza uma busca de similaridade no Qdrant utilizando filtros.
        """
        print(f"Realizando busca com filtro: query={query}, student_email={student_email}, disciplina={disciplina}")
        print(f"Tipos: query={type(query)}, student_email={type(student_email)}, disciplina={type(disciplina)}")

        # Ajustar filtro para considerar o formato dos metadados aninhados
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.student_email",  # Acessando corretamente o campo aninhado
                    match=models.MatchValue(value=student_email)
                ),
                models.FieldCondition(
                    key="metadata.disciplina",  # Acessando corretamente o campo aninhado
                    match=models.MatchValue(value=str(disciplina))  # Garantir string
                ),
            ]
        )

        print(f"Filtro construído: {query_filter}")

        try:
            # Realizando a busca com filtro
            results = self.vector_store.similarity_search(query=query, k=k, filter=query_filter)
            print(f"{len(results)} documentos encontrados com filtro.")
            for doc in results:
                print(f"Conteúdo: {doc.page_content} | Metadados: {doc.metadata}")
            return results
        except Exception as e:
            print(f"Erro na busca com filtro: {e}")
            return []



    def similarity_search_without_filter(self, query: str, k: int = 5):
        print(f"Realizando busca sem filtro: query={query}")

        try:
            results = self.vector_store.similarity_search(query=query, k=k)
            print(f"{len(results)} documentos encontrados sem filtro.")
            return results
        except Exception as e:
            print(f"Erro na busca sem filtro: {e}")
            return []

    def debug_metadata(self):
        print("Depurando metadados...")

        try:
            # Recupera documentos da coleção com scroll
            results, _ = self.client.scroll(collection_name=self.collection_name, limit=10)
            print(f"{len(results)} documentos encontrados.")

            for result in results:
                # Exibe todo o payload para análise completa
                metadata = result.payload
                print(f"Metadados Recuperados: {metadata}")

                # Acessa o campo 'disciplina' corretamente
                if 'metadata' in metadata:
                    disciplina_value = metadata['metadata'].get('disciplina')
                    print(f"Disciplina: {disciplina_value} (Tipo: {type(disciplina_value)})")
                else:
                    print("⚠️ O campo 'metadata' está ausente ou mal formatado.")

                # Validar se 'disciplina' está correta
                if disciplina_value is None:
                    print("⚠️ O campo 'disciplina' está ausente ou vazio.")
                elif not isinstance(disciplina_value, str):
                    print("⚠️ O campo 'disciplina' não é uma string.")

        except Exception as e:
            print(f"Erro ao listar documentos: {e}")




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

    def fix_document_metadata(self):
        """
        Corrige os metadados dos documentos que possuem 'disciplina' como None.
        """
        try:
            results, _ = self.client.scroll(collection_name=self.collection_name, limit=10)
            for result in results:
                metadata = result.payload
                if metadata.get("disciplina") is None:
                    metadata["disciplina"] = "1"  # Corrigir para string

                    # Atualizar o documento no Qdrant
                    self.client.update_point(
                        collection_name=self.collection_name,
                        id=result.id,
                        payload=metadata
                    )
                    print(f"Metadados corrigidos para o documento {result.id}")
        except Exception as e:
            print(f"Erro ao corrigir metadados: {e}")




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