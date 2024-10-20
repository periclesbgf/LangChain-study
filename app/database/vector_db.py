# app/database/vector_db.py

from langchain.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings  # Corrige depreciação
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
import uuid
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
)
from utils import OPENAI_API_KEY, QDRANT_URL
import base64
import io

class QdrantHandler:
    def __init__(self, url, collection_name, embeddings):
        """
        Inicializa o cliente Qdrant e as variáveis necessárias.
        :param url: URL da instância do Qdrant
        :param collection_name: Nome da coleção a ser usada no Qdrant
        :param embeddings: Função de embeddings (ex: OpenAIEmbeddings)
        """
        self.url = url
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = self.create_client()
        self.ensure_collection_exists()

    def create_client(self):
        """
        Cria um cliente Qdrant.
        :return: Instância de QdrantClient
        """
        return QdrantClient(url=self.url, prefer_grpc=False)

    def ensure_collection_exists(self):
        """
        Garante que a coleção exista no Qdrant.
        """
        collections = self.client.get_collections().collections
        if self.collection_name not in [col.name for col in collections]:
            vector_size = len(self.embeddings.embed_query("test"))
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance="Cosine"
                )
            )

    def add_document(self, student_email, disciplina, content, embedding, metadata):
        """
        Adiciona um documento ao banco vetorial com metadados.
        """
        payload = {
            "student_email": student_email,
            "disciplina": disciplina,
            "content": content,
            **metadata,
        }
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload,
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def similarity_search(self, embedding, student_email, disciplina, k=5):
        """
        Realiza uma busca de similaridade filtrada por student_email e disciplina.
        """
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="student_email",
                    match=MatchValue(value=student_email)
                ),
                FieldCondition(
                    key="disciplina",
                    match=MatchValue(value=disciplina)
                ),
            ]
        )
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=k,
        )
        return [result.payload for result in results]


class TextSplitter:
    def __init__(self, chunk_size=1024, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)


class Embeddings:
    def __init__(self):
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        # Corrigindo para a nova importação
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def get_embeddings(self):
        return self.embeddings
