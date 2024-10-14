# app/database/vector_db.py

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
import uuid
from qdrant_client.http.models import PointStruct
from langchain.vectorstores import Qdrant
from PIL import Image
import base64
import io
import fitz


class QdrantHandler:
    def __init__(self, url, collection_name, embeddings):
        """
        Inicializa o cliente Qdrant e as variáveis necessárias.
        :param url: URL da instância do Qdrant
        :param collection_name: Nome da coleção a ser usada no Qdrant
        :param embeddings: Função de embeddings (ex: HuggingFaceEmbeddings)
        """
        self.url = url
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = self.create_client()

    def create_client(self):
        """
        Cria um cliente Qdrant.
        :return: Instância de QdrantClient
        """
        return QdrantClient(url=self.url, prefer_grpc=False)

    def create_index(self, texts, tables, images, text_summaries, table_summaries, image_summaries):
        """
        Cria ou usa uma coleção existente no Qdrant e indexa os documentos (textos, tabelas, imagens).
        :param texts: Lista de textos a serem indexados
        :param tables: Lista de tabelas a serem indexadas
        :param images: Lista de imagens base64
        :param text_summaries: Resumos dos textos
        :param table_summaries: Resumos das tabelas
        :param image_summaries: Resumos das imagens
        """

        # Verifica se a coleção existe, caso contrário, cria
        if not self.client.get_collection(self.collection_name):
            self.client.create_collection(self.collection_name)

        # Indexar os textos
        if texts and text_summaries:
            self._add_documents(text_summaries, texts, "text")

        # Indexar as tabelas
        if tables and table_summaries:
            self._add_documents(table_summaries, tables, "table")

        # Indexar as imagens
        if images and image_summaries:
            self._add_documents(image_summaries, images, "image")

    def _add_documents(self, summaries, contents, content_type):
        """
        Função auxiliar para adicionar documentos ao Qdrant.
        :param summaries: Resumos dos documentos (textos, tabelas, imagens)
        :param contents: Conteúdo real dos documentos
        :param content_type: Tipo de conteúdo ('text', 'table', 'image')
        """

        embeddings = [self.embeddings.embed_text(summary) for summary in summaries]

        # Cria pontos estruturados para o Qdrant
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # Gera um ID único para cada documento
                vector=embedding,  # O vetor de embeddings do resumo
                payload={"content_type": content_type, "content": content, "summary": summary},
            )
            for content, embedding, summary in zip(contents, embeddings, summaries)
        ]

        # Insere os pontos na coleção
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"{len(points)} {content_type}s indexados com sucesso!")

    def similarity_search(self, query, k=5):
        """
        Realiza uma busca de similaridade no Qdrant.
        :param query: Texto da query
        :param k: Número de resultados a retornar (default: 5)
        :return: Lista de documentos mais similares
        """
        # Gera o embedding para a query
        query_embedding = self.embeddings.embed_text(query)

        # Realiza a busca de similaridade
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
        )

        # Retorna os documentos mais similares com suas pontuações e tipo de conteúdo
        return [(result.payload['content'], result.payload['content_type'], result.score) for result in results]

    def encode_image(self, image_path):
        """
        Codifica uma imagem em uma string base64.
        :param image_path: Caminho da imagem a ser codificada
        :return: String base64 da imagem
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def resize_base64_image(self, base64_string, size=(128, 128)):
        """
        Redimensiona uma imagem codificada como uma string Base64.
        :param base64_string: String base64 da imagem a ser redimensionada
        :param size: Tamanho final da imagem redimensionada (default: 128x128)
        :return: String base64 da imagem redimensionada
        """
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))

        # Redimensiona a imagem
        resized_img = img.resize(size, Image.LANCZOS)

        # Salva a imagem redimensionada em um buffer de bytes
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)

        # Codifica a imagem redimensionada para base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)

    def recursive_text_splitter(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)


class Embeddings:
    def __init__(self):
        self.model_name = "BAAI/bge-large-en"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        return HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def get_embeddings(self):
        return self.embeddings
