from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from io import BytesIO
from langchain.docstore.document import Document

from qdrant_client import QdrantClient
import fitz

class DocumentLoader:
    def __init__(self, file_bytes=None, file_path=None):
        self.file_bytes = file_bytes
        self.file_path = file_path

    def load_documents(self):
        if self.file_bytes:
            document = fitz.open(stream=self.file_bytes, filetype="pdf")
        else:
            document = fitz.open(self.file_path)

        documents = []
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text()
            documents.append(Document(page_content=text, metadata={"page_number": page_num}))

        return documents

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

class QdrantIndex:
    def __init__(self, url, collection_name, embeddings):
        self.url = url
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = self.create_client()

    def create_index(self, text):
        qdrant = Qdrant.from_documents(
            text,
            self.embeddings,
            url=self.url,
            prefer_grpc=False,
            collection_name=self.collection_name,
        )
        return qdrant

    def create_client(self):
        return QdrantClient(
            url=self.url,
            prefer_grpc=False,
        )

    def create_db(self):
        return Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    def similarity_search(self, query, k=5):
        return self.create_db().similarity_search_with_score(query=query, k=k)

