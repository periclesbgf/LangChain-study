from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_documents(self):
        loader = PyPDFLoader(file_path=self.file_path)
        documents = loader.load()
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
    def __init__(self, model_name, model_kwargs=None, encode_kwargs=None):
        if encode_kwargs is None:
            encode_kwargs = {'normalize_embeddings': False}
        self.model_name = model_name
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = encode_kwargs
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

    def create_index(self, text):
        qdrant = Qdrant.from_documents(
            text,
            self.embeddings,
            url=self.url,
            prefer_grpc=False,
            collection_name=self.collection_name,
        )
        return qdrant

def main(data_path):
    # Load Documents
    loader = DocumentLoader(file_path=data_path)
    documents = loader.load_documents()
    print("Type of documents:", type(documents))
    print("First few documents:", documents)

    splitter = TextSplitter()
    text = splitter.split_documents(documents)

    if not text:
        print("No text documents were processed. Please check the format of the input data.")
        return

    model_name = "BAAI/bge-large-en"
    embeddings_obj = Embeddings(model_name=model_name)
    embeddings = embeddings_obj.get_embeddings()
    print("Embedding models loaded")

    url = "http://localhost:6333"
    collection_name = "gpt_db"
    qdrant_index = QdrantIndex(url, collection_name, embeddings)
    qdrant = qdrant_index.create_index(text)
    print("Qdrant Index created")

if __name__ == "__main__":
    data_path = "/Users/peric/projects/LangChain-study/app/database/data/Liquidity_Is_All_You_Need.pdf"
    main(data_path)
