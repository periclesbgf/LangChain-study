from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

class EmbeddingsModel:
    def __init__(self, model_name, model_kwargs=None, encode_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {'device': 'cpu'}
        if encode_kwargs is None:
            encode_kwargs = {'normalize_embeddings': False}
        self.model_name = model_name
        self.model_kwargs = model_kwargs
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

class QdrantDatabase:
    def __init__(self, url, collection_name, embeddings):
        self.url = url
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = self.create_client()
        self.db = self.create_db()


def main():
    model_name = "BAAI/bge-large-en"
    url = "http://localhost:6333"
    collection_name = "gpt_db"

    # Load Embeddings Model
    embeddings_model = EmbeddingsModel(model_name=model_name)
    embeddings = embeddings_model.get_embeddings()
    print("Embedding models loaded")

    # Setup Qdrant Database
    qdrant_db = QdrantDatabase(url=url, collection_name=collection_name, embeddings=embeddings)
    print("Qdrant Database setup complete")

    # Process Query
    query = "como IA pode ajudar na educacao?"
    docs = qdrant_db.similarity_search(query=query, k=5)

    for doc, score in docs:
        print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

if __name__ == "__main__":
    main()
