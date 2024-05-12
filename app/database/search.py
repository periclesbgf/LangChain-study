from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient


model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding models loaded")

url = "http://localhost:6333"
collection_name = "gpt_db"


qdrant_client = QdrantClient(
    url=url,
    prefer_grpc=False,
)
print(qdrant_client)
print("####################")

db = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings
)

print(db)
print("####################")

query = "como IA pode ajudar na educacao?"

docs = db.similarity_search_with_score(query=query, k=5)

for i in docs:
    doc,score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})