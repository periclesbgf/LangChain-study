from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import PyPDFLoader

data_path = "/Users/peric/projects/LangChain-study/app/database/126887.pdf"


loader = PyPDFLoader(file_path=data_path)
documents = loader.load()

print("Type of documents:", type(documents))
print("First few documents:", documents)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


text = text_splitter.split_documents(documents)

if not text:
    print("No text documents were processed. Please check the format of the input data.")

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

if text:
    qdrant = Qdrant.from_documents(
        text,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name=collection_name,
    )
    print("Qdrant Index created")
else:
    print("Failed to create Qdrant Index due to lack of valid text data.")
