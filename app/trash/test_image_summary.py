import os
import io
import re
import uuid
import base64
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from openai import OpenAI as OpenAI_vLLM
from langchain_community.llms.vllm import VLLMOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CODE = os.getenv("CODE")
SECRET_EDUCATOR_CODE = os.getenv("SECRET_EDUCATOR_CODE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_URI = os.getenv("MONGO_URI")

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')

dir = "test_data"

# Extract elements from PDF
def extract_pdf_elements(path, fname):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )

# Categorize elements by type
def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

# File path
folder_path = "./test_data/"
file_name = "pbgf_sec.pdf"

# Get elements
raw_pdf_elements = extract_pdf_elements(folder_path, file_name)
print("No of Elements Extracted:", len(raw_pdf_elements))

# Get text, tables
texts, tables = categorize_elements(raw_pdf_elements)

# Enforce a specific token size for texts
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000, chunk_overlap = 0
)
joined_texts = " ".join(texts)
texts_token = text_splitter.split_text(joined_texts)

print("No of Textual Chunks:", len(texts))
print("No of Table Elements:", len(tables))
print("No of Text Chunks after Tokenization:", len(texts_token))

# Initialize vLLM API server
llm_client = VLLMOpenAI(
    api_key = OPENAI_API_KEY,
    model_name = "llava-hf/llava-1.5-7b-hf",
    temperature = 1.0,
    max_tokens = 300
)

# Generate summaries of table elements
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \
    Give a concise summary of the table that is well optimized for retrieval. Make sure to capture all the details. \
    Input: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    summarize_chain = {"element": lambda x: x} | prompt | llm_client | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    elif texts:
        text_summaries = texts
    
    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 3})
    return text_summaries, table_summaries


# Get text, table summaries
text_summaries, table_summaries = generate_text_summaries(
    texts_token, tables, summarize_texts=False
)
print("No of Text Summaries:", len(text_summaries))
print("No of Table Summaries:", len(table_summaries))

# Display text summaries
for i, summary in enumerate(text_summaries):
    print(f"Text Summary {i+1}: {summary}")

# vLLM OpenAI-compatible API client
api_key = OPENAI_API_KEY
vlm_client = OpenAI_vLLM(
    api_key = api_key,
)

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat_response = vlm_client.chat.completions.create(
        model="llava-hf/llava-1.5-7b-hf",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                },
            ],
        }],
        stream=False
    )
    return chat_response.choices[0].message.content.strip()

def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """
    # Store base64 encoded images
    img_base64_list = []
    # Store image summaries
    image_summaries = []
    # Prompt
    prompt = """You are an assistant tasked with summarizing images for optimal retrieval. \
    These summaries will be embedded and used to retrieve the raw image.
    Write a clear and concise summary that captures all the important information, including any statistics or key points present in the image."""
    # Apply to images
    for img_file in tqdm(sorted(os.listdir(path))):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            # image_summaries.append(image_summarize(base64_image, prompt))
            generated_summary = image_summarize(base64_image, prompt)
            print(generated_summary)
            image_summaries.append(generated_summary)
    return img_base64_list, image_summaries


# Image summaries
img_base64_list, image_summaries = generate_img_summaries(folder_path)
assert len(img_base64_list) == len(image_summaries)


def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)
    return retriever

# The vectorstore to use to index the summaries
vectorstore = Chroma(
    collection_name="mm_rag_vectorstore", embedding_function=embeddings, persist_directory="./chroma_db" 
)


# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)







def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(64, 64)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)
    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(64, 64))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are an AI assistant with expertise in finance and business metrics.\n"
            "You will be given information that may include text, tables, and charts related to business performance and industry trends.\n"
            "Your task is to analyze this information and provide a clear, concise answer to the user's question.\n"
            "Focus on the most relevant data points and insights that directly address the user's query.\n"
            f"User's question: {data_dict['question']}\n\n"
            "Information provided:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    return [HumanMessage(content=messages)]

def multi_modal_rag_context_chain(retriever):
    """Multi-modal RAG context chain"""
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
    )
    return chain


# Retrieve the relevant context including text and images
chain_multimodal_context = multi_modal_rag_context_chain(retriever_multi_vector_img)


# Check retrieval
query = "How has the median YoY ARR growth rate for public SaaS companies changed from 2020 to 2024?"
docs = retriever_multi_vector_img.invoke(query)