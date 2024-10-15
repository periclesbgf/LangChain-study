from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CODE = os.getenv("CODE")
SECRET_EDUCATOR_CODE = os.getenv("SECRET_EDUCATOR_CODE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_URI = os.getenv("MONGO_URI")
QDRANT_URL = os.getenv("QDRANT_URL")