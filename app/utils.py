from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CODE = os.getenv("CODE")
SECRET_EDUCATOR_CODE = os.getenv("SECRET_EDUCATOR_CODE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_URI = os.getenv("MONGO_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
APP_URL = os.getenv("APP_URL")
RESET_PASSWORD_URL = os.getenv("RESET_PASSWORD_URL")

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

LOKI_ADDRESS = os.getenv("LOKI_ADDRESS")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")