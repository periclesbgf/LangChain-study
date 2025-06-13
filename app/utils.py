from dotenv import load_dotenv
import os

load_dotenv()

def _decode_env(var_name: str) -> str:
    raw = os.getenv(var_name, "")
    return raw.encode('utf-8').decode('unicode_escape')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CODE = os.getenv("CODE")
SECRET_EDUCATOR_CODE = os.getenv("SECRET_EDUCATOR_CODE")
SECRET_KEY = os.getenv("SECRET_KEY")

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_URI = _decode_env("MONGO_URI")

QDRANT_URL = _decode_env("QDRANT_URL")
APP_URL = _decode_env("APP_URL")
RESET_PASSWORD_URL = _decode_env("RESET_PASSWORD_URL")
LOKI_ADDRESS = _decode_env("LOKI_ADDRESS")

SMTP_SERVER = _decode_env("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")