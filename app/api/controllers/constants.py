from dotenv import load_dotenv
import os

load_dotenv()


key = os.getenv("SECRET_KEY")

SECRET_KEY = key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
