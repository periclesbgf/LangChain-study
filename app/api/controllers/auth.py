# controllers/auth.py

from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from google.oauth2 import credentials
from google_auth_oauthlib.flow import Flow
import smtplib
from email.mime.text import MIMEText
from api.controllers.constants import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    RESET_TOKEN_EXPIRY_MINUTES,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    GOOGLE_PROJECT_ID,
)
from utils import(
    SMTP_SERVER,
    SMTP_PORT,
    SENDER_EMAIL,
    SENDER_PASSWORD,
    RESET_PASSWORD_URL,
)
from logg import logger

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Escopos necessários para Calendar, Classroom, userinfo e OpenID Connect
SCOPES = [
    # Google Calendar (somente leitura)
    "https://www.googleapis.com/auth/calendar.readonly",

    # Google Classroom (somente leitura)
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/classroom.announcements.readonly",

    # Escopos adicionais para obter informações básicas do perfil (se necessário)
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload if payload else None
    except JWTError:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado")
    return payload

def create_google_flow(request: Request):  # Função para criar o Flow OAuth2
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "project_id": GOOGLE_PROJECT_ID,  # Certifique-se de que essa variável esteja definida
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": [GOOGLE_REDIRECT_URI],
                "javascript_origins": [str(request.base_url).rstrip("/")]
            }
        },
        scopes=SCOPES,
        redirect_uri=GOOGLE_REDIRECT_URI
    )

def credentials_to_dict(credentials_obj: credentials.Credentials) -> dict:
    """Função para serializar as credentials."""
    return {
        'token': credentials_obj.token,
        'refresh_token': credentials_obj.refresh_token,
        'token_uri': credentials_obj.token_uri,
        'client_id': credentials_obj.client_id,
        'client_secret': credentials_obj.client_secret,
        'scopes': credentials_obj.scopes
    }

def dict_to_credentials(credentials_dict: dict) -> credentials.Credentials:
    """Função para desserializar as credentials."""
    return credentials.Credentials(**credentials_dict)

def create_reset_token(email: str, expires_delta: timedelta = None):
    payload = {"sub": email}

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_EXPIRY_MINUTES)
    payload.update({"exp": expire})
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

def decode_reset_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload if payload else None
    except JWTError:
        return None

def send_reset_email(recipient_email: str, reset_token: str):
    """
    Envia um email contendo um link para resetar a senha para o endereço do usuário.
    O link é composto pela URL base (definida em RESET_LINK_BASE_URL) com o token
    passado como parâmetro de query.
    """
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        raise HTTPException(status_code=500, detail="Configuração de email inválida.")

    reset_link = f"{RESET_PASSWORD_URL}?token={reset_token}"

    subject = "Reset de Senha - Eden AI"
    body = (
        f"Olá,\n\nVocê solicitou o reset da sua senha. Por favor, clique no link abaixo para redefinir sua senha:\n\n"
        f"{reset_link}\n\n"
        "O link é válido por 30 minutos. Após esse período, será necessário solicitar um novo reset.\n\n"
        "Se você não fez essa solicitação, por favor, ignore este email."
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            logger.info(f"[RESET_EMAIL] Email enviado para {recipient_email}")
        return True
    except Exception as e:
        print(f"Erro ao enviar email: {e}")
        raise HTTPException(status_code=500, detail="Erro ao enviar email de reset.")