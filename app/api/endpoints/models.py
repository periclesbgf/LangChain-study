from pydantic import BaseModel
from typing import Optional
from fastapi import Depends
from api.controllers.auth import get_current_user
from datetime import datetime


class Question(BaseModel):
    question: str
    code: str

class ResponseModel(BaseModel):
    response: str
    audio: Optional[str] = None

class RegisterModel(BaseModel):
    nome: str
    email: str
    senha: str
    tipo_usuario: str

class LoginModel(BaseModel):
    email: str
    senha: str

class Token(BaseModel):
    access_token: str
    token_type: str

class PromptModel(BaseModel):
    question: str
    code: str

class StudySessionCreate(BaseModel):
    IdCurso: int
    Assunto: str
    Inicio: Optional[datetime] = None
    Fim: Optional[datetime] = None
    Produtividade: Optional[int] = None
    FeedbackDoAluno: Optional[str] = None
