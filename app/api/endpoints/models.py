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
    instituicao: Optional[str] = None
    special_code: Optional[str] = None

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

class CalendarEvent(BaseModel):
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: str

class CalendarEventUpdate(BaseModel):
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: str

class DisciplineCreate(BaseModel):
    nome_curso: str
    ementa: Optional[str] = None
    objetivos: Optional[str] = None
    educator: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "nome_curso": "Computer Science 101",
                "ementa": "Introduction to Computer Science fundamentals",
                "objetivos": "Teach students the basics of computer programming, algorithms, and data structures."
            }
        }

class DisciplineUpdate(BaseModel):
    nome_curso: Optional[str] = None
    ementa: Optional[str] = None
    objetivos: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "nome_curso": "Advanced Computer Science",
                "ementa": "Advanced topics in Computer Science including algorithms, AI, and networking.",
                "objetivos": "Provide a deeper understanding of complex computer science concepts."
            }
        }

class StudentCreate(BaseModel):
    name: str
    matricula: Optional[str] = None

    class Config:
        from_attributes = True

class StudentUpdate(BaseModel):
    name: Optional[str] = None
    matricula: Optional[str] = None

    class Config:
        from_attributes = True

class EducatorCreate(BaseModel):
    name: str
    instituicao: str
    especializacao_disciplina: str

    class Config:
        from_attributes = True


class EducatorUpdate(BaseModel):
    name: Optional[str] = None
    instituicao: Optional[str] = None
    especializacao_disciplina: Optional[str] = None

    class Config:
        from_attributes = True

class MessageRequest(BaseModel):
    session_id: int
    message: str
