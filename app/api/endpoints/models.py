from pydantic import BaseModel
from typing import Optional, Dict, List
from fastapi import Depends,Form, UploadFile, File
from pydantic import BaseModel, Field, EmailStr
from api.controllers.auth import get_current_user
from datetime import datetime, date, timezone


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
    message: Optional[str] = None
    discipline_id: int
    file: Optional[UploadFile] = None

    @classmethod
    def as_form(
        cls,
        session_id: int = Form(...),
        message: Optional[str] = Form(None),
        discipline_id: int = Form(...),
        file: Optional[UploadFile] = File(None)
    ):
        return cls(session_id=session_id, message=message, discipline_id=discipline_id, file=file)



class EstiloAprendizagem(BaseModel):
    Percepcao: str = Field(..., example="Sensorial")
    Entrada: str = Field(..., example="Visual")
    Processamento: str = Field(..., example="Ativo")
    Entendimento: str = Field(..., example="Sequencial")

class Feedback(BaseModel):
    data: Optional[datetime] = Field(None, example="2024-10-14")
    disciplina: Optional[str] = Field(None, example="Introdução à Programação")
    feedback: Optional[str] = Field(None, example="Bom progresso.")
    acoesRecomendadas: Optional[List[str]] = Field(None, example=["Praticar loops"])

class PreferenciasAprendizado(BaseModel):
    Dificuldades: Optional[List[str]] = Field(None, example=["Estruturas de decisão"])
    PreferenciaRecursos: Optional[str] = Field(None, example="Exemplos visuais")
    feedbackDoTutor: Optional[List[Feedback]] = Field(default_factory=list)
    feedbackDoProfessor: Optional[List[Feedback]] = Field(default_factory=list)




class Recurso(BaseModel):
    tipo: str = Field(..., example="Vídeo")
    descricao: str = Field(..., example="Fluxograma explicando estruturas de decisão.")
    url: Optional[str] = Field(None, example="https://example.com/video")

class Atividade(BaseModel):
    descricao: str = Field(..., example="Resolver exercícios práticos em grupo.")
    tipo: str = Field(..., example="Exercício prático")
    formato: Optional[str] = Field(None, example="Colaborativo")

class SecaoPlano(BaseModel):
    titulo: str = Field(..., example="Introdução à Programação Funcional")
    duracao: str = Field(..., example="25 minutos")
    descricao: Optional[str] = Field(None, example="Exploração de conceitos fundamentais e exemplos práticos.")
    conteudo: List[str] = Field(..., example=[
        "Definição de programação funcional.",
        "Uso de funções lambda em Python."
    ])
    recursos: Optional[List[Recurso]] = None
    atividade: Optional[Atividade] = None
    progresso: int = Field(..., example=25)

class PlanoExecucao(BaseModel):
    id_sessao: int = Field(..., example=1)  # Novo campo para associar à sessão de estudos
    disciplina: str = Field(..., example="Programação Imperativa e Funcional")
    descricao: Optional[str] = Field(None, example="Revisão detalhada dos principais conceitos de programação imperativa e funcional.")
    objetivo_sessao: str = Field(..., example="Introdução e revisão de conceitos fundamentais.")
    plano_execucao: List[SecaoPlano] = Field(...)
    duracao_total: str = Field(..., example="90 minutos")
    progresso_total: int = Field(..., example=100)
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))

class StudySessionCreate(BaseModel):
    discipline_id: int = Field(..., description="ID da disciplina associada à sessão de estudo")
    subject: str = Field(..., description="Assunto da sessão de estudo", min_length=1)
    start_time: datetime = Field(..., description="Data e hora de início da sessão")
    end_time: datetime = Field(..., description="Data e hora de término da sessão")
