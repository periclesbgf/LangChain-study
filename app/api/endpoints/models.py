from pydantic import BaseModel
from typing import Optional, Dict, List
from fastapi import Depends,Form, UploadFile, File
from pydantic import BaseModel, Field, EmailStr
from api.controllers.auth import get_current_user
from datetime import datetime, date, timezone
from enum import Enum
from uuid import UUID


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
    matricula: Optional[str] = None

class LoginModel(BaseModel):
    email: str
    senha: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

class RefreshTokenRequest(BaseModel):
    refresh_token: str

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
    categoria: str
    importancia: str
    material: Optional[str] = None

class CalendarEventUpdate(BaseModel):
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: str
    categoria: str
    importancia: str
    material: Optional[str] = None

class DisciplineCreate(BaseModel):
    nome_curso: str
    ementa: Optional[str] = None
    objetivos: Optional[str] = None
    educator: Optional[str] = None
    turno_estudo: str  # 'manha', 'tarde', 'noite'
    horario_inicio: str  # formato HH:MM
    horario_fim: str    # formato HH:MM

    class Config:
        json_schema_extra = {
            "example": {
                "nome_curso": "Computer Science 101",
                "ementa": "Introduction to Computer Science fundamentals",
                "objetivos": "Teach students the basics of computer programming, algorithms, and data structures.",
                "educator": "John Doe",
                "turno_estudo": "manha",
                "horario_inicio": "08:00",
                "horario_fim": "10:00"
            }
        }

class DisciplineUpdate(BaseModel):
    nome_curso: Optional[str] = None
    ementa: Optional[str] = None
    objetivos: Optional[str] = None
    turno_estudo: Optional[str] = None
    horario_inicio: Optional[str] = None
    horario_fim: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "nome_curso": "Advanced Computer Science",
                "ementa": "Advanced topics in Computer Science",
                "objetivos": "Provide a deeper understanding of concepts",
                "turno_estudo": "tarde",
                "horario_inicio": "14:00",
                "horario_fim": "16:00"
            }
        }

class DisciplinePDFCreate(BaseModel):
    turno_estudo: str
    horario_inicio: str
    horario_fim: str

    class Config:
        json_schema_extra = {
            "example": {
                "turno_estudo": "manha",
                "horario_inicio": "08:00",
                "horario_fim": "10:00"
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

class StudySessionCreate(BaseModel):
    discipline_id: int
    subject: str
    start_time: datetime
    end_time: datetime

class ExecutionStep(BaseModel):
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[Dict]
    atividade: Dict
    progresso: int = 0


class StudyPlan(BaseModel):
    # Campos de identificação
    id_sessao: str
    disciplina_id: Optional[str] = None
    disciplina: Optional[str] = None
    
    # Campos de conteúdo
    descricao: Optional[str] = None
    objetivo_sessao: Optional[str] = None
    plano_execucao: List[ExecutionStep] = Field(default_factory=list)
    
    # Campos de controle
    duracao_total: str
    progresso_total: int = Field(default=0, ge=0, le=100)
    
    # Campos de timestamp
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    # Campos de feedback e análise
    feedback_geral: Dict = Field(default_factory=dict)

class AccessLevel(str, Enum):
    GLOBAL = "global"
    DISCIPLINE = "discipline"
    SESSION = "session"

class Material(BaseModel):
    id: str
    name: str
    type: str  # "pdf", "image", "doc"
    access_level: AccessLevel
    discipline_id: Optional[str]
    session_id: Optional[str]
    student_email: str
    content_hash: str
    created_at: datetime
    updated_at: datetime

class AudioResponseModel(BaseModel):
    response: str
    audio: Optional[str]

class GoogleLoginRequest(BaseModel):
    token: str

class StudySession(BaseModel):
    id: int
    course_id: int
    user_id: int
    title: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    status: int
    notes: Optional[str]
    resources: Optional[str]
    period: str

class AutomaticStudyPlanRequest(BaseModel):
    #disciplina_id: Optional[str] = Field(..., description="ID da disciplina")
    session_id: str = Field(..., description="ID da sessão do usuário")
    tema: str = Field(..., description="Descrição do tema de estudo")
    duracao_desejada: int = Field(..., description="Duração desejada em minutos")
    periodo: str = Field(..., description="Período de estudo (manhã, tarde ou noite)")
    objetivos: Optional[List[str]] = Field(default_factory=list, description="Lista de objetivos")

class AutomaticStudyPlanResponse(BaseModel):
    message: str
    plano: dict

class ResetPasswordModel(BaseModel):
    reset_token: str
    new_password: str
    confirm_password: str

class ForgotPasswordModel(BaseModel):
    user_email: str

class SupportRequest(BaseModel):
    message_type: str
    subject: str
    page: str
    message: str
    images: Optional[List[UploadFile]] = None

