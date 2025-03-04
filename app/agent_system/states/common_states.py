from pydantic import BaseModel
from typing import Any, TypedDict, List, Dict, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Dict, Any


class UserProfile(BaseModel):
    Nome: str
    Email: str
    EstiloAprendizagem: Dict[str, str]
    Feedback: Optional[Dict[str, Any]] = None
    PreferenciaAprendizado: Optional[Dict[str, Any]] = None

class ExecutionStep(BaseModel):
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[Dict]
    atividade: Dict
    progresso: int

class ExecutionPlan(BaseModel):
    id_sessao: str
    disciplina: str
    plano_execucao: List[ExecutionStep]
    duracao_total: str
    progresso_total: int
    created_at: str
    updated_at: str



class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_plan: str
    user_profile: Dict[str, Any]
    extracted_context: Dict[str, Any]
    next_step: Optional[str]
    iteration_count: int
    chat_history: List[BaseMessage]
    needs_retrieval: Optional[bool]
    evaluation_reason: Optional[str]
    web_search_results: Dict[str, str]
    answer_type: Optional[str]
    current_progress: Dict[str, Any]
    session_id: str
    actions_history: List[Dict[str, Any]]
    thoughts_history: List[Dict[str, Any]]
    memories: List[Dict[str, Any]]
    thoughts: str
    final_answer: Optional[str]
    observations: Optional[str]


