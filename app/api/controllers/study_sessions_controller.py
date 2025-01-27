# controllers/study_sessions_controller.py

from chains.chain_setup import DisciplinChain
from database.sql_database_manager import DatabaseManager
from sql_test.sql_test_create import tabela_cursos, tabela_sessoes_estudo, tabela_encontros, tabela_eventos_calendario
from database.search import execute_query
from database.vector_db import TextSplitter, Embeddings, QdrantHandler
from audio.text_to_speech import AudioService
from fastapi.logger import logger
from datetime import datetime
from api.controllers.plan_controller import PlanController
from api.dispatchers.study_sessions_dispatcher import StudySessionsDispatcher
from api.dispatchers.discipline_dispatcher import DisciplineDispatcher
import json
from datetime import datetime, timezone
from typing import Dict, Any

from utils import OPENAI_API_KEY, CODE


class StudySessionsController:
    def __init__(self, dispatcher: StudySessionsDispatcher, disciplin_chain: DisciplinChain = None, discipline_dispatcher:DisciplineDispatcher=None):
        self.dispatcher = dispatcher
        self.disciplin_chain = disciplin_chain
        self.plan_controller = PlanController()

    def get_all_study_sessions(self, user_email: str):
        try:
            # Usar o dispatcher para buscar todas as sessões de estudo pelo ID do estudante
            study_sessions = self.dispatcher.get_all_study_sessions(user_email)
            return study_sessions
        except Exception as e:
            raise Exception(f"Error fetching study sessions: {e}")

    async def create_study_session(self, user_email: str, discipline_id: int, subject: str, start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        print(f"Creating new study session")
        try:
            # Criar a sessão de estudo usando o dispatcher e obter o session_id
            session_id = self.dispatcher.create_study_session(
                user_email, discipline_id, subject, start_time, end_time
            )

            # Após criar a sessão de estudo, criar um plano de estudo vazio
            empty_plan = {
                "id_sessao": str(session_id),
                "disciplina_id": str(discipline_id),
                "disciplina": subject,
                "descricao": "",  # Descrição vazia por enquanto
                "objetivo_sessao": "",
                "plano_execucao": [],
                "duracao_total": "",
                "progresso_total": 0,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),  # Novo
                "feedback_geral": {}  # Novo
            }

            # Usar o PlanController para criar o plano vazio
            plan_result = await self.plan_controller.create_study_plan(empty_plan)

            if not plan_result:
                print(f"Failed to create empty study plan for session {session_id}")

            return {"session_id": session_id}
        except Exception as e:
            raise Exception(f"Error creating study session: {e}")

    def update_study_session(self, session_id: int, session_data: dict):
        try:
            # Usar o dispatcher para atualizar uma sessão de estudo existente com base no ID do estudante
            updated_session = self.dispatcher.update_study_session(session_id, session_data)
            return updated_session
        except Exception as e:
            raise Exception(f"Error updating study session: {e}")

    def delete_study_session(self, session_id: int):
        try:
            # Usar o dispatcher para deletar uma sessão de estudo existente com base no ID do estudante
            self.dispatcher.delete_study_session(session_id)
        except Exception as e:
            raise Exception(f"Error deleting study session: {e}")

    def get_study_session_from_discipline(self, discipline_id: int, user_email: str):
        try:
            # Usar o dispatcher para buscar as sessões de estudo com base no ID da disciplina e no usuário
            study_sessions = self.dispatcher.get_study_session_from_discipline_id(discipline_id, user_email)
            return study_sessions
        except Exception as e:
            raise Exception(f"Error fetching study session from discipline: {e}")

    def get_study_session_by_id(self, session_id: int, user_email: str):
        try:
            study_session = self.dispatcher.get_study_session_by_id(session_id, user_email)
            return study_session
        except Exception as e:
            raise Exception(f"Error fetching study session by ID: {e}")

    def get_study_session_from_discipline(self, discipline_id: int, user_email: str):
        """
        Busca sessões de estudo para uma disciplina específica.
        """
        try:
            study_sessions = self.dispatcher.get_study_session_from_discipline_id(discipline_id, user_email)
            return study_sessions
        except Exception as e:
            raise Exception(f"Erro ao buscar sessões de estudo para a disciplina: {e}")
