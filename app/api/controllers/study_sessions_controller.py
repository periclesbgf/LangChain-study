# controllers/study_sessions_controller.py

from chains.chain_setup import DisciplinChain
from database.sql_database_manager import DatabaseManager
from sql_test.sql_test_create import tabela_cursos, tabela_sessoes_estudo, tabela_encontros, tabela_eventos_calendario
from database.search import execute_query
from database.vector_db import DocumentLoader, TextSplitter, Embeddings, QdrantIndex
from audio.text_to_speech import AudioService
from fastapi.logger import logger
from datetime import datetime
import re
from api.dispatchers.study_sessions_dispatcher import StudySessionsDispatcher
import json

from utils import OPENAI_API_KEY, CODE


class StudySessionsController:
    def __init__(self, dispatcher: StudySessionsDispatcher, disciplin_chain: DisciplinChain = None):
        self.dispatcher = dispatcher
        self.disciplin_chain = disciplin_chain

    def get_all_study_sessions(self, user_email: str):
        try:
            # Usar o dispatcher para buscar todas as sessões de estudo pelo ID do estudante
            study_sessions = self.dispatcher.get_all_study_sessions(user_email)
            return study_sessions
        except Exception as e:
            raise Exception(f"Error fetching study sessions: {e}")

    def create_study_session(self, user_email: str, discipline_name: str):
        try:
            # Usar o dispatcher para criar uma nova sessão de estudo com base no ID do estudante
            new_session = self.dispatcher.create_study_session(user_email, discipline_name)
            return new_session
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
