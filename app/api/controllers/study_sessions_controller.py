# controllers/study_sessions_controller.py

from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from sql_test.sql_test_create import tabela_usuarios, tabela_sessoes_estudo
from database.search import execute_query
from database.vector_db import DocumentLoader, TextSplitter, Embeddings, QdrantIndex
from audio.text_to_speech import AudioService
from fastapi.logger import logger
from datetime import datetime


from utils import OPENAI_API_KEY, CODE


class StudySessionsController:
    def __init__(self, session):
        self.session = session

    def get_all_study_sessions(self, user_email: str):
        try:
            # Busca todas as sessões de estudo para o usuário logado
            study_sessions = self.session.query(tabela_sessoes_estudo).filter_by(email=user_email).all()
            return study_sessions
        except Exception as e:
            raise Exception(f"Error fetching study sessions: {e}")

    def create_study_session(self, user_email: str, discipline_name: str):
        try:
            # Lógica para criar uma nova sessão de estudo
            new_session = tabela_sessoes_estudo.insert().values(
                NomeDisciplina=discipline_name,
                EmailUsuario=user_email,
                CriadoEm=datetime.now(),
                AtualizadoEm=datetime.now()
            )
            self.session.execute(new_session)
            self.session.commit()
            return new_session
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error creating study session: {e}")

    def update_study_session(self, session_id: int, session_data: dict):
        try:
            # Lógica para atualizar uma sessão de estudo existente
            study_session = self.session.query(tabela_sessoes_estudo).filter_by(IdSessao=session_id).first()
            if not study_session:
                raise Exception("Study session not found")

            for key, value in session_data.items():
                setattr(study_session, key, value)

            study_session.AtualizadoEm = datetime.now()
            self.session.commit()
            return study_session
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error updating study session: {e}")

    def delete_study_session(self, session_id: int):
        try:
            # Lógica para deletar uma sessão de estudo existente
            study_session = self.session.query(tabela_sessoes_estudo).filter_by(IdSessao=session_id).first()
            if not study_session:
                raise Exception("Study session not found")

            self.session.delete(study_session)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error deleting study session: {e}")
