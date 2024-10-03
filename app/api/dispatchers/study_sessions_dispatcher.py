# dispatchers/study_sessions_dispatcher.py

from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_usuarios, tabela_sessoes_estudo
from datetime import datetime, timezone
from api.controllers.auth import hash_password, verify_password

class StudySessionsDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    # Exemplo de método que pode ser adicionado para queries específicas
    def get_study_sessions_for_user(self, user_id: int):
        try:
            result = self.database_manager.session.query(tabela_sessoes_estudo).filter_by(IdUsuario=user_id).all()
            return result
        except Exception as e:
            raise Exception(f"Error retrieving study sessions for user {user_id}: {e}")
