# dispatchers/study_sessions_dispatcher.py

from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_eventos_calendario, tabela_sessoes_estudo, tabela_cronograma, tabela_usuarios, tabela_estudantes
from datetime import datetime, timedelta
from api.controllers.auth import hash_password, verify_password
from sqlalchemy import text
import json


class StudySessionsDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.session = database_manager.session

    def get_student_id_by_email(self, user_email: str):
        try:
            # Primeiro, buscamos o IdUsuario na tabela Usuarios com base no email
            user_query = text('SELECT "IdUsuario" FROM "Usuarios" WHERE "Email" = :email')
            user = self.session.execute(user_query, {'email': user_email}).fetchone()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")

            user_id = user[0]
            print(f"User ID: {user_id}")
            # Agora, buscamos o IdEstudante na tabela Estudantes com base no IdUsuario
            student_query = text('SELECT "IdEstudante" FROM "Estudantes" WHERE "IdUsuario" = :user_id')
            student = self.session.execute(student_query, {'user_id': user_id}).fetchone()

            if student is None:
                raise HTTPException(status_code=404, detail="Student not found")

            return student[0]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching student ID: {e}")

    def get_all_study_sessions(self, user_email: str):
        try:
            # Busca o ID do estudante pelo e-mail
            student_id = self.get_student_id_by_email(user_email)

            # Busca todas as sessões de estudo associadas ao ID do estudante
            study_sessions = self.session.query(tabela_sessoes_estudo).filter_by(IdEstudante=student_id).all()
            return study_sessions
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching study sessions: {e}")

    def create_study_session(self, user_email: str, discipline_name: str):
        try:
            # Busca o ID do estudante pelo e-mail
            student_id = self.get_student_id_by_email(user_email)

            # Cria uma nova sessão de estudo com base no ID do estudante
            new_session = tabela_sessoes_estudo.insert().values(
                NomeDisciplina=discipline_name,
                IdEstudante=student_id,
                CriadoEm=datetime.now(),
                AtualizadoEm=datetime.now()
            )
            self.session.execute(new_session)
            self.session.commit()
            return new_session
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating study session: {e}")

    def update_study_session(self, session_id: int, session_data: dict):
        try:
            # Busca a sessão de estudo existente
            study_session = self.session.query(tabela_sessoes_estudo).filter_by(IdSessao=session_id).first()

            if not study_session:
                raise HTTPException(status_code=404, detail="Study session not found")

            # Atualiza os campos fornecidos
            for key, value in session_data.items():
                setattr(study_session, key, value)

            study_session.AtualizadoEm = datetime.now()
            self.session.commit()
            return study_session
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating study session: {e}")

    def delete_study_session(self, session_id: int):
        try:
            # Busca a sessão de estudo existente
            study_session = self.session.query(tabela_sessoes_estudo).filter_by(IdSessao=session_id).first()

            if not study_session:
                raise HTTPException(status_code=404, detail="Study session not found")

            # Deleta a sessão de estudo
            self.session.delete(study_session)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error deleting study session: {e}")

    def get_study_session_from_discipline(self, discipline_name: str, user_email: str):
        try:
            # 1. Obter o IdEstudante com base no e-mail do usuário
            student_id = self.database_manager.get_student_by_user_email(user_email)

            # 2. Obter o IdCurso com base no nome da disciplina
            course_id = self.database_manager.get_course_by_name(discipline_name)

            # 3. Obter as sessões de estudo para o estudante e o curso
            study_sessions = self.database_manager.get_study_sessions_by_course_and_student(course_id, student_id)

            return study_sessions
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao buscar sessões de estudo para a disciplina '{discipline_name}': {e}")