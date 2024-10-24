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
from fastapi.encoders import jsonable_encoder



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

    def create_study_session(self, user_email: str, course_id: int, subject: str):
        print(f"Dispatcher - Creating new study session")
        print(f"User email: {user_email}, Course ID: {course_id}, Subject: {subject}")
        try:
            # Busca o ID do estudante pelo e-mail
            student_id = self.get_student_id_by_email(user_email)

            # Cria uma nova sessão de estudo e insere no banco de dados
            new_session = tabela_sessoes_estudo.insert().values(
                IdEstudante=student_id,
                IdCurso=course_id,  # Usando o ID do curso
                Assunto=subject,
                Inicio=datetime.now(),  # Registrando o início da sessão
                Fim=None,  # Pode ser atualizado depois
                Produtividade=0,  # Opcional, pode ser preenchido mais tarde
                FeedbackDoAluno=None,  # Pode ser adicionado depois
                HistoricoConversa=None  # Inicialmente vazio
            )

            # Executar a inserção no banco de dados
            self.session.execute(new_session)
            self.session.commit()

            return {"message": "Study session created successfully"}

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

    def get_study_session_from_discipline_id(self, discipline_id: int, user_email: str):
        try:
            student_id = self.get_student_id_by_email(user_email)
            course_id = discipline_id  # Já temos o ID da disciplina

            # Buscar as sessões de estudo do banco de dados
            study_sessions = self.database_manager.get_study_sessions_by_course_and_student(course_id, student_id)

            # Converter as sessões para dicionários
            session_list = [
                {
                    "IdSessao": session.IdSessao,
                    "IdEstudante": session.IdEstudante,
                    "IdCurso": session.IdCurso,
                    "Assunto": session.Assunto,
                    "Inicio": session.Inicio.isoformat() if session.Inicio else None,
                    "Fim": session.Fim.isoformat() if session.Fim else None,
                    "Produtividade": session.Produtividade,
                    "FeedbackDoAluno": session.FeedbackDoAluno,
                }
                for session in study_sessions
            ]

            # Retornar um dicionário JSON serializável
            return jsonable_encoder({"study_sessions": session_list})

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao buscar sessões de estudo para a disciplina com ID '{discipline_id}': {e}")

    def get_study_session_by_id(self, session_id: int, user_email: str):
        try:
            student_id = self.get_student_id_by_email(user_email)
            study_session = self.database_manager.get_study_session_by_id_and_student(session_id, student_id)
            if not study_session:
                return None
            # Converter a sessão para dicionário
            session_data = {
                "IdSessao": study_session.IdSessao,
                "IdEstudante": study_session.IdEstudante,
                "IdCurso": study_session.IdCurso,
                "Assunto": study_session.Assunto,
                "Inicio": study_session.Inicio.isoformat() if study_session.Inicio else None,
                "Fim": study_session.Fim.isoformat() if study_session.Fim else None,
                "Produtividade": study_session.Produtividade,
                "FeedbackDoAluno": study_session.FeedbackDoAluno,
                "HistoricoConversa": study_session.HistoricoConversa
            }
            return session_data
        except Exception as e:
            raise Exception(f"Error fetching study session by ID: {e}")