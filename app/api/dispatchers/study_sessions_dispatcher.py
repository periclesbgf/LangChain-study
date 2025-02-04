# dispatchers/study_sessions_dispatcher.py

import select
from sqlalchemy.exc import IntegrityError
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_eventos_calendario, tabela_sessoes_estudo, tabela_cronograma, tabela_usuarios, tabela_estudantes
from datetime import datetime, timezone
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

    def create_study_session(self, user_email: str, course_id: int, subject: str, start_time: datetime = None, end_time: datetime = None):
        print(f"Dispatcher - Creating new study session")
        print(f"User email: {user_email}, Course ID: {course_id}, Subject: {subject}")
        try:
            # Busca o ID do estudante pelo e-mail
            student_id = self.get_student_id_by_email(user_email)

            # Usa a data e hora atuais se start_time for None
            if start_time is None:
                start_time = datetime.now()
            if end_time is None:
                end_time = start_time + timedelta(hours=1)  # Duração padrão de 1 hora

            # Cria uma nova sessão de estudo e insere no banco de dados
            new_session = tabela_sessoes_estudo.insert().values(
                IdEstudante=student_id,
                IdCurso=course_id,
                Assunto=subject,
                Inicio=start_time,
                Fim=end_time,
                Produtividade=0,
                FeedbackDoAluno=None,
                HistoricoConversa=None
            )

            # Executar a inserção no banco de dados
            result = self.session.execute(new_session)
            self.session.commit()
            session_id = result.inserted_primary_key[0]

            return session_id

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
            return session_list

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

    def get_study_sessions_by_course(self, user_email: str, discipline_id: int):
        """
        Retorna todas as sessões de estudo de uma disciplina específica.
        """
        try:
            # Obter o ID do estudante pelo e-mail
            student_id = self.get_student_id_by_email(user_email)

            # Query para buscar sessões da disciplina específica
            query = (
                select(tabela_sessoes_estudo)
                .where(
                    (tabela_sessoes_estudo.c.IdCurso == discipline_id) &
                    (tabela_sessoes_estudo.c.IdEstudante == student_id)
                )
            )
            result = self.session.execute(query).fetchall()

            return [
                {
                    "IdSessao": row.IdSessao,
                    "IdEstudante": row.IdEstudante,
                    "IdCurso": row.IdCurso,
                    "Assunto": row.Assunto,
                    "Inicio": row.Inicio,
                    "Fim": row.Fim,
                    "Produtividade": row.Produtividade,
                    "FeedbackDoAluno": row.FeedbackDoAluno
                }
                for row in result
            ]
        except Exception as e:
            print(f"Erro ao buscar sessões de estudo: {e}")
            raise Exception("Erro ao buscar sessões de estudo.")
