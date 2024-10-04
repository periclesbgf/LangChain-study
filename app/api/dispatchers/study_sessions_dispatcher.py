# dispatchers/study_sessions_dispatcher.py

from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_eventos_calendario, tabela_sessoes_estudo, tabela_cronograma, tabela_usuarios
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

    def create_discipline_from_pdf(self, discipline_json: dict, user_email: str):
        try:
            # 1. Extrair dados do JSON
            print("Extraindo dados do JSON...")
            curso_data = discipline_json.get('curso', {})
            cronograma_data = discipline_json.get('cronograma', [])
            print(f"Curso: {curso_data}")
            print(f"Cronograma: {cronograma_data}")

            # 2. Buscar o ID do usuário com base no e-mail
            print(f"Buscando ID do usuário com e-mail: {user_email}")
            user_query = self.session.execute(
                tabela_usuarios.select().where(tabela_usuarios.c.Email == user_email)
            )
            user = user_query.fetchone()

            if user is None:
                raise Exception(f"Usuário com e-mail {user_email} não encontrado.")
            
            # Certifique-se de que 'user' é uma tupla e obtenha o ID pelo índice
            user_id = user[0]  # 'IdUsuario' é provavelmente o primeiro campo retornado
            print(f"ID do usuário: {user_id}")

            # 3. Inserir a disciplina na tabela Cursos
            print("Inserindo nova disciplina no banco de dados...")
            new_course = tabela_cursos.insert().values(
                NomeCurso=curso_data.get('nome', 'Sem Nome'),
                Ementa=curso_data.get('ementa', 'Sem Ementa'),
                Objetivos=json.dumps(curso_data.get('objetivos', [])),
                CriadoEm=datetime.now()
            )
            result = self.session.execute(new_course)
            self.session.commit()  # Commit para garantir que o curso é persistido e podemos pegar seu ID
            course_id = result.inserted_primary_key[0]
            print(f"Curso criado com ID: {course_id}")

            # 4. Inserir o cronograma para o curso
            print(f"Inserindo cronograma para o curso ID: {course_id}")
            new_cronograma = tabela_cronograma.insert().values(
                IdCurso=course_id,
                NomeCronograma=f"Cronograma de {curso_data.get('nome', 'Sem Nome')}"
            )
            result_cronograma = self.session.execute(new_cronograma)
            self.session.commit()  # Commit para garantir que o cronograma é persistido e podemos pegar seu ID
            cronograma_id = result_cronograma.inserted_primary_key[0]
            print(f"Cronograma criado com ID: {cronograma_id}")

            # 5. Criar as sessões de estudo e adicionar eventos no calendário
            for encontro in cronograma_data:
                # Parse da data do encontro
                session_date = datetime.strptime(encontro['data'], "%d/%m/%Y")
                print(f"Inserindo sessão {encontro['numero_encontro']} para o cronograma {cronograma_id}...")

                # Inserir o encontro na tabela Encontros
                new_session = tabela_encontros.insert().values(
                    IdCronograma=cronograma_id,  # Agora estamos usando o cronograma_id correto
                    NumeroEncontro=encontro['numero_encontro'],
                    DataEncontro=session_date,
                    Conteudo=encontro['conteudo'].strip(),
                    Estrategia=encontro.get('estrategia', "Não especificada"),
                    Avaliacao=encontro.get('avaliacao')  # Campo opcional de avaliação
                )
                self.session.execute(new_session)

                # Adicionar evento no calendário
                new_event = tabela_eventos_calendario.insert().values(
                    GoogleEventId=f"event-{course_id}-{encontro['numero_encontro']}",
                    Titulo=f"Encontro {encontro['numero_encontro']} - {curso_data.get('nome', 'Sem Nome')}",
                    Descricao=encontro['conteudo'].strip(),
                    Inicio=session_date,
                    Fim=session_date + timedelta(hours=2),
                    Local="Sala de Aula Física",
                    CriadoPor=user_id  # Usar o ID do usuário em vez do e-mail
                )
                self.session.execute(new_event)

            # 6. Commit para salvar os encontros e eventos
            self.session.commit()
            print(f"Curso '{curso_data.get('nome', 'Sem Nome')}' e sessões criadas com sucesso.")
            
        except Exception as e:
            print(f"Erro detectado: {e}")
            self.session.rollback()
            raise HTTPException(status_code=500, detail=f"Erro ao criar disciplina e sessões: {e}")
