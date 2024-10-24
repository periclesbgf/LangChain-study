from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sql_test.sql_test_create import tabela_usuarios, tabela_educadores, tabela_cursos, tabela_sessoes_estudo, tabela_estudante_curso, tabela_eventos_calendario, tabela_estudantes, tabela_perfil_aprendizado_aluno
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from sqlalchemy.sql import text
import json
from sqlalchemy.sql import select
from datetime import datetime

# Carregar variáveis de ambiente
load_dotenv()

# Conectar ao banco de dados PostgreSQL
user = os.getenv("POSTGRES_USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")

# Configuração da conexão do SQLAlchemy com o PostgreSQL
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
metadata = MetaData()

# Configuração da sessão para interagir com o banco de dados
Session = sessionmaker(bind=engine)
session = Session()

class DatabaseManager:
    def __init__(self, session, metadata):
        self.session = session
        self.metadata = metadata

    def criar_tabela(self, nome_tabela, colunas):
        try:
            tabela = Table(nome_tabela, self.metadata, *colunas)
            self.metadata.create_all(engine)
            print(f"Tabela {nome_tabela} criada com sucesso.")
            return tabela
        except Exception as e:
            print(f"Erro ao criar tabela {nome_tabela}: {e}")
            return None

    def inserir_dado(self, tabela, dados):
        try:
            result = self.session.execute(tabela.insert().returning(tabela.c.IdUsuario).values(dados))
            self.session.commit()
            print(f"Dado inserido com sucesso na tabela {tabela.name}")
            return result.fetchone()
        except IntegrityError as e:
            self.session.rollback()
            raise HTTPException(status_code=400, detail="Duplicated entry.")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao inserir dado na tabela {tabela.name}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    def inserir_dado_retorna_id(self, tabela, dados, id_column_name):
        """
        Método para inserir um novo registro em uma tabela e retornar o ID recém-criado.
        :param tabela: A tabela onde o dado será inserido.
        :param dados: Dicionário com os dados a serem inseridos.
        :param id_column_name: Nome da coluna do ID que será retornado.
        :return: O ID do registro recém-criado.
        """
        try:
            result = self.session.execute(
                tabela.insert().returning(getattr(tabela.c, id_column_name)).values(dados)
            )
            self.session.commit()
            inserted_id = result.fetchone()[0]  # O ID recém-inserido é retornado
            print(f"Inserted record with ID: {inserted_id}")
            return inserted_id
        except IntegrityError as e:
            self.session.rollback()
            print(f"IntegrityError during insertion: {e}")
            raise HTTPException(status_code=400, detail="Duplicated entry.")
        except Exception as e:
            self.session.rollback()
            print(f"Error during insertion: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    def deletar_dado(self, tabela, condicao):
        try:
            self.session.execute(tabela.delete().where(condicao))
            self.session.commit()
            print(f"Dado deletado com sucesso da tabela {tabela.name}")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao deletar dado na tabela {tabela.name}: {e}")

    def atualizar_dado(self, tabela, condicao, novos_dados):
        try:
            self.session.execute(tabela.update().where(condicao).values(novos_dados))
            self.session.commit()
            print(f"Dado atualizado com sucesso na tabela {tabela.name}")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao atualizar dado na tabela {tabela.name}: {e}")

    def selecionar_dados(self, tabela, condicao=None):
        try:
            if condicao:
                result = self.session.execute(tabela.select().where(condicao)).fetchall()
            else:
                result = self.session.execute(tabela.select()).fetchall()
            return result
        except Exception as e:
            print(f"Erro ao selecionar dados da tabela {tabela.name}: {e}")
            return None

    def get_user_by_email(self, email: str):
        try:
            user = self.session.query(tabela_usuarios).filter(tabela_usuarios.c.Email == email).first()
            print(f"Usuário encontrado: {user}")
            return user
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao buscar usuário por email: {e}")
            return None

    def get_user_id_by_email(self, user_email: str):
        """
        Função para obter o IdUsuario de um usuário com base no e-mail.
        """
        try:
            user_query = text('SELECT "IdUsuario" FROM "Usuarios" WHERE "Email" = :email')
            user = self.session.execute(user_query, {'email': user_email}).fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return user[0]  # Retorna o IdUsuario
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching user ID: {str(e)}")

    def get_educator_id_by_email(self, user_email: str):
        """
        Função para buscar o ID do educador com base no e-mail do usuário.
        """
        try:
            user_query = text('SELECT "IdUsuario" FROM "Usuarios" WHERE "Email" = :email')
            user = self.session.execute(user_query, {'email': user_email}).fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            user_id = user[0]

            educator_query = text('SELECT "IdEducador" FROM "Educadores" WHERE "IdUsuario" = :user_id')
            educator = self.session.execute(educator_query, {'user_id': user_id}).fetchone()
            if not educator:
                raise HTTPException(status_code=404, detail="Educator not found")
            return educator[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching educator ID: {str(e)}")

    def get_all_educator_names(self):
        """
        Função para buscar todos os nomes dos educadores.
        """
        try:
            query = select(tabela_usuarios.c.Nome).select_from(
                tabela_usuarios.join(tabela_educadores, tabela_usuarios.c.IdUsuario == tabela_educadores.c.IdUsuario)
            )
            result = self.session.execute(query).fetchall()
            educator_names = [row[0] for row in result]
            return educator_names
        except Exception as e:
            print(f"Erro ao buscar os nomes dos educadores: {e}")
            raise HTTPException(status_code=500, detail="Error fetching educator names.")

    def get_all_events_by_user(self, tabela_eventos, user_id: int):
        """
        Função para buscar todos os eventos de um usuário específico na tabela de eventos,
        sem filtro de curso.
        """
        try:
            # Criar a consulta para selecionar eventos criados pelo usuário específico
            query = select(tabela_eventos).where(tabela_eventos.c.CriadoPor == user_id)

            result = self.session.execute(query).fetchall()
            return result
        except Exception as e:
            print(f"Erro ao selecionar eventos do usuário {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao selecionar eventos: {e}")



    def get_student_by_user_email(self, user_email: str):
        try:
            user = self.get_user_by_email(user_email)
            if not user:
                raise HTTPException(status_code=404, detail="Usuário não encontrado.")
            query = select(tabela_estudantes.c.IdEstudante).where(tabela_estudantes.c.IdUsuario == user.IdUsuario)
            student = self.session.execute(query).fetchone()
            if not student:
                return None
            return student[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail="Erro ao buscar estudante.")


    def get_course_by_name(self, discipline_name: str):
        try:
            query = select(tabela_cursos.c.IdCurso).where(tabela_cursos.c.NomeCurso == discipline_name)
            course = self.session.execute(query).fetchone()
            if not course:
                return None
            return course[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao buscar curso para a disciplina '{discipline_name}'.")

    def get_user_name_by_email(self, user_email: str) -> str:
        """
        Obtém o nome do usuário com base no e-mail.
        :param user_email: E-mail do usuário.
        :return: Nome do usuário.
        """
        try:
            query = select(tabela_usuarios.c.Nome).where(tabela_usuarios.c.Email == user_email)
            result = self.session.execute(query).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Usuário não encontrado.")
            return result[0]
        except Exception as e:
            print(f"Erro ao buscar o nome do usuário: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar o nome do usuário.")

    def get_study_sessions_by_course_and_student(self, course_id: int, student_id: int):
        try:
            query = select(tabela_sessoes_estudo).where(
                (tabela_sessoes_estudo.c.IdCurso == course_id) &
                (tabela_sessoes_estudo.c.IdEstudante == student_id)
            )
            sessions = self.session.execute(query).fetchall()

            # Certifique-se de que as sessões sejam retornadas como dicionários
            return sessions
        except Exception as e:
            print(f"Erro ao buscar sessões de estudo: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar sessões de estudo.")

    def get_learning_profiles_by_user_id(self, user_id: int):
        """
        Função para obter todos os perfis de aprendizado do aluno com base no IdUsuario.
        """
        try:
            query = select(tabela_perfil_aprendizado_aluno).where(tabela_perfil_aprendizado_aluno.c.IdUsuario == user_id)
            profiles = self.session.execute(query).fetchall()
            if not profiles:
                return []
            return profiles
        except Exception as e:
            print(f"Erro ao buscar perfis de aprendizado para o usuário {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar perfis de aprendizado.")

    def get_course_by_id(self, course_id: int):
        """
        Função para buscar curso pelo IdCurso.
        """
        try:
            query = select(tabela_cursos).where(tabela_cursos.c.IdCurso == course_id)
            course = self.session.execute(query).fetchone()
            if not course:
                raise HTTPException(status_code=404, detail="Curso não encontrado.")
            return course
        except Exception as e:
            print(f"Erro ao buscar curso {course_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar curso.")

    def get_educator_by_name(self, nome_educador: str):
        """
        Função para buscar o educador pelo nome na tabela Usuarios.
        """
        try:
            query = select(tabela_usuarios).where(tabela_usuarios.c.Nome == nome_educador)
            educator = self.session.execute(query).fetchone()
            if not educator:
                print(f"Educador {nome_educador} não encontrado.")
            return educator
        except Exception as e:
            print(f"Erro ao buscar educador {nome_educador}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao buscar educador {nome_educador}: {e}")

    def get_courses_by_student_id(self, student_id: int):
        """
        Função para buscar as disciplinas de um estudante com base no IdEstudante.
        """
        try:
            query = select(tabela_cursos).select_from(
                tabela_cursos.join(tabela_estudante_curso, tabela_cursos.c.IdCurso == tabela_estudante_curso.c.IdCurso)
            ).where(tabela_estudante_curso.c.IdEstudante == student_id)

            courses = self.session.execute(query).fetchall()
            return courses
        except Exception as e:
            print(f"Erro ao buscar cursos para o estudante {student_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar cursos.")

    def associar_aluno_curso(self, estudante_id: int, curso_id: int):
        """
        Método personalizado para associar um aluno a um curso na tabela EstudanteCurso.
        :param estudante_id: ID do aluno.
        :param curso_id: ID do curso.
        """
        try:
            # Inserir a associação aluno-curso na tabela EstudanteCurso
            nova_associacao = tabela_estudante_curso.insert().values(
                IdEstudante=estudante_id,
                IdCurso=curso_id,
                CriadoEm=datetime.now()
            )
            self.session.execute(nova_associacao)
            self.session.commit()
            print(f"Aluno {estudante_id} associado ao curso {curso_id} com sucesso.")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao associar aluno {estudante_id} ao curso {curso_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao associar aluno ao curso: {e}")

    def inserir_dado_evento(self, tabela_eventos, dados_evento: dict):
        """
        Método para inserir um novo evento na tabela de eventos e retornar o ID recém-criado.
        :param tabela_eventos: A tabela de eventos onde o dado será inserido.
        :param dados_evento: Dicionário com os dados do evento a serem inseridos.
        :return: O ID do evento recém-criado.
        """
        try:
            result = self.session.execute(
                tabela_eventos.insert().returning(tabela_eventos.c.IdEvento).values(dados_evento)
            )
            self.session.commit()
            inserted_id = result.fetchone()[0]  # O ID recém-inserido é retornado
            print(f"Evento inserido com ID: {inserted_id}")
            return inserted_id
        except IntegrityError as e:
            self.session.rollback()
            print(f"IntegrityError durante inserção de evento: {e}")
            raise HTTPException(status_code=400, detail="Evento duplicado ou dados inválidos.")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao inserir evento: {e}")
            raise HTTPException(status_code=500, detail="Erro interno do servidor ao inserir evento.")


    def get_calendar_events_by_user_id(self, user_id: int):
        """
        Obtém todos os eventos de calendário para um usuário específico.
        :param user_id: ID do usuário.
        :return: Lista de eventos.
        """
        try:
            query = select(tabela_eventos_calendario).where(tabela_eventos_calendario.c.CriadoPor == user_id)
            result = self.session.execute(query).fetchall()
            events = [dict(event) for event in result]
            return events
        except Exception as e:
            print(f"Erro ao buscar eventos para o usuário {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar eventos do calendário.")

    def create_calendar_event(self, event_data: dict):
        """
        Cria um novo evento de calendário.
        :param event_data: Dicionário com os dados do evento.
        :return: ID do evento recém-criado.
        """
        try:
            result = self.session.execute(
                tabela_eventos_calendario.insert().returning(tabela_eventos_calendario.c.IdEvento).values(event_data)
            )
            self.session.commit()
            inserted_id = result.fetchone()[0]
            print(f"Evento inserido com ID: {inserted_id}")
            return inserted_id
        except IntegrityError as e:
            self.session.rollback()
            print(f"IntegrityError ao criar evento: {e}")
            raise HTTPException(status_code=400, detail="Erro de integridade ao criar evento.")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao criar evento de calendário: {e}")
            raise HTTPException(status_code=500, detail="Erro ao criar evento de calendário.")

    def update_calendar_event(self, event_id: int, updated_data: dict):
        """
        Atualiza um evento de calendário existente.
        :param event_id: ID do evento a ser atualizado.
        :param updated_data: Dicionário com os dados atualizados do evento.
        """
        try:
            result = self.session.execute(
                tabela_eventos_calendario.update()
                .where(tabela_eventos_calendario.c.IdEvento == event_id)
                .values(updated_data)
            )
            self.session.commit()
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Evento não encontrado.")
            print(f"Evento com ID {event_id} atualizado com sucesso.")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao atualizar evento de calendário {event_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao atualizar evento de calendário.")

    def delete_calendar_event(self, event_id: int):
        """
        Deleta um evento de calendário.
        :param event_id: ID do evento a ser deletado.
        """
        try:
            result = self.session.execute(
                tabela_eventos_calendario.delete().where(tabela_eventos_calendario.c.IdEvento == event_id)
            )
            self.session.commit()
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Evento não encontrado.")
            print(f"Evento com ID {event_id} deletado com sucesso.")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao deletar evento de calendário {event_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao deletar evento de calendário.")

    def get_study_session_by_id_and_student(self, session_id: int, student_id: int):
        try:
            query = select(tabela_sessoes_estudo).where(
                (tabela_sessoes_estudo.c.IdSessao == session_id) &
                (tabela_sessoes_estudo.c.IdEstudante == student_id)
            )
            session = self.session.execute(query).fetchone()
            return session
        except Exception as e:
            print(f"Erro ao buscar sessão de estudo: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar sessão de estudo.")

    def get_course_by_id(self, course_id: int):
        """
        Função para buscar curso pelo IdCurso.
        """
        try:
            query = select(tabela_cursos).where(tabela_cursos.c.IdCurso == course_id)
            course = self.session.execute(query).fetchone()
            if not course:
                raise HTTPException(status_code=404, detail="Curso não encontrado.")
            return course
        except Exception as e:
            print(f"Erro ao buscar curso {course_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar curso.")