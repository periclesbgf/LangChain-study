import base64
from typing import Any, Dict, Optional
from sqlalchemy import and_, create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey, select, join, update
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sql_interface.sql_tables import (
    tabela_usuarios,
    tabela_educadores,
    tabela_cursos,
    tabela_encontros,
    tabela_cronograma,
    tabela_sessoes_estudo,
    tabela_estudante_curso,
    tabela_eventos_calendario,
    tabela_estudantes,
    tabela_perfil_aprendizado_aluno,
    tabela_support_request_images,
    tabela_support_requests
    )
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from fastapi import HTTPException
from sqlalchemy.sql import text, select
import json
from datetime import datetime, time, timezone
from logg import logger

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

    @staticmethod
    def get_db():
        """Dependency function that provides a new SQLAlchemy session."""
        db = Session()  # create a new session using your sessionmaker (Session is your sessionmaker)
        try:
            yield db
        finally:
            db.close()

    def criar_tabela(self, nome_tabela, colunas):
        try:
            tabela = Table(nome_tabela, self.metadata, *colunas)
            self.metadata.create_all(engine)
            logger.info(f"Tabela {nome_tabela} criada com sucesso.")
            return tabela
        except Exception as e:
            logger.error(f"Erro ao criar tabela {nome_tabela}: {e}")
            return None

    def inserir_dado(self, tabela, dados):
        try:
            result = self.session.execute(tabela.insert().returning(tabela.c.IdUsuario).values(dados))
            self.session.commit()
            logger.info(f"Dado inserido com sucesso na tabela {tabela.name}")
            return result.fetchone()
        except IntegrityError as e:
            self.session.rollback()
            raise HTTPException(status_code=400, detail="Duplicated entry.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao inserir dado na tabela {tabela.name}: {e}")
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
            logger.info(f"Inserted record with ID: {inserted_id}")
            return inserted_id
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"IntegrityError during insertion: {e}")
            raise HTTPException(status_code=409, detail="Conflict")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error during insertion: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    def deletar_dado(self, tabela, condicao):
        try:
            self.session.execute(tabela.delete().where(condicao))
            self.session.commit()
            logger.info(f"Dado deletado com sucesso da tabela {tabela.name}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao deletar dado na tabela {tabela.name}: {e}")

    def atualizar_dado(self, tabela, condicao, novos_dados):
        try:
            self.session.execute(tabela.update().where(condicao).values(novos_dados))
            self.session.commit()
            logger.info(f"Dado atualizado com sucesso na tabela {tabela.name}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao atualizar dado na tabela {tabela.name}: {e}")

    def selecionar_dados(self, tabela, condicao=None):
        try:
            if condicao:
                result = self.session.execute(tabela.select().where(condicao)).fetchall()
            else:
                result = self.session.execute(tabela.select()).fetchall()
            return result
        except Exception as e:
            logger.error(f"Erro ao selecionar dados da tabela {tabela.name}: {e}")
            return None

    def get_user_by_email(self, email: str):
        try:
            user = self.session.query(tabela_usuarios).filter(tabela_usuarios.c.Email == email).first()
            return user
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao buscar usuário por email: {e}")
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
            logger.error(f"Erro ao buscar os nomes dos educadores: {e}")
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
            logger.error(f"Erro ao selecionar eventos do usuário {user_id}: {e}")
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
            logger.error(f"Erro ao buscar o nome do usuário: {e}")
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
            logger.error(f"Erro ao buscar sessões de estudo: {e}")
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
            logger.error(f"Erro ao buscar perfis de aprendizado para o usuário {user_id}: {e}")
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
            logger.error(f"Erro ao buscar curso {course_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar curso.")

    def get_educator_by_name(self, nome_educador: str):
        """
        Função para buscar o educador pelo nome na tabela Usuarios.
        """
        try:
            query = select(tabela_usuarios).where(tabela_usuarios.c.Nome == nome_educador)
            educator = self.session.execute(query).fetchone()
            if not educator:
                logger.info(f"Educador {nome_educador} não encontrado.")
            return educator
        except Exception as e:
            logger.error(f"Erro ao buscar educador {nome_educador}: {e}")
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
            logger.error(f"Erro ao buscar cursos para o estudante {student_id}: {e}")
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
            logger.info(f"Aluno {estudante_id} associado ao curso {curso_id} com sucesso.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao associar aluno {estudante_id} ao curso {curso_id}: {e}")
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
            logger.info(f"Evento inserido com ID: {inserted_id}")
            return inserted_id
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"IntegrityError durante inserção de evento: {e}")
            raise HTTPException(status_code=400, detail="Evento duplicado ou dados inválidos.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao inserir evento: {e}")
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
            logger.error(f"Erro ao buscar eventos para o usuário {user_id}: {e}")
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
            logger.info(f"Evento inserido com ID: {inserted_id}")
            return inserted_id
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"IntegrityError ao criar evento: {e}")
            raise HTTPException(status_code=400, detail="Erro de integridade ao criar evento.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao criar evento de calendário: {e}")
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
            logger.info(f"Evento com ID {event_id} atualizado com sucesso.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao atualizar evento de calendário {event_id}: {e}")
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
            logger.info(f"Evento com ID {event_id} deletado com sucesso.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao deletar evento de calendário {event_id}: {e}")
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
            logger.error(f"Erro ao buscar sessão de estudo: {e}")
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
            logger.error(f"Erro ao buscar curso {course_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar curso.")

    def update_session_start_time(self, session_id: int, start_datetime: str, end_datetime: str):
        """
        Atualiza o horário de início e fim da sessão no banco de dados SQL.
        """
        try:
            # Atualize o horário de início e fim da sessão
            self.session.execute(
                tabela_sessoes_estudo.update()
                .where(tabela_sessoes_estudo.c.IdSessao == session_id)
                .values(Inicio=start_datetime, Fim=end_datetime)
            )
            self.session.commit()
            logger.info(f"Horários da sessão {session_id} atualizados para Início: {start_datetime}, Fim: {end_datetime}.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao atualizar os horários da sessão no SQL: {e}")
            raise HTTPException(status_code=500, detail="Erro ao atualizar os horários da sessão.")

    def get_encontro_horarios(self, session_id: str) -> dict:
        """
        Busca os horários do encontro relacionado à sessão de estudo usando SQLAlchemy
        """
        logger.info(f"\n[DEBUG] Iniciando get_encontro_horarios para session_id: {session_id}")
        
        if not session_id:
            logger.info("[ERROR] Session ID não fornecido")
            raise HTTPException(
                status_code=400,
                detail="ID da sessão é obrigatório"
            )

        try:
            # Verificar se a sessão existe e obter seu assunto
            logger.info(f"[DEBUG] Verificando existência da sessão {session_id}")
            session_exists = select(
                tabela_sessoes_estudo.c.IdSessao,
                tabela_sessoes_estudo.c.Assunto,
                tabela_sessoes_estudo.c.IdCurso,
                tabela_sessoes_estudo.c.PreferenciaHorario
            ).where(
                tabela_sessoes_estudo.c.IdSessao == session_id
            )
            session_result = self.session.execute(session_exists).first()
            
            logger.info(f"[DEBUG] Resultado da verificação da sessão: {session_result}")
            
            if not session_result:
                logger.info(f"[ERROR] Sessão {session_id} não encontrada no banco")
                raise HTTPException(
                    status_code=404,
                    detail=f"Sessão {session_id} não encontrada"
                )

            # Construir a query usando SQLAlchemy para encontrar o encontro específico
            logger.info("[DEBUG] Construindo query para buscar horários")
            query = select(
                tabela_encontros.c.HorarioInicio,
                tabela_encontros.c.HorarioFim,
                tabela_encontros.c.DataEncontro,
                tabela_sessoes_estudo.c.PreferenciaHorario
            ).select_from(
                join(
                    tabela_encontros,
                    tabela_cronograma,
                    tabela_encontros.c.IdCronograma == tabela_cronograma.c.IdCronograma
                ).join(
                    tabela_sessoes_estudo,
                    and_(
                        tabela_cronograma.c.IdCurso == tabela_sessoes_estudo.c.IdCurso,
                        tabela_encontros.c.Conteudo == tabela_sessoes_estudo.c.Assunto
                    )
                )
            ).where(
                tabela_sessoes_estudo.c.IdSessao == session_id
            )

            logger.info(f"[DEBUG] Query SQL gerada: {query}")
            
            result = self.session.execute(query).first()
            logger.info(f"[DEBUG] Resultado da query: {result}")
            
            if not result:
                logger.info(f"[ERROR] Nenhum resultado encontrado para a sessão {session_id}")
                return None

            # Log dos dados retornados
            horarios = {
                "horario_inicio": result.HorarioInicio,
                "horario_fim": result.HorarioFim,
                "data_encontro": result.DataEncontro,
                "preferencia_horario": result.PreferenciaHorario
            }
            
            logger.info("[DEBUG] Dados formatados do encontro:")
            logger.info(f"  Horário Início: {horarios['horario_inicio']}")
            logger.info(f"  Horário Fim: {horarios['horario_fim']}")
            logger.info(f"  Data Encontro: {horarios['data_encontro']}")
            logger.info(f"  Preferência Horário: {horarios['preferencia_horario']}")
            
            return horarios
                
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.info(f"[ERROR] Erro inesperado ao buscar horários do encontro: {str(e)}")
            logger.info(f"[ERROR] Tipo do erro: {type(e)}")
            import traceback
            logger.info(f"[ERROR] Traceback completo: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao buscar horários do encontro: {str(e)}"
            )

    def update_session_times(self, session_id: str, start_time: datetime, end_time: datetime) -> bool:
        """
        Atualiza os horários de início e fim de uma sessão de estudo
        
        Args:
            session_id: ID da sessão de estudo
            start_time: Novo horário de início (datetime)
            end_time: Novo horário de fim (datetime)
            
        Returns:
            bool: True se a atualização foi bem sucedida, False caso contrário
        """
        try:
            if not session_id:
                raise HTTPException(
                    status_code=400,
                    detail="ID da sessão é obrigatório"
                )

            if not start_time or not end_time:
                raise HTTPException(
                    status_code=400,
                    detail="Horários de início e fim são obrigatórios"
                )

            if end_time <= start_time:
                raise HTTPException(
                    status_code=400,
                    detail="Horário de fim deve ser posterior ao horário de início"
                )

            # Verificar se a sessão existe
            session_exists = select(tabela_sessoes_estudo).where(
                tabela_sessoes_estudo.c.IdSessao == session_id
            )
            if not self.session.execute(session_exists).first():
                raise HTTPException(
                    status_code=404,
                    detail=f"Sessão {session_id} não encontrada"
                )

            # Construir o update
            update_stmt = (
                tabela_sessoes_estudo.update()
                .where(tabela_sessoes_estudo.c.IdSessao == session_id)
                .values(
                    Inicio=start_time,
                    Fim=end_time
                )
            )

            # Executar o update
            result = self.session.execute(update_stmt)
            self.session.commit()

            # Verificar se a atualização foi bem sucedida
            rows_affected = result.rowcount
            success = rows_affected > 0

            if success:
                logger.info(f"Horários da sessão {session_id} atualizados com sucesso")
                return True
            else:
                logger.info(f"Nenhuma alteração realizada para a sessão {session_id}")
                return False

        except HTTPException as he:
            self.session.rollback()
            raise he
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao atualizar horários da sessão: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao atualizar horários da sessão: {str(e)}"
            )

    def get_student_by_user_id(self, user_id: int) -> int:
        """
        Busca o ID do estudante com base no ID do usuário
        """
        try:
            query = select(tabela_estudantes.c.IdEstudante).where(
                tabela_estudantes.c.IdUsuario == user_id
            )
            result = self.session.execute(query).first()
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Estudante não encontrado para o usuário {user_id}"
                )
            return result.IdEstudante
        except Exception as e:
            logger.error(f"Erro ao buscar estudante: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao buscar estudante: {str(e)}"
            )

    def format_time(self, t: Optional[time]) -> Optional[str]:
        """Helper method to format time objects."""
        return t.strftime('%H:%M') if t else None

    def format_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """Helper method to format datetime objects."""
        return dt.isoformat() if dt else None

    def parse_objectives(self, objectives: str) -> list:
        """
        Parse JSON objectives with proper error handling.
        """
        if not objectives:
            return []
        
        try:
            if isinstance(objectives, str):
                return json.loads(objectives)
            elif isinstance(objectives, list):
                return objectives
            else:
                logger.info(f"AVISO: Tipo inesperado para objetivos: {type(objectives)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Erro: Falha ao interpretar JSON dos objetivos: {e}")
            return []

    def safe_get(self, dict_data: Dict, key: str, default: Any = None) -> Any:
        """
        Safely get a value from a dictionary with detailed logging.
        """
        try:
            value = dict_data.get(key, default)
            logger.info(f"Acessando chave '{key}': {value}")
            return value
        except Exception as e:
            logger.error(f"Erro ao acessar chave '{key}': {e}")
            return default

    def get_discipline_details(self, discipline_id: int, student_id: int) -> Dict[str, Any]:
        """
        Get complete details of a discipline, verifying student access.
        """
        try:
            logger.info(f"Buscando detalhes da disciplina {discipline_id} para o estudante {student_id}")

            # Check student enrollment
            enrollment_query = select(tabela_estudante_curso.c.Id).where(
                and_(
                    tabela_estudante_curso.c.IdEstudante == student_id,
                    tabela_estudante_curso.c.IdCurso == discipline_id
                )
            )

            logger.info(f"Query de matrícula gerada: {enrollment_query}")
            enrollment = self.session.execute(enrollment_query).first()

            if not enrollment:
                logger.info(f"AVISO: Estudante {student_id} não está matriculado na disciplina {discipline_id}")
                raise HTTPException(
                    status_code=404,
                    detail="Estudante não matriculado na disciplina"
                )

            # Get discipline details
            discipline_query = select(
                tabela_cursos.c.IdCurso,
                tabela_cursos.c.NomeCurso,
                tabela_cursos.c.Ementa,
                tabela_cursos.c.Objetivos,
                tabela_cursos.c.HorarioInicio,
                tabela_cursos.c.HorarioFim,
                tabela_cursos.c.CriadoEm,
                tabela_cursos.c.IdEducador,
                tabela_cursos.c.NomeEducador
            ).where(tabela_cursos.c.IdCurso == discipline_id)

            logger.info(f"Query de disciplina gerada: {discipline_query}")
            result = self.session.execute(discipline_query).mappings().first()

            if not result:
                logger.error(f"Erro: Disciplina {discipline_id} não encontrada")
                raise HTTPException(
                    status_code=404,
                    detail="Disciplina não encontrada"
                )

            logger.info(f"Tipo do resultado: {type(result)}")
            logger.info(f"Chaves disponíveis: {result.keys() if hasattr(result, 'keys') else 'Sem chaves disponíveis'}")
            logger.info(f"Resultado da query: {result}")

            try:
                # Transform the result into the desired format with safe access
                discipline_details = {
                    "IdCurso": self.safe_get(result, "IdCurso"),
                    "NomeCurso": self.safe_get(result, "NomeCurso"),
                    "Ementa": self.safe_get(result, "Ementa"),
                    "Objetivos": self.parse_objectives(self.safe_get(result, "Objetivos")),
                    "HorarioInicio": self.format_time(self.safe_get(result, "HorarioInicio")),
                    "HorarioFim": self.format_time(self.safe_get(result, "HorarioFim")),
                    "CriadoEm": self.format_datetime(self.safe_get(result, "CriadoEm")),
                    "IdEducador": self.safe_get(result, "IdEducador"),
                    "NomeEducador": self.safe_get(result, "NomeEducador")
                }

                logger.info(f"Detalhes da disciplina formatados com sucesso: {discipline_details}")
                return discipline_details

            except Exception as e:
                logger.error(f"Erro durante a formatação dos detalhes: {e}")
                logger.info(f"Stack trace completo:", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Erro ao formatar detalhes da disciplina: {str(e)}"
                )

        except SQLAlchemyError as e:
            logger.error(f"Erro: Erro no banco de dados ao buscar detalhes da disciplina: {e}")
            raise HTTPException(
                status_code=500,
                detail="Erro no banco de dados ao buscar detalhes da disciplina"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Erro: Erro ao interpretar JSON dos dados da disciplina: {e}")
            raise HTTPException(
                status_code=500,
                detail="Erro ao interpretar dados da disciplina"
            )

        except Exception as e:
            logger.error(f"Erro: Erro inesperado em get_discipline_details: {e}")
            logger.info("Stack trace completo:", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Ocorreu um erro inesperado"
            )

    def get_discipline_by_session_id(self, session_id: int):
        try:
            # Realiza o join entre a tabela Cursos e a tabela SessoesEstudo,
            # utilizando a coluna IdCurso, e filtra pela sessão desejada.
            query = (
                select(tabela_cursos)  # Seleciona todas as colunas da tabela Cursos
                .join(tabela_sessoes_estudo, tabela_cursos.c.IdCurso == tabela_sessoes_estudo.c.IdCurso)
                .where(tabela_sessoes_estudo.c.IdSessao == session_id)
            )
            result = self.session.execute(query).fetchone()
            if not result:
                return None

            # Caso deseje retornar os dados em formato de dicionário:
            return dict(result._mapping)

            # Se preferir retornar o objeto result diretamente, utilize:
            # return result

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao buscar o curso para a sessão '{session_id}': {str(e)}"
            )

    def update_user_account(self, email: str, updated_data: dict):
        """
        Atualiza os dados da conta de um usuário com base no e-mail.

        Parâmetros:
            email (str): E-mail do usuário a ser atualizado.
            updated_data (dict): Dicionário contendo os campos e valores a serem atualizados.

        Exemplo de updated_data:
            {
                "Nome": "Novo Nome",
                "Instituicao": "Nova Instituição",
                "SenhaHash": "novo_hash_da_senha"  # se for atualizar a senha
            }

        Retorna:
            Nenhum valor, mas realiza a atualização no banco de dados.

        Lança:
            HTTPException com status 404 se o usuário não for encontrado.
            HTTPException com status 500 em caso de erro interno.
        """
        try:
            # Atualiza o campo de data/hora de modificação
            updated_data["AtualizadoEm"] = datetime.now(timezone.utc)

            # Cria a instrução de atualização com base no email
            stmt = tabela_usuarios.update() \
                .where(tabela_usuarios.c.Email == email) \
                .values(**updated_data)

            result = self.session.execute(stmt)
            self.session.commit()

            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Usuário não encontrado.")

            logger.info(f"Conta do usuário com email {email} atualizada com sucesso.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao atualizar a conta do usuário: {e}")
            raise HTTPException(status_code=500, detail="Erro ao atualizar a conta do usuário.")

    def update_user_password(self, email: str, new_password_hash: str) -> bool:
        """
        Atualiza apenas o hash da senha do usuário com base no e-mail.

        Parâmetros:
            email (str): E-mail do usuário que terá a senha atualizada.
            new_password_hash (str): Novo hash da senha a ser armazenado.

        Retorna:
            bool: True se a atualização for bem-sucedida.

        Lança:
            HTTPException com status 404 se o usuário não for encontrado.
            HTTPException com status 500 em caso de erro interno.
        """
        try:
            updated_data = {
                "SenhaHash": new_password_hash,
                "AtualizadoEm": datetime.now(timezone.utc)
            }
            stmt = tabela_usuarios.update().where(tabela_usuarios.c.Email == email).values(**updated_data)
            result = self.session.execute(stmt)
            self.session.commit()
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Usuário não encontrado.")
            logger.info(f"Senha do usuário com email {email} atualizada com sucesso.")
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro ao atualizar a senha do usuário: {e}")
            raise HTTPException(status_code=500, detail="Erro ao atualizar a senha do usuário.")

    def insert_support_request_com_imagens(self, support_data: dict, images: list) -> dict:
        """
        Insere um novo registro de suporte/feedback e as imagens associadas diretamente no banco de dados SQL.

        Parâmetros:
            support_data (dict): Dicionário com os dados do suporte, por exemplo:
                {
                    "UserEmail": "usuario@exemplo.com",
                    "MessageType": "duvida",  # ou "problema", "sugestao", etc.
                    "Subject": "Assunto do chamado",
                    "Page": "Página ou módulo",
                    "Message": "Mensagem detalhada do usuário"
                }
            images (list): Lista com os dados binários (bytes) de cada imagem a ser armazenada.

        Retorna:
            dict: Dicionário com o ID do suporte inserido e os IDs das imagens associadas, ex.:
                {
                    "support_request_id": 123,
                    "images_ids": [1, 2, 3]
                }
        """
        try:
            support_request_id = self.inserir_dado_retorna_id(tabela_support_requests, support_data, "IdSupportRequest")

            inserted_images_ids = []

            for image in images:
                result = self.session.execute(
                    tabela_support_request_images.insert().returning(tabela_support_request_images.c.IdImage)
                    .values(
                        IdSupportRequest=support_request_id,
                        ImageData=image
                    )
                )
                image_id = result.fetchone()[0]
                inserted_images_ids.append(image_id)

            self.session.commit()
            return {"support_request_id": support_request_id, "images_ids": inserted_images_ids}

        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"[SUPPORT] IntegrityError ao inserir suporte com imagens: {e}")
            raise HTTPException(status_code=400, detail="Erro de integridade ao inserir suporte com imagens.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"[SUPPORT] Erro ao inserir suporte com imagens: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao inserir suporte com imagens.")

    def list_support_requests(self, user_email: str) -> list:
        """
        Lista todos os registros de suporte/feedback associados ao e-mail do usuário.

        Parâmetros:
            user_email (str): O e-mail do usuário para o qual os tickets devem ser listados.

        Retorna:
            list: Uma lista de dicionários, onde cada dicionário representa um ticket de suporte.
                Exemplo:
                [
                    {
                        "IdSupportRequest": <UUID>,
                        "UserEmail": "usuario@exemplo.com",
                        "MessageType": "duvida",
                        "Subject": "Assunto do chamado",
                        "Page": "Página ou módulo",
                        "Message": "Mensagem detalhada do usuário",
                        "CreatedAt": "2025-03-30T15:00:00"
                    },
                    ...
                ]
        """
        try:
            query = tabela_support_requests.select().where(tabela_support_requests.c.UserEmail == user_email)
            result = self.session.execute(query).fetchall()

            # Use row._mapping para converter a linha para dicionário
            tickets = [dict(row._mapping) for row in result]
            return tickets

        except Exception as e:
            self.session.rollback()
            logger.error(f"[SUPPORT] Erro ao listar tickets de suporte para o usuário {user_email}: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao listar tickets de suporte.")

    def get_support_request_by_id(self, support_id: str, user_email: str) -> dict:
        """
        Busca um registro de suporte específico, filtrando pelo ID do ticket e pelo e-mail do usuário.

        Parâmetros:
            support_id (str): O ID do ticket de suporte (UUID como string).
            user_email (str): O e-mail do usuário para garantir que o ticket pertence a ele.

        Retorna:
            dict: Um dicionário representando o ticket de suporte.
                Exemplo:
                {
                    "IdSupportRequest": <UUID>,
                    "UserEmail": "usuario@exemplo.com",
                    "MessageType": "duvida",
                    "Subject": "Assunto do chamado",
                    "Page": "Página ou módulo",
                    "Message": "Mensagem detalhada do usuário",
                    "CreatedAt": "2025-03-30T15:00:00"
                }
            Se não encontrar, retorna None.
        """
        try:
            query = tabela_support_requests.select().where(
                (tabela_support_requests.c.IdSupportRequest == support_id) &
                (tabela_support_requests.c.UserEmail == user_email)
            )
            result = self.session.execute(query).fetchone()
            if result is None:
                return None
            # Utilize row._mapping para converter a linha em dicionário
            return dict(result._mapping)
        except Exception as e:
            self.session.rollback()
            logger.error(f"[SUPPORT] Erro ao buscar ticket de suporte {support_id} para o usuário {user_email}: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao buscar ticket de suporte.")

    def get_support_request_images(self, support_id: str) -> list:
        """
        Recupera a lista de imagens associadas a um ticket de suporte.

        Parâmetros:
            support_id (str): O ID do ticket de suporte (UUID como string).

        Retorna:
            list: Uma lista de dicionários, onde cada dicionário representa uma imagem associada.
                Exemplo:
                [
                    {
                        "IdImage": 1,
                        "IdSupportRequest": "<UUID>",
                        "ImageData": "base64encodedstring1"
                    },
                    ...
                ]
        """
        try:
            query = tabela_support_request_images.select().where(
                tabela_support_request_images.c.IdSupportRequest == support_id
            )
            result = self.session.execute(query).fetchall()
            images = []
            for row in result:
                # Converte a linha para dicionário usando row._mapping
                data = dict(row._mapping)
                # Converte os dados binários da imagem para uma string base64
                if data.get("ImageData"):
                    data["ImageData"] = base64.b64encode(data["ImageData"]).decode("utf-8")
                images.append(data)
            return images
        except Exception as e:
            self.session.rollback()
            logger.error(f"[SUPPORT] Erro ao recuperar imagens para o ticket {support_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao recuperar imagens do ticket de suporte.")