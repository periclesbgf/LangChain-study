# api/dispatchers/discipline_dispatcher.py

from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_eventos_calendario, tabela_sessoes_estudo, tabela_cronograma, tabela_usuarios
from datetime import datetime, timedelta
from api.controllers.auth import hash_password, verify_password
from sqlalchemy import text
import json

class DisciplineDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get_all_disciplines(self, current_user: str):
        # Get the educator's ID based on the user email
        educator_id = self.database_manager.get_educator_id_by_email(current_user)
        try:
            # Fetch all disciplines created by the educator
            disciplines = self.database_manager.selecionar_dados(
                tabela_cursos,
                tabela_cursos.c.IdEducador == educator_id
            )
            return disciplines

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching disciplines: {e}")

    def create_discipline(self, discipline_data: dict, current_user: str):
        try:
            # Get the educator's ID
            educator_id = self.database_manager.get_educator_id_by_email(current_user)

            # Assign the correct educator ID to the discipline data
            discipline_data['IdEducador'] = educator_id

            # Insert the discipline into the database
            self.database_manager.inserir_dado(tabela_cursos, discipline_data)
            return True
        except IntegrityError:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=400, detail="Discipline already exists.")
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating discipline: {e}")

    def update_discipline(self, discipline_id: int, updated_data: dict, current_user: str):
        try:
            # Get the educator's ID
            educator_id = self.database_manager.get_educator_id_by_email(current_user)

            # Update the discipline if it belongs to the educator
            self.database_manager.atualizar_dado(
                tabela_cursos,
                (tabela_cursos.c.IdCurso == discipline_id) & (tabela_cursos.c.IdEducador == educator_id),
                updated_data
            )
            return True
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating discipline: {e}")

    def delete_discipline(self, discipline_id: int, current_user: str):
        try:
            # Get the educator's ID
            educator_id = self.database_manager.get_educator_id_by_email(current_user)

            # Delete the discipline if it belongs to the educator
            self.database_manager.deletar_dado(
                tabela_cursos,
                (tabela_cursos.c.IdCurso == discipline_id) & (tabela_cursos.c.IdEducador == educator_id)
            )
            return True
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error deleting discipline: {e}")

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
            user_query = self.database_manager.session.execute(
                tabela_usuarios.select().where(tabela_usuarios.c.Email == user_email)
            )
            user = user_query.fetchone()

            if user is None:
                raise Exception(f"Usuário com e-mail {user_email} não encontrado.")

            # Certifique-se de que 'user' é uma tupla e obtenha o ID pelo índice
            user_id = user[0]  # 'IdUsuario' é provavelmente o primeiro campo retornado
            print(f"ID do usuário: {user_id}")

            # 3. Verificar se o curso já existe
            existing_course_query = self.database_manager.session.execute(
                tabela_cursos.select().where(tabela_cursos.c.NomeCurso == curso_data.get('nome', 'Sem Nome'))
            )
            existing_course = existing_course_query.fetchone()

            if existing_course:
                course_id = existing_course[0]
                print(f"Curso já existe com ID: {course_id}")
            else:
                # Inserir o curso na tabela Cursos se não existir
                print("Inserindo nova disciplina no banco de dados...")
                new_course = tabela_cursos.insert().values(
                    NomeCurso=curso_data.get('nome', 'Sem Nome'),
                    Ementa=curso_data.get('ementa', 'Sem Ementa'),
                    Objetivos=json.dumps(curso_data.get('objetivos', [])),
                    CriadoEm=datetime.now()
                )
                result = self.database_manager.session.execute(new_course)
                self.database_manager.session.commit()
                course_id = result.inserted_primary_key[0]
                print(f"Curso criado com ID: {course_id}")

            # 4. Verificar se o cronograma já existe
            existing_cronograma_query = self.database_manager.session.execute(
                tabela_cronograma.select().where(tabela_cronograma.c.IdCurso == course_id)
            )
            existing_cronograma = existing_cronograma_query.fetchone()

            if existing_cronograma:
                cronograma_id = existing_cronograma[0]
                print(f"Cronograma já existe com ID: {cronograma_id}")
            else:
                # Inserir o cronograma para o curso se não existir
                print(f"Inserindo cronograma para o curso ID: {course_id}")
                new_cronograma = tabela_cronograma.insert().values(
                    IdCurso=course_id,
                    NomeCronograma=f"Cronograma de {curso_data.get('nome', 'Sem Nome')}"
                )
                result_cronograma = self.database_manager.session.execute(new_cronograma)
                self.database_manager.session.commit()
                cronograma_id = result_cronograma.inserted_primary_key[0]
                print(f"Cronograma criado com ID: {cronograma_id}")

            # 5. Criar as sessões de estudo e adicionar eventos no calendário
            for encontro in cronograma_data:
                # Parse da data do encontro
                session_date = datetime.strptime(encontro['data'], "%d/%m/%Y")
                session_number = encontro['numero_encontro']

                # Verificar se a sessão já existe
                existing_session_query = self.database_manager.session.execute(
                    tabela_encontros.select().where(
                        (tabela_encontros.c.IdCronograma == cronograma_id) &
                        (tabela_encontros.c.NumeroEncontro == session_number)
                    )
                )
                existing_session = existing_session_query.fetchone()

                if existing_session:
                    print(f"Sessão {session_number} já existe para o cronograma {cronograma_id}. Pulando inserção.")
                    continue  # Pula a criação desta sessão se já existir

                print(f"Inserindo sessão {session_number} para o cronograma {cronograma_id}...")

                # Inserir o encontro na tabela Encontros
                new_session = tabela_encontros.insert().values(
                    IdCronograma=cronograma_id,
                    NumeroEncontro=session_number,
                    DataEncontro=session_date,
                    Conteudo=encontro['conteudo'].strip(),
                    Estrategia=encontro.get('estrategia', "Não especificada"),
                    Avaliacao=encontro.get('avaliacao')  # Campo opcional de avaliação
                )
                self.database_manager.session.execute(new_session)

                # Verificar se o evento do calendário já existe
                existing_event_query = self.database_manager.session.execute(
                    tabela_eventos_calendario.select().where(
                        (tabela_eventos_calendario.c.GoogleEventId == f"event-{course_id}-{session_number}")
                    )
                )
                existing_event = existing_event_query.fetchone()

                if existing_event:
                    print(f"Evento para a sessão {session_number} já existe no calendário. Pulando inserção.")
                    continue  # Pula a criação deste evento se já existir

                # Adicionar evento no calendário
                print(f"Adicionando evento no calendário para a sessão {session_number}...")
                new_event = tabela_eventos_calendario.insert().values(
                    GoogleEventId=f"event-{course_id}-{session_number}",
                    Titulo=f"Encontro {session_number} - {curso_data.get('nome', 'Sem Nome')}",
                    Descricao=encontro['conteudo'].strip(),
                    Inicio=session_date,
                    Fim=session_date + timedelta(hours=2),
                    Local="Sala de Aula Física",
                    CriadoPor=user_id  # Usar o ID do usuário em vez do e-mail
                )
                self.database_manager.session.execute(new_event)

            # 6. Commit para salvar os encontros e eventos
            self.database_manager.session.commit()
            print(f"Curso '{curso_data.get('nome', 'Sem Nome')}' e sessões criadas com sucesso.")

        except Exception as e:
            print(f"Erro detectado: {e}")
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Erro ao criar disciplina e sessões: {e}")