# api/dispatchers/discipline_dispatcher.py

from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_educadores, tabela_eventos_calendario, tabela_estudantes, tabela_cronograma, tabela_usuarios, tabela_perfil_aprendizado_aluno
from datetime import datetime, timedelta
from api.controllers.auth import hash_password, verify_password
from sqlalchemy import text
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
import json

class DisciplineDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get_all_disciplines_for_student(self, current_user: str):
        try:
            # Primeiro, obtenha o IdUsuario do estudante com base no e-mail do current_user
            user = self.database_manager.get_user_by_email(current_user)
            if not user:
                raise HTTPException(status_code=404, detail="Usuário não encontrado.")

            # Use o IdUsuario para buscar os perfis de aprendizado
            profiles = self.database_manager.get_learning_profiles_by_user_id(user.IdUsuario)
            if not profiles:
                return {"message": "Nenhuma disciplina encontrada para o estudante."}
            print(f"Perfis de aprendizado encontrados para o estudante {current_user}: {profiles}")

            # Obtenha as disciplinas (cursos) associadas aos perfis de aprendizado
            courses = []
            for profile in profiles:
                course = self.database_manager.get_course_by_id(profile.IdCurso)
                if course:
                    courses.append(course)

            return courses

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao buscar disciplinas do estudante: {e}")


    def create_discipline(self, discipline_data: dict, current_user: str):
        try:
            # Get the user's ID based on the email
            user_id = self.database_manager.get_user_id_by_email(current_user)
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
            professores = curso_data.get('professores', [])  # Pegando os professores do JSON
            print(f"Curso: {curso_data}")
            print(f"Cronograma: {cronograma_data}")
            print(f"Professores: {professores}")

            # 2. Buscar o ID do usuário com base no e-mail
            print(f"Buscando ID do usuário com e-mail: {user_email}")
            user = self.database_manager.get_user_by_email(user_email)
            if not user:
                raise Exception(f"Usuário com e-mail {user_email} não encontrado.")
            user_id = user.IdUsuario
            print(f"ID do usuário: {user_id}")

            # 3. Verificar ou criar os educadores
            educator_ids = []
            nome_educadores = []
            for professor in professores:
                # Verificar se o educador já existe na tabela Usuarios
                print(f"Buscando educador {professor}...")
                educator = self.database_manager.get_educator_by_name(professor)

                if educator is None:
                    # Se o educador não for encontrado, associar apenas o nome
                    nome_educadores.append(professor)
                    print(f"Educador {professor} não encontrado. Associando o nome na coluna NomeEducador.")
                else:
                    educator_id = educator.IdUsuario  # Usando o ID do usuário como educador
                    educator_ids.append(educator_id)
                    print(f"Educador {professor} encontrado com ID: {educator_id}")

            # 4. Verificar se o curso já existe
            existing_course_query = self.database_manager.session.execute(
                tabela_cursos.select().where(
                    (tabela_cursos.c.NomeCurso == curso_data.get('nome', 'Sem Nome')) &
                    (tabela_cursos.c.IdEducador.in_(educator_ids) if educator_ids else True)
                )
            )
            existing_course = existing_course_query.fetchone()

            if existing_course:
                course_id = existing_course[0]
                print(f"Curso já existe com ID: {course_id}.")
            else:
                # Inserir o curso na tabela Cursos se não existir
                print("Inserindo nova disciplina no banco de dados...")
                new_course = tabela_cursos.insert().values(
                    NomeCurso=curso_data.get('nome', 'Sem Nome'),
                    Ementa=curso_data.get('ementa', 'Sem Ementa'),
                    Objetivos=json.dumps(curso_data.get('objetivos', [])),
                    IdEducador=educator_ids[0] if educator_ids else None,  # Usar o primeiro ID de educador ou None
                    NomeEducador=', '.join(nome_educadores) if nome_educadores else None,  # Inserir nomes dos educadores se não houver ID
                    CriadoEm=datetime.now()
                )
                result = self.database_manager.session.execute(new_course)
                self.database_manager.session.commit()
                course_id = result.inserted_primary_key[0]
                print(f"Curso criado com ID: {course_id}")

            # 5. Verificar ou criar o cronograma
            existing_cronograma_query = self.database_manager.session.execute(
                tabela_cronograma.select().where(
                    (tabela_cronograma.c.IdCurso == course_id) &
                    (tabela_cronograma.c.NomeCronograma == f"Cronograma de {curso_data.get('nome', 'Sem Nome')}")
                )
            )
            existing_cronograma = existing_cronograma_query.fetchone()

            if existing_cronograma:
                cronograma_id = existing_cronograma[0]
                print(f"Cronograma já existe com ID: {cronograma_id} para o curso {course_id}")
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
                print(f"Cronograma criado com ID: {cronograma_id} para o curso {course_id}")

            # 6. Criar sessões de estudo e adicionar eventos no calendário
            for encontro in cronograma_data:
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
                    print(f"Sessão {session_number} já existe. Pulando inserção.")
                    continue

                # Inserir a sessão
                new_session = tabela_encontros.insert().values(
                    IdCronograma=cronograma_id,
                    NumeroEncontro=session_number,
                    DataEncontro=session_date,
                    Conteudo=encontro['conteudo'].strip(),
                    Estrategia=encontro.get('estrategia', 'Não especificada'),
                    Avaliacao=encontro.get('avaliacao')
                )
                self.database_manager.session.execute(new_session)

                # Adicionar evento ao calendário
                new_event = tabela_eventos_calendario.insert().values(
                    GoogleEventId=f"event-{course_id}-{session_number}",
                    Titulo=f"Encontro {session_number} - {curso_data.get('nome', 'Sem Nome')}",
                    Descricao=encontro['conteudo'].strip(),
                    Inicio=session_date,
                    Fim=session_date + timedelta(hours=2),
                    Local="Sala de Aula Física",
                    CriadoPor=user_id
                )
                self.database_manager.session.execute(new_event)

            # 7. Adicionar o curso ao perfil de aprendizado do aluno
            print(f"Associando o curso {course_id} ao perfil de aprendizado do aluno {user_id}...")
            new_profile = tabela_perfil_aprendizado_aluno.insert().values(
                IdUsuario=user_id,
                IdCurso=course_id,
                DadosPerfil=json.dumps({"progresso": "iniciado"})  # Pode-se ajustar o JSON conforme necessário
            )
            self.database_manager.session.execute(new_profile)

            # Commit para salvar todos os dados
            self.database_manager.session.commit()
            print(f"Curso '{curso_data.get('nome', 'Sem Nome')}' e sessões criadas com sucesso.")

        except Exception as e:
            self.database_manager.session.rollback()
            print(f"Erro ao criar disciplina: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao criar disciplina: {e}")
