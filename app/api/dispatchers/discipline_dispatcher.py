# api/dispatchers/discipline_dispatcher.py

from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_educadores, tabela_eventos_calendario, tabela_estudantes, tabela_cronograma, tabela_usuarios, tabela_perfil_aprendizado_aluno, tabela_estudante_curso
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

            # Obtenha o IdEstudante
            student_id = self.database_manager.get_student_by_user_email(current_user)
            if not student_id:
                raise HTTPException(status_code=404, detail="Estudante não encontrado.")

            # Use o IdEstudante para buscar as disciplinas associadas
            courses = self.database_manager.get_courses_by_student_id(student_id)
            if not courses:
                return {"message": "Nenhuma disciplina encontrada para o estudante."}

            # Converter as disciplinas em formato JSON-friendly (dicionários)
            courses_list = []
            for course in courses:
                course_dict = {
                    "IdCurso": course.IdCurso,
                    "NomeCurso": course.NomeCurso,
                    "Ementa": course.Ementa,
                    "Objetivos": course.Objetivos,
                    "CriadoEm": course.CriadoEm.isoformat()  # Converter datetime para string
                }
                courses_list.append(course_dict)

            return courses_list

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao buscar disciplinas do estudante: {e}")


    def create_discipline(self, discipline_data: dict, current_user: str):
        try:
            # 1. Obter o ID do aluno com base no e-mail do usuário atual
            print(f"Buscando ID do usuário com e-mail: {current_user}")
            user_id = self.database_manager.get_user_id_by_email(current_user)
            if not user_id:
                raise HTTPException(status_code=404, detail="Aluno não encontrado.")
            print(f"ID do usuário encontrado: {user_id}")

            # 2. Inserir a disciplina no banco de dados com os dados passados
            print("Inserindo nova disciplina no banco de dados...")
            new_course = tabela_cursos.insert().values(
                NomeCurso=discipline_data.get('NomeCurso', 'Sem Nome'),
                Ementa=discipline_data.get('Ementa', 'Sem Ementa'),
                Objetivos=discipline_data.get('Objetivos', 'Sem Objetivos'),
                CriadoEm=datetime.now()
            )
            result = self.database_manager.session.execute(new_course)
            self.database_manager.session.commit()
            course_id = result.inserted_primary_key[0]
            print(f"Disciplina criada com ID: {course_id}")

            # 3. Utilizar o método personalizado para associar o aluno ao curso
            print(f"Associando o curso {course_id} ao aluno {user_id}...")
            self.database_manager.associar_aluno_curso(user_id, course_id)

            return {"message": "Disciplina criada e associada com sucesso."}

        except IntegrityError as e:
            self.database_manager.session.rollback()
            print(f"IntegrityError: {e}")
            raise HTTPException(status_code=400, detail="Disciplina já existe.")
        except Exception as e:
            self.database_manager.session.rollback()
            print(f"Erro ao criar disciplina: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao criar disciplina: {e}")



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

            # 3. Inserir a disciplina no banco de dados
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

            # 4. Inserir o cronograma e encontros no banco de dados
            print(f"Inserindo cronograma para o curso ID: {course_id}")
            cronograma_query = tabela_cronograma.insert().values(
                IdCurso=course_id,
                NomeCronograma=f"Cronograma de {curso_data.get('nome', 'Sem Nome')}"
            )
            cronograma_result = self.database_manager.session.execute(cronograma_query)
            self.database_manager.session.commit()
            cronograma_id = cronograma_result.inserted_primary_key[0]
            print(f"Cronograma criado com ID: {cronograma_id} para o curso {course_id}")

            # 5. Inserir cada encontro no banco de dados
            for encontro in cronograma_data:
                session_date = datetime.strptime(encontro['data'], "%d/%m/%Y")
                session_number = encontro['numero_encontro']
                new_session = tabela_encontros.insert().values(
                    IdCronograma=cronograma_id,
                    NumeroEncontro=session_number,
                    DataEncontro=session_date,
                    Conteudo=encontro['conteudo'].strip(),
                    Estrategia=encontro.get('estrategia', 'Não especificada'),
                    Avaliacao=encontro.get('avaliacao')
                )
                self.database_manager.session.execute(new_session)

            self.database_manager.session.commit()
            print(f"Encontros do cronograma inseridos com sucesso para o curso {course_id}.")

            # 6. Associar o curso ao aluno na tabela EstudanteCurso
            print(f"Associando o curso {course_id} ao aluno {user_email}...")
            self.database_manager.session.execute(
                tabela_estudante_curso.insert().values(
                    IdEstudante=user_id,
                    IdCurso=course_id,
                    CriadoEm=datetime.now()
                )
            )
            self.database_manager.session.commit()
            print(f"Curso '{curso_data.get('nome', 'Sem Nome')}' associado ao aluno {user_email} com sucesso.")

        except Exception as e:
            self.database_manager.session.rollback()
            print(f"Erro ao criar disciplina: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao criar disciplina: {e}")
