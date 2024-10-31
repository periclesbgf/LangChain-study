# api/dispatchers/discipline_dispatcher.py

from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_sessoes_estudo, tabela_eventos_calendario, tabela_estudantes, tabela_cronograma, tabela_usuarios, tabela_perfil_aprendizado_aluno, tabela_estudante_curso
from datetime import datetime, timedelta
from api.controllers.auth import hash_password, verify_password
from sqlalchemy import text
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
import json
from datetime import datetime

class DisciplineDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get_discipline_by_id(self, discipline_id: int, current_user: str):
        """
        Busca uma disciplina específica, verificando se o usuário atual tem acesso a ela.
        """
        try:
            # Obter o ID do estudante
            student_id = self.database_manager.get_student_by_user_email(current_user)
            if not student_id:
                raise HTTPException(
                    status_code=404,
                    detail="Estudante não encontrado."
                )

            # Buscar os detalhes da disciplina
            discipline = self.database_manager.get_discipline_details(discipline_id, student_id)
            if not discipline:
                raise HTTPException(
                    status_code=404,
                    detail="Disciplina não encontrada ou você não tem acesso a ela."
                )

            # Retornar diretamente o objeto discipline sem envolvê-lo em outro dicionário
            return discipline

        except HTTPException as http_error:
            raise http_error
        except Exception as e:
            print(f"Erro ao buscar disciplina: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro interno ao buscar disciplina: {str(e)}"
            )

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


    async def create_discipline_from_pdf(self, discipline_json: dict, user_email: str, horario_inicio: str, horario_fim: str, turno_estudo: str):
        try:
            # 1. Extrair dados do JSON
            curso_data = discipline_json.get('curso', {})
            encontros_data = discipline_json.get('cronograma', [])  # Renomeado para evitar conflito
            professores = curso_data.get('professores', [])

            # 2. Buscar o ID do usuário e do estudante
            user = self.database_manager.get_user_by_email(user_email)
            if not user:
                raise Exception(f"Usuário com e-mail {user_email} não encontrado.")
            user_id = user.IdUsuario

            # 3. Buscar o ID do estudante
            student_id = self.database_manager.get_student_by_user_id(user_id)
            if not student_id:
                raise Exception(f"Estudante não encontrado para o usuário {user_id}")

            # 4. Converter horários para objetos TIME
            horario_inicio_obj = datetime.strptime(horario_inicio, '%H:%M').time()
            horario_fim_obj = datetime.strptime(horario_fim, '%H:%M').time()

            # 5. Inserir a disciplina no banco de dados
            new_course = tabela_cursos.insert().values(
                NomeCurso=curso_data.get('nome', 'Sem Nome'),
                Ementa=curso_data.get('ementa', 'Sem Ementa'),
                Objetivos=json.dumps(curso_data.get('objetivos', [])),
                HorarioInicio=horario_inicio_obj,
                HorarioFim=horario_fim_obj,
                CriadoEm=datetime.now()
            )
            result = self.database_manager.session.execute(new_course)
            self.database_manager.session.commit()
            course_id = result.inserted_primary_key[0]

            # 6. Inserir o cronograma
            cronograma_query = tabela_cronograma.insert().values(
                IdCurso=course_id,
                NomeCronograma=f"Cronograma de {curso_data.get('nome', 'Sem Nome')}"
            )
            cronograma_result = self.database_manager.session.execute(cronograma_query)
            self.database_manager.session.commit()
            cronograma_id = cronograma_result.inserted_primary_key[0]

            # 7. Inserir encontros e criar sessões de estudo
            session_ids = []
            for encontro in encontros_data:  # Usando a nova variável encontros_data
                # Converter a data do encontro
                session_date = datetime.strptime(encontro['data'], "%d/%m/%Y")
                session_number = encontro['numero_encontro']
                conteudo = encontro['conteudo'].strip()

                # Inserir o encontro
                new_session = tabela_encontros.insert().values(
                    IdCronograma=cronograma_id,
                    NumeroEncontro=session_number,
                    DataEncontro=session_date,
                    Conteudo=conteudo,
                    Estrategia=encontro.get('estrategia', 'Não especificada'),
                    Avaliacao=encontro.get('avaliacao'),
                    HorarioInicio=horario_inicio_obj,
                    HorarioFim=horario_fim_obj
                )
                self.database_manager.session.execute(new_session)

                # Inserir evento no calendário
                new_event = tabela_eventos_calendario.insert().values(
                    GoogleEventId=f"event-{course_id}-{session_number}",
                    Titulo=f"Encontro {session_number} - {curso_data.get('nome', 'Sem Nome')}",
                    Descricao=conteudo,
                    Inicio=datetime.combine(session_date.date(), horario_inicio_obj),
                    Fim=datetime.combine(session_date.date(), horario_fim_obj),
                    Local="Sala de Aula Física",
                    CriadoPor=user_id
                )
                self.database_manager.session.execute(new_event)

                # Criar sessão de estudo
                new_study_session = tabela_sessoes_estudo.insert().values(
                    IdEstudante=student_id,
                    IdCurso=course_id,
                    Assunto=conteudo,
                    Inicio=datetime.combine(session_date.date(), horario_inicio_obj),  # Definindo horário inicial
                    Fim=datetime.combine(session_date.date(), horario_fim_obj),        # Definindo horário final
                    Produtividade=0,
                    FeedbackDoAluno=None,
                    HistoricoConversa=None,
                    PreferenciaHorario=turno_estudo
                )
                result = self.database_manager.session.execute(new_study_session)
                session_id = result.inserted_primary_key[0]
                session_ids.append(session_id)

            # 8. Associar o curso ao estudante
            self.database_manager.session.execute(
                tabela_estudante_curso.insert().values(
                    IdEstudante=student_id,
                    IdCurso=course_id,
                    CriadoEm=datetime.now()
                )
            )
            self.database_manager.session.commit()

            return course_id, session_ids

        except Exception as e:
            self.database_manager.session.rollback()
            print(f"Erro ao criar disciplina: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao criar disciplina: {e}")