# dispatchers/study_sessions_dispatcher.py

from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_cursos, tabela_encontros, tabela_eventos_calendario, tabela_sessoes_estudo, tabela_cronograma, tabela_usuarios
from datetime import datetime, timedelta
from api.controllers.auth import hash_password, verify_password
import json


class StudySessionsDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.session = database_manager.session

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
