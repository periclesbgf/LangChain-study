from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sql_test.sql_test_create import tabela_usuarios, tabela_educadores, tabela_cursos, tabela_sessoes_estudo, tabela_eventos_calendario, tabela_estudantes, tabela_perfil_aprendizado_aluno
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from sqlalchemy.sql import text
import json
from sqlalchemy.sql import select

load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")

engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
metadata = MetaData()

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
            # Execute a inserção e capture o resultado correto para a tabela de eventos
            result = self.session.execute(tabela.insert().returning(tabela.c.IdUsuario).values(dados))
            self.session.commit()
            print(f"Dado inserido com sucesso na tabela {tabela.name}")
            # Retorne o ID recém-inserido (IdEvento neste caso)
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
            # Realiza a inserção e retorna o ID do novo registro
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

    def inserir_dado_evento(self, tabela, dados):
        try:
            # Execute a inserção e capture o resultado correto para a tabela de eventos
            result = self.session.execute(tabela.insert().returning(tabela.c.IdEvento).values(dados))
            self.session.commit()
            inserted_id = result.fetchone()[0]  # Obter o ID do evento recém-inserido
            print(f"Dado inserido com sucesso na tabela {tabela.name} com IdEvento: {inserted_id}")
            return inserted_id
        except IntegrityError as e:
            self.session.rollback()
            print(f"IntegrityError during event insertion: {e}")
            raise HTTPException(status_code=400, detail="Duplicated entry.")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao inserir dado na tabela {tabela.name}: {e}")
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
        Function to fetch the educator's ID based on the user's email.
        """
        try:
            # First, get the user ID by email
            user_query = text('SELECT "IdUsuario" FROM "Usuarios" WHERE "Email" = :email')
            user = self.session.execute(user_query, {'email': user_email}).fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            user_id = user[0]

            # Now, get the educator ID based on the user ID
            educator_query = text('SELECT "IdEducador" FROM "Educadores" WHERE "IdUsuario" = :user_id')
            educator = self.session.execute(educator_query, {'user_id': user_id}).fetchone()
            if not educator:
                raise HTTPException(status_code=404, detail="Educator not found")
            return educator[0]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching educator ID: {str(e)}")

    def get_all_educator_names(self):
        """
        Função para recuperar todos os nomes dos educadores da tabela Usuarios e Educadores.
        """
        try:
            # Correção: Usar select direto, sem colchetes
            query = select(tabela_usuarios.c.Nome).select_from(
                tabela_usuarios.join(tabela_educadores, tabela_usuarios.c.IdUsuario == tabela_educadores.c.IdUsuario)
            )

            result = self.session.execute(query).fetchall()

            # Transformar o resultado em uma lista de nomes
            educator_names = [row[0] for row in result]  # Acessa o primeiro campo de cada linha (Nome)
            return educator_names

        except Exception as e:
            print(f"Erro ao buscar os nomes dos educadores: {e}")
            raise HTTPException(status_code=500, detail="Error fetching educator names.")

    def get_all_events_by_user(self, tabela_eventos, user_id: int):
        """
        Função para buscar todos os eventos de um usuário específico na tabela de eventos.
        :param tabela_eventos: A tabela onde os eventos estão armazenados.
        :param user_id: O ID do usuário criador dos eventos.
        :return: Uma lista de eventos criados pelo usuário.
        """
        try:
            # Criar a consulta para selecionar eventos criados pelo usuário específico
            query = select(tabela_eventos).where(tabela_eventos.c.CriadoPor == user_id)
            print(f"Executing query: {query}")  # Log para depuração da consulta

            # Executar a consulta e buscar todos os eventos
            result = self.session.execute(query).fetchall()
            return result
        except Exception as e:
            print(f"Erro ao selecionar eventos do usuário {user_id}: {e}")
            return None

    def get_student_by_user_email(self, user_email: str):
        """
        Função para obter o IdEstudante com base no e-mail do usuário.
        :param user_email: O e-mail do usuário.
        :return: O IdEstudante associado ao usuário ou None se não existir.
        """
        try:
            # Obtenha o IdUsuario com base no e-mail do usuário
            user = self.get_user_by_email(user_email)
            if not user:
                raise HTTPException(status_code=404, detail="Usuário não encontrado.")

            # Busque o estudante com base no IdUsuario
            query = select(tabela_estudantes.c.IdEstudante).where(tabela_estudantes.c.IdUsuario == user.IdUsuario)
            student = self.session.execute(query).fetchone()
            
            if not student:
                print(f"Estudante não encontrado para o usuário {user_email}.")
                return None
            
            return student[0]  # Retorna o IdEstudante
        except Exception as e:
            print(f"Erro ao buscar estudante para o e-mail {user_email}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar estudante.")


    def get_course_by_name(self, discipline_name: str):
        """
        Função para obter o curso com base no nome da disciplina.
        :param discipline_name: O nome da disciplina.
        :return: O IdCurso associado à disciplina.
        """
        try:
            # Buscar o curso com base no nome da disciplina
            query = select(tabela_cursos.c.IdCurso).where(tabela_cursos.c.NomeCurso == discipline_name)
            course = self.session.execute(query).fetchone()

            if not course:
                raise HTTPException(status_code=404, detail=f"Disciplina '{discipline_name}' não encontrada.")
            
            return course[0]  # Retorna o IdCurso
        except Exception as e:
            print(f"Erro ao buscar curso para a disciplina {discipline_name}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar curso.")

    def get_study_sessions_by_course_and_student(self, course_id: int, student_id: int):
        """
        Função para buscar todas as sessões de estudo de um estudante para um curso específico.
        :param course_id: O ID do curso.
        :param student_id: O ID do estudante.
        :return: Todas as sessões de estudo do estudante para o curso.
        """
        try:
            # Buscar as sessões de estudo com base no IdCurso e IdEstudante
            query = select(tabela_sessoes_estudo).where(
                (tabela_sessoes_estudo.c.IdCurso == course_id) &
                (tabela_sessoes_estudo.c.IdEstudante == student_id)
            )
            sessions = self.session.execute(query).fetchall()

            if not sessions:
                raise HTTPException(status_code=404, detail="Nenhuma sessão de estudo encontrada.")

            return sessions
        except Exception as e:
            print(f"Erro ao buscar sessões de estudo para o curso {course_id} e estudante {student_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar sessões de estudo.")

    def get_learning_profiles_by_user_id(self, user_id: int):
        """
        Função para obter todos os perfis de aprendizado do aluno com base no IdUsuario.
        """
        try:
            query = select(tabela_perfil_aprendizado_aluno).where(tabela_perfil_aprendizado_aluno.c.IdUsuario == user_id)
            profiles = self.session.execute(query).fetchall()
            
            if not profiles:
                print(f"Nenhum perfil de aprendizado encontrado para o usuário {user_id}")
                return []  # Retorna uma lista vazia ao invés de levantar exceção
                
            return profiles
        except Exception as e:
            print(f"Erro ao buscar perfis de aprendizado para o usuário {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao buscar perfis de aprendizado.")


    # Método para buscar curso pelo IdCurso
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
        :param nome_educador: Nome do educador.
        :return: O educador encontrado ou None se não existir.
        """
        try:
            # Busca o educador pelo nome na tabela Usuarios
            query = select(tabela_usuarios).where(tabela_usuarios.c.Nome == nome_educador)
            educator = self.session.execute(query).fetchone()
            
            # Verifica se o educador foi encontrado
            if educator is None:
                print(f"Educador {nome_educador} não encontrado.")
            return educator
        except Exception as e:
            print(f"Erro ao buscar educador {nome_educador}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao buscar educador {nome_educador}: {e}")
