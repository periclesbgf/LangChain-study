from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sql_test.sql_test_create import tabela_usuarios
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from sqlalchemy.sql import text

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

