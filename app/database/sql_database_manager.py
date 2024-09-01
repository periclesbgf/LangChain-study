from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

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
            self.session.execute(tabela.insert().values(dados))
            self.session.commit()
            print(f"Dado inserido com sucesso na tabela {tabela.name}")
        except Exception as e:
            self.session.rollback()
            print(f"Erro ao inserir dado na tabela {tabela.name}: {e}")

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

