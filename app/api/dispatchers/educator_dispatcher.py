from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from sql_test.sql_test_create import tabela_educadores
from database.sql_database_manager import DatabaseManager

class EducatorDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get_all_educators(self, current_user: str):
        try:
            # Buscar todos os educadores
            educator_list = self.database_manager.get_all_educator_names()
            return educator_list

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching educators: {e}")

    def create_educator(self, educator_data: dict):
        try:
            # Criar um novo educador no banco de dados
            self.database_manager.inserir_dado(tabela_educadores, educator_data)
            return True
        except IntegrityError:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=400, detail="Educator already exists.")
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating educator: {e}")

    def update_educator(self, educator_id: int, updated_data: dict, current_user: str):
        try:
            # Atualizar o educador no banco de dados
            self.database_manager.atualizar_dado(
                tabela_educadores,
                tabela_educadores.c.IdEducador == educator_id,
                updated_data
            )
            return True
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating educator: {e}")

    def delete_educator(self, educator_id: int, current_user: str):
        try:
            # Deletar o educador do banco de dados
            self.database_manager.deletar_dado(
                tabela_educadores,
                tabela_educadores.c.IdEducador == educator_id
            )
            return True
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error deleting educator: {e}")
