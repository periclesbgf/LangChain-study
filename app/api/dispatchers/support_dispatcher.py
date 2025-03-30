# api/dispatchers/support_dispatcher.py

from fastapi import HTTPException
from sql_interface.sql_tables import tabela_support_requests, tabela_support_request_images
from database.sql_database_manager import DatabaseManager
from logg import logger


class SupportDispatcher:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def create_support_request(self, support_data: dict, image_bytes_list: list):
        """
        Cria um novo registro de suporte/feedback e insere as imagens associadas.
        
        Parâmetros:
            support_data (dict): Dados do suporte, incluindo o campo "UserEmail" que referencia o usuário.
            image_bytes_list (list): Lista com os dados binários das imagens.
        
        Retorna:
            dict: Resultado com o ID do suporte criado e os IDs das imagens inseridas.
        """
        try:
            result = self.db_manager.insert_support_request_com_imagens(support_data, image_bytes_list)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no dispatcher ao criar suporte: {str(e)}")

    def list_support_requests(self, user_email: str) -> list:
        """
        Chama o método de listagem do DatabaseManager para retornar os tickets de suporte associados ao e-mail do usuário.
        """
        try:
            return self.db_manager.list_support_requests(user_email)
        except Exception as e:
            logger.error(f"Erro no dispatcher ao listar tickets de suporte para {user_email}: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao listar tickets de suporte.")

    def get_support_request_by_id(self, support_id: str, user_email: str) -> dict:
        """
        Recupera o ticket de suporte pelo ID, garantindo que ele pertença ao usuário.
        """
        try:
            return self.db_manager.get_support_request_by_id(support_id, user_email)
        except Exception as e:
            logger.error(f"Erro no dispatcher ao buscar ticket de suporte {support_id} para {user_email}: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao buscar ticket de suporte.")

    def get_support_request_images(self, support_id: str) -> list:
        """
        Recupera as imagens associadas ao ticket de suporte.
        """
        try:
            return self.db_manager.get_support_request_images(support_id)
        except Exception as e:
            logger.error(f"Erro no dispatcher ao buscar imagens para o ticket {support_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao buscar imagens do ticket de suporte.")