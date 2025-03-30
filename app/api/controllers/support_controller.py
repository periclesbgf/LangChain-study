# api/controllers/support_controller.py

from fastapi import HTTPException
from api.dispatchers.support_dispatcher import SupportDispatcher
import datetime
import uuid


class SupportController:
    def __init__(self, dispatcher: SupportDispatcher):
        self.dispatcher = dispatcher

    def create_support_request(self, support_data: dict, image_bytes_list: list):
        """
        Cria um novo registro de suporte/feedback, juntamente com as imagens associadas.

        Parâmetros:
            support_data (dict): Dados do suporte, incluindo a FK (UserEmail).
            image_bytes_list (list): Lista com os dados binários de cada imagem.

        Retorna:
            dict: Resultado com o ID do suporte criado e os IDs das imagens.
        """
        try:
            uuid_str = str(uuid.uuid4())
            created_at = datetime.datetime.now(datetime.timezone.utc)
            support_data.update({
                "IdSupportRequest": uuid_str,
                "CreatedAt": created_at
            })
            print("support_data:")
            print(support_data)
            result = self.dispatcher.create_support_request(support_data, image_bytes_list)
            return result
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=e.detail)

    def get_all_support_requests(self, user_email: str) -> list:
        """
        Recupera todos os tickets de suporte do usuário chamando o método do dispatcher.
        """
        try:
            return self.dispatcher.list_support_requests(user_email)
        except Exception as e:
            raise HTTPException(status_code=500, detail=e.detail)


    def get_support_request_by_id(self, support_id: str, user_email: str) -> dict:
        """
        Recupera o ticket de suporte com base no ID e valida que pertence ao usuário.
        Retorna os dados do ticket e a lista de imagens associadas.
        """
        try:
            ticket = self.dispatcher.get_support_request_by_id(support_id, user_email)
            if ticket is None:
                raise HTTPException(status_code=404, detail="Ticket de suporte não encontrado.")
            # Recupera as imagens associadas ao ticket
            images = self.dispatcher.get_support_request_images(support_id)
            # Adiciona as imagens ao dicionário do ticket
            ticket["images"] = images
            return ticket
        except Exception as e:
            raise HTTPException(status_code=500, detail=e.detail)