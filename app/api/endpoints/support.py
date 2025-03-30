# api/endpoints/support.py

from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends
from typing import List, Optional
from logg import logger
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.controllers.support_controller import SupportController
from api.dispatchers.support_dispatcher import SupportDispatcher
from api.endpoints.models import SupportRequest


router_support = APIRouter()

async def get_support_request_form(
    message_type: str = Form(...),
    subject: str = Form(...),
    page: str = Form(...),
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
) -> SupportRequest:
    """
    Extrai os dados do form-data e monta uma instância de SupportRequest.
    """
    return SupportRequest(
        message_type=message_type,
        subject=subject,
        page=page,
        message=message,
        images=images
    )

@router_support.post("/support")
async def create_support_request(
    support_request: SupportRequest = Depends(get_support_request_form),
    current_user: dict = Depends(get_current_user)
):
    """
    Cria um novo registro de suporte/feedback.

    Utiliza o modelo SupportRequest para receber os dados via form-data,
    incluindo uma lista de imagens (UploadFile). As imagens são lidas e convertidas para bytes.
    O suporte é associado ao e-mail do usuário autenticado.
    """
    try:
        print("support_request:")
        print(support_request)
        image_bytes_list = []
        if support_request.images:
            for img in support_request.images:
                image_bytes = await img.read()
                image_bytes_list.append(image_bytes)

        support_data = {
            "UserEmail": current_user['sub'],
            "MessageType": support_request.message_type,
            "Subject": support_request.subject,
            "Page": support_request.page,
            "Message": support_request.message
        }

        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = SupportDispatcher(sql_database_manager)
        controller = SupportController(dispatcher)

        logger.info(f"[SUPPORT_CREATE] Usuário {current_user.get('sub')} criou um novo ticket de {support_request.message_type}")

        result = controller.create_support_request(support_data, image_bytes_list)
        return {"message": "Ticket criado com sucesso", "data": result}

    except HTTPException as e:
        logger.error(f"[SUPPORT_CREATE] Erro ao criar suporte: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"[SUPPORT_CREATE] Erro inesperado ao criar suporte: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao criar suporte")

@router_support.get("/support")
async def get_all_support_requests(current_user: dict = Depends(get_current_user)):
    """
    Retorna todos os tickets de suporte associados ao usuário autenticado.

    O e-mail do usuário (obtido via get_current_user) é utilizado para filtrar os tickets.
    """
    try:
        # Instancia o DatabaseManager, Dispatcher e Controller
        db_manager = DatabaseManager(session, metadata)
        dispatcher = SupportDispatcher(db_manager)
        controller = SupportController(dispatcher)

        # Assumindo que current_user possua o campo "email"
        user_email = current_user['sub']
        if not user_email:
            raise HTTPException(status_code=400, detail="Email do usuário não encontrado.")

        tickets = controller.get_all_support_requests(user_email)
        return {"support_requests": tickets}
    except HTTPException as e:
        logger.error(f"[SUPPORT_GET_ALL] Erro ao obter tickets: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"[SUPPORT_GET_ALL] Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao obter tickets de suporte.")

@router_support.get("/support/{support_id}")
async def get_support_request_by_id(support_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retorna os detalhes de um ticket de suporte específico, incluindo as imagens associadas,
    garantindo que o ticket pertença ao usuário autenticado.

    Parâmetros:
        support_id (str): O ID do ticket de suporte (UUID como string).
        current_user: Objeto do usuário autenticado (deve conter o campo "email").
    """
    try:
        # Instancia os componentes: DatabaseManager, Dispatcher e Controller
        db_manager = DatabaseManager(session, metadata)
        dispatcher = SupportDispatcher(db_manager)
        controller = SupportController(dispatcher)

        user_email = current_user['sub']
        if not user_email:
            raise HTTPException(status_code=400, detail="Email do usuário não encontrado.")

        ticket = controller.get_support_request_by_id(support_id, user_email)
        return {"support_request": ticket}
    except HTTPException as e:
        logger.error(f"[SUPPORT_GET] Erro ao obter ticket: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"[SUPPORT_GET] Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=e.detail)
