# api/endpoints/discipline.py

from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends, HTTPException
from fastapi.logger import logger
from sqlalchemy.orm import Session
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.controllers.discipline_controller import DisciplineController
from api.dispatchers.discipline_dispatcher import DisciplineDispatcher
from api.endpoints.models import DisciplineCreate, DisciplineUpdate
from utils import OPENAI_API_KEY, CODE
from chains.chain_setup import DisciplinChain
import pdfplumber

router_disciplines = APIRouter()

# GET - List all disciplines
@router_disciplines.get("/disciplines")
async def get_all_user_disciplines(current_user: dict = Depends(get_current_user)):
    logger.info(f"Buscando todas as disciplinas para o usuário: {current_user['sub']}")
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Fetch all disciplines for the current user
        disciplines = controller.get_all_user_disciplines(current_user['sub'])
        print("Disciplinas: ", disciplines)
        return {"disciplines": disciplines}
    except Exception as e:
        logger.error(f"Erro ao buscar disciplinas para o usuário: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# POST - Create a new discipline
@router_disciplines.post("/disciplines")
async def create_discipline(
    discipline: DisciplineCreate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Criando nova disciplina para o usuário: {current_user['sub']}")
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Create a new discipline
        controller.create_discipline(
            discipline.nome_curso,
            discipline.ementa,
            discipline.objetivos,
            current_user['sub']
        )

        logger.info(f"Nova disciplina criada com sucesso para o usuário: {current_user['sub']}")
        return {"message": "Discipline created successfully"}
    except Exception as e:
        logger.error(f"Erro ao criar disciplina para o usuário: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# PUT - Update an existing discipline
@router_disciplines.put("/disciplines/{discipline_id}")
async def update_discipline(
    discipline_id: int,
    discipline_data: DisciplineUpdate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Atualizando disciplina {discipline_id} para o usuário: {current_user['sub']}")
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Update discipline
        controller.update_discipline(
            discipline_id,
            nome_curso=discipline_data.nome_curso,
            ementa=discipline_data.ementa,
            objetivos=discipline_data.objetivos,
            current_user=current_user['sub']
        )

        logger.info(f"Disciplina {discipline_id} atualizada com sucesso para o usuário: {current_user['sub']}")
        return {"message": "Discipline updated successfully"}
    except Exception as e:
        logger.error(f"Erro ao atualizar disciplina {discipline_id} para o usuário: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# DELETE - Delete a discipline
@router_disciplines.delete("/disciplines/{discipline_id}")
async def delete_discipline(
    discipline_id: int,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Deletando disciplina {discipline_id} para o usuário: {current_user['sub']}")
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Delete discipline
        controller.delete_discipline(discipline_id, current_user['sub'])

        logger.info(f"Disciplina {discipline_id} deletada com sucesso para o usuário: {current_user['sub']}")
        return {"message": "Discipline deleted successfully"}
    except Exception as e:
        logger.error(f"Erro ao deletar disciplina {discipline_id} para o usuário: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_disciplines.post("/create_discipline_from_pdf")
async def endpoint_create_discipline_from_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Ler o conteúdo do arquivo PDF
        file_bytes = await file.read()

        # Utilizar pdfplumber para extrair o texto
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        print("Texto extraído do PDF")
        print(text)
        # Instanciar o dispatcher e passar para o controlador
        sql_database_manager = DatabaseManager(session, metadata)
        print("SQL Database Manager: ", sql_database_manager)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        print("Dispatcher: ", dispatcher)
        disciplin_chain = DisciplinChain(OPENAI_API_KEY)
        controller = DisciplineController(dispatcher, disciplin_chain)
        

        # Chamar o controlador para processar a lógica e salvar os dados
        controller.create_discipline_from_pdf(text, current_user['sub'])

        return {"message": "Disciplina e sessões de estudo criadas com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
