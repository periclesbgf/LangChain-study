# api/endpoints/discipline.py

from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends, HTTPException
from fastapi.logger import logger
from sqlalchemy.orm import Session
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.controllers.discipline_controller import DisciplineController
from api.dispatchers.discipline_dispatcher import DisciplineDispatcher
from api.endpoints.models import DisciplineCreate, DisciplinePDFCreate, DisciplineUpdate
from utils import OPENAI_API_KEY, CODE
from chains.chain_setup import DisciplinChain
import pdfplumber
import re

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

@router_disciplines.get("/disciplines/{discipline_id}")
async def get_discipline_by_id(
    discipline_id: int,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Buscando disciplina {discipline_id} para o usuário: {current_user['sub']}")
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Fetch the discipline by ID
        discipline = controller.get_discipline_by_id(discipline_id, current_user['sub'])
        if not discipline:
            raise HTTPException(status_code=404, detail="Disciplina não encontrada.")

        logger.info(f"Disciplina {discipline_id} encontrada para o usuário: {current_user['sub']}")
        return {"discipline_name": discipline["NomeCurso"]}
    except Exception as e:
        logger.error(f"Erro ao buscar disciplina {discipline_id} para o usuário: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_disciplines.post("/disciplines")
async def create_discipline(
    discipline: DisciplineCreate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Criando nova disciplina para o usuário: {current_user['sub']}")
    try:
        # Validar turno
        if discipline.turno_estudo not in ['manha', 'tarde', 'noite']:
            raise HTTPException(status_code=400, detail="Turno inválido")

        # Validar formato do horário (HH:MM)
        time_pattern = re.compile(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')
        if not time_pattern.match(discipline.horario_inicio) or not time_pattern.match(discipline.horario_fim):
            raise HTTPException(status_code=400, detail="Formato de horário inválido")

        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Create a new discipline with all fields
        controller.create_discipline(
            nome_curso=discipline.nome_curso,
            ementa=discipline.ementa,
            objetivos=discipline.objetivos,
            educator=discipline.educator,
            turno_estudo=discipline.turno_estudo,
            horario_inicio=discipline.horario_inicio,
            horario_fim=discipline.horario_fim,
            current_user=current_user['sub']
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
        # Validar turno se fornecido
        if discipline_data.turno_estudo and discipline_data.turno_estudo not in ['manha', 'tarde', 'noite']:
            raise HTTPException(status_code=400, detail="Turno inválido")

        # Validar formato do horário se fornecido
        time_pattern = re.compile(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')
        if discipline_data.horario_inicio and not time_pattern.match(discipline_data.horario_inicio):
            raise HTTPException(status_code=400, detail="Formato de horário de início inválido")
        if discipline_data.horario_fim and not time_pattern.match(discipline_data.horario_fim):
            raise HTTPException(status_code=400, detail="Formato de horário de fim inválido")

        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        controller = DisciplineController(dispatcher)

        # Update discipline with all possible fields
        controller.update_discipline(
            discipline_id=discipline_id,
            nome_curso=discipline_data.nome_curso,
            ementa=discipline_data.ementa,
            objetivos=discipline_data.objetivos,
            turno_estudo=discipline_data.turno_estudo,
            horario_inicio=discipline_data.horario_inicio,
            horario_fim=discipline_data.horario_fim,
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
async def create_discipline_from_pdf(
    file: UploadFile = File(...),
    turno_estudo: str = Form(...),
    horario_inicio: str = Form(...),
    horario_fim: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Validar turno
        if turno_estudo not in ['manha', 'tarde', 'noite']:
            raise HTTPException(status_code=400, detail="Turno inválido")

        # Validar formato do horário (HH:MM)
        time_pattern = re.compile(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')
        if not time_pattern.match(horario_inicio) or not time_pattern.match(horario_fim):
            raise HTTPException(status_code=400, detail="Formato de horário inválido")

        # Ler o conteúdo do arquivo PDF
        file_bytes = await file.read()

        # Utilizar pdfplumber para extrair o texto
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        # Processar os dados
        sql_database_manager = DatabaseManager(session, metadata)
        dispatcher = DisciplineDispatcher(sql_database_manager)
        disciplin_chain = DisciplinChain(OPENAI_API_KEY)
        controller = DisciplineController(dispatcher, disciplin_chain)

        # Criar disciplina com os novos campos
        await controller.create_discipline_from_pdf(
            text=text,
            user_email=current_user['sub'],
            turno_estudo=turno_estudo,
            horario_inicio=horario_inicio,
            horario_fim=horario_fim
        )

        return {"message": "Disciplina e sessões de estudo criadas com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))