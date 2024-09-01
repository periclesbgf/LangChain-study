from api.endpoints.models import Question, ResponseModel
from api.controllers.controller import (
    code_confirmation,
    build_chain,
    build_sql_chain,
    route_request,
    insertDocsInVectorDatabase
    )
from api.controllers.database_controller import DatabaseController
from database.sql_database_manager import DatabaseManager, session, metadata
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from agent.chat import ConversationHistory
from fastapi.logger import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy import insert


history = ConversationHistory()

router = APIRouter()


@router.post("/create_account")
async def create_account(
    nome: str = Form(...),
    email: str = Form(...),
    senha: str = Form(...),
    tipo_usuario: str = Form(..., regex="^(student|educator)$"),
):
    print("Creating account")
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = DatabaseController(sql_database_manager)
        print("connecting to database")
        sql_database_controller.create_account(nome, email, senha, tipo_usuario)

        return {"message": "Conta criada com sucesso"}

    except IntegrityError as e:
        raise HTTPException(status_code=400, detail="Email já cadastrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/prompt", response_model=ResponseModel)
async def read_prompt(
    question: str = Form(...),
    code: str = Form(...),
) -> ResponseModel:
    if not code_confirmation(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    try:
        speech_file_path, prompt_response = build_chain(question, history)
        if not speech_file_path:
            return ResponseModel(response=prompt_response, audio=None)

        with open(speech_file_path, 'rb') as f:
            wav_data = f.read()
        return ResponseModel(
            response=prompt_response,
            audio=wav_data.decode("latin1")
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sql")
def read_sql(question: Question):
    if not code_confirmation(question.code):
        raise HTTPException(status_code=400, detail="Invalid code")

    try:
        text = question.question
        response = build_sql_chain(text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/route")
async def read_route(
    question: str = Form(...),
    code: str = Form(...),
    file: UploadFile = File(None)
):
    if not code_confirmation(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    try:
        if file is None:
            response = route_request(question)
            return response
        file_bytes = await file.read()
        response = route_request(question, file_bytes)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_file") #incomplete
async def read_route(
    question: str = Form(...),
    code: str = Form(...),
    file: UploadFile = File()
):
    if not code_confirmation(code):
        raise HTTPException(status_code=400, detail="Invalid code")
    try:
        file_bytes = await file.read()
        response = insertDocsInVectorDatabase(file_bytes)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
