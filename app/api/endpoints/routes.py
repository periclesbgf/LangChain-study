# endpoints/routes.py

from api.endpoints.models import Question, PromptModel, ResponseModel, RegisterModel, LoginModel, Token
from api.controllers.controller import (
    code_confirmation,
    build_chain,
    build_sql_chain,
    route_request,
    #insertDocsInVectorDatabase
    )
from api.dispatchers.login_dispatcher import CredentialsDispatcher
from database.sql_database_manager import DatabaseManager, session, metadata
from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends, HTTPException
from fastapi.logger import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy import insert
from fastapi.security import OAuth2PasswordRequestForm
from api.controllers.auth import create_access_token, get_current_user
from utils import SECRET_EDUCATOR_CODE
from database.mongo_database_manager import MongoDatabaseManager
from datetime import datetime, timezone


router = APIRouter()


@router.post("/create_account")
async def create_account(
    register_model: RegisterModel,
):
    print("Creating account")
    try:
        # Inicializa o gerenciador SQL
        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = CredentialsDispatcher(sql_database_manager)
        
        print("connecting to database")

        # Verifica se o código especial está correto para educadores
        if register_model.tipo_usuario == "educator":
            if register_model.special_code != SECRET_EDUCATOR_CODE:
                raise HTTPException(status_code=400, detail="Invalid special code")

        # Cria a conta no banco de dados SQL
        sql_database_controller.create_account(
            register_model.nome,
            register_model.email,
            register_model.senha,
            register_model.tipo_usuario,
            register_model.instituicao,
        )

        # Cria automaticamente o perfil no MongoDB
        await create_profile_in_mongo(register_model.nome, register_model.email)

        return {"message": "Conta e perfil criados com sucesso"}

    except IntegrityError:
        raise HTTPException(status_code=400, detail="Email já cadastrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def create_profile_in_mongo(nome: str, email: str):
    try:
        # Inicializa o gerenciador MongoDB
        mongo_manager = MongoDatabaseManager()

        # Monta os dados do perfil
        profile_data = {
            "Nome": nome,
            "Email": email,
            "EstiloAprendizagem": None,  # Inicialmente como None
            "Feedback": None,
            "PreferenciaAprendizado": None,
            "created_at": datetime.now(timezone.utc)
        }

        # Cria o perfil no MongoDB
        profile_id = await mongo_manager.create_student_profile(
            email=email,
            profile_data=profile_data
        )

        if not profile_id:
            raise HTTPException(status_code=500, detail="Erro ao criar perfil.")

        print(f"Perfil criado com ID: {profile_id}")

    except Exception as e:
        print(f"Erro ao criar perfil no MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar perfil: {str(e)}")

@router.post("/login")
async def login(
    login_model: LoginModel,
):
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = CredentialsDispatcher(sql_database_manager)
        print("Tentando login")

        user = sql_database_controller.login(
            login_model.email,
            login_model.senha
        )

        access_token = create_access_token(data={"sub": user.Email})
        print("Login efetuado com sucesso")
        print(access_token)
        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/prompt", response_model=ResponseModel)
async def read_prompt(
    prompt_model: PromptModel,
    current_user: dict = Depends(get_current_user),
) -> ResponseModel:
    print(current_user)
    if not code_confirmation(prompt_model.code):
        print(prompt_model.code)
        print("Invalid code")
        raise HTTPException(status_code=400, detail="Invalid code")

    try:
        speech_file_path, prompt_response = build_chain(prompt_model.question, history)
        print("Prompt response: ", prompt_response)
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

# @router.post("/upload_file") #incomplete
# async def read_route(
#     question: str = Form(...),
#     code: str = Form(...),
#     file: UploadFile = File()
# ):
#     if not code_confirmation(code):
#         raise HTTPException(status_code=400, detail="Invalid code")
#     try:
#         file_bytes = await file.read()
#         response = insertDocsInVectorDatabase(file_bytes)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
