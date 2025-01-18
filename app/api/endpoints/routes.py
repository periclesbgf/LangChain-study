# endpoints/routes.py

from typing import Dict, Optional
from api.endpoints.models import Question, PromptModel, ResponseModel, RegisterModel, LoginModel, Token, AudioResponseModel
from api.controllers.controller import (
    code_confirmation,
    build_chain,
    build_sql_chain,
    route_request,
    #insertDocsInVectorDatabase
    )
from api.dispatchers.login_dispatcher import CredentialsDispatcher
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
from api.controllers.calendar_controller import CalendarController
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
from agent.calendar_agent import CalendarAgent, CalendarOrchestrator
from chains.chain_setup import ClassificationChain
from audio.text_to_speech import AudioService
from utils import OPENAI_API_KEY

router = APIRouter()



@router.post("/webrtc-audio", response_model=ResponseModel)
async def process_webrtc_audio(
    audio_file: UploadFile = File(...),
) -> ResponseModel:
    """
    Recebe um arquivo de áudio via WebRTC, processa a entrada e retorna uma resposta de áudio e texto.
    """
    try:
        file_path = f"audio/{audio_file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        print(f"[INFO] Audio file received: {audio_file.filename}")

        audio_service = AudioService()
        transcribed_text = audio_service.speech_to_text(file_path)

        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")

        classification_chain = ClassificationChain(api_key=OPENAI_API_KEY)
        route = classification_chain.setup_chain(text=transcribed_text)

        if route == "calendario":
            db_manager = DatabaseManager(session, metadata)
            calendar_dispatcher = CalendarDispatcher(db_manager)
            calendar_orchestrator = CalendarOrchestrator(calendar_dispatcher)

            # Processar a entrada
            result = await calendar_orchestrator.process_input(
                text_input=transcribed_text,
                user_email="pbgf@1234"
            )
            print(f"[DEBUG] Orchestrator result: {result}")
            messages = result.get('messages', [])
            if messages and len(messages) >= 2:
                prompt_response = messages[-1].content
            else:
                prompt_response = "Desculpe, não entendi."
        else:
            prompt_response = build_chain(text=transcribed_text)

        speech_file_path = audio_service.text_to_speech(prompt_response)
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


@router.post("/question")
async def read_question(
    question: str,
    user_email: str = "pbgf@1234"
) -> Dict:
    """
    Processa perguntas e comandos relacionados ao calendário.
    
    Args:
        question: Texto da pergunta ou comando do usuário
        user_email: Email do usuário (idealmente vindo da autenticação)
        
    Returns:
        Dict contendo a resposta processada e detalhes da operação
    """
    try:
        print(f"[INFO] Processing calendar question: {question}")
        print(f"[INFO] User email: {user_email}")
    
        # Inicializar todos os componentes dentro do endpoint
        print("[INIT] Initializing calendar components...")
        
        # Database e Dispatcher
        db_manager = DatabaseManager(session, metadata)
        calendar_dispatcher = CalendarDispatcher(db_manager)
        calendar_controller = CalendarController(calendar_dispatcher)
        
        # Calendar Orchestrator
        calendar_orchestrator = CalendarOrchestrator(calendar_controller)
        print("[INIT] Calendar components initialized successfully")

        # Processar a entrada do usuário
        result = await calendar_orchestrator.process_input(
            text_input=question,
            user_email=user_email
        )
        print(f"[DEBUG] Orchestrator result: {result}")

        # Verificar se houve erro no processamento
        if "error" in result:
            print(f"[ERROR] Processing error: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=result['text_response']
            )

        # Formatar resposta
        response = {
            "message": result['text_response'],
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "operation_result": result.get('operation_result'),
                "plan": result.get('response_plan')
            }
        }
        
        print(f"[INFO] Calendar response: {response['message']}")
        
        # Limpar recursos
        db_manager.session.close()
        
        return response
        
    except HTTPException as e:
        # Repassar exceções HTTP
        if 'db_manager' in locals():
            db_manager.session.close()
        raise e
    except Exception as e:
        error_msg = f"Error processing calendar question: {str(e)}"
        print(f"[ERROR] {error_msg}")
        # Tentar obter mais detalhes do erro
        print(f"[ERROR] Exception type: {type(e).__name__}")
        print(f"[ERROR] Exception args: {e.args}")
        
        # Limpar recursos em caso de erro
        if 'db_manager' in locals():
            db_manager.session.close()
            
        raise HTTPException(
            status_code=500,
            detail={
                "message": error_msg,
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
        )


# @router.post("/prompt", response_model=ResponseModel)
# async def read_prompt(
#     prompt_model: PromptModel,
# ) -> ResponseModel:

#     if not code_confirmation(prompt_model.code):
#         print(prompt_model.code)
#         print("Invalid code")
#         raise HTTPException(status_code=400, detail="Invalid code")

#     try:
#         print(f"[INFO] Processing prompt: {prompt_model.question}")
#         classification_chain = ClassificationChain(api_key=OPENAI_API_KEY)
#         route = classification_chain.setup_chain(text=prompt_model.question)
#         print(f"[INFO] Route: {route}")

#         if route == "calendario":
#             print(f"[INFO] Processing calendar question: {prompt_model.question}")

#             # Inicializar todos os componentes dentro do endpoint
#             print("[INIT] Initializing calendar components...")

#             # Database e Dispatcher
#             db_manager = DatabaseManager(session, metadata)
#             calendar_dispatcher = CalendarDispatcher(db_manager)
#             calendar_controller = CalendarController(calendar_dispatcher)
            

#             # Calendar Orchestrator
#             calendar_orchestrator = CalendarOrchestrator(calendar_controller)
#             print("[INIT] Calendar components initialized successfully")

#             # Processar a entrada do usuário
#             result = await calendar_orchestrator.process_input(
#                 text_input=prompt_model.question,
#                 user_email="pbgf@1234"
#             )
#             print(f"[DEBUG] Orchestrator result: {result}")
#             messages = result.get('messages', [])
#             if messages and len(messages) >= 2:
#                 prompt_response = messages[-1].content
#             else:
#                 prompt_response = "Desculpe, não entendi."
#         else:
#             prompt_response = build_chain(text=prompt_model.question)
#         print("Prompt response: ", prompt_response)
#         audio_service = AudioService()
#         speech_file_path = audio_service.text_to_speech(prompt_response)
#         if not speech_file_path:
#             return ResponseModel(response=prompt_response, audio=None)

#         with open(speech_file_path, 'rb') as f:
#             wav_data = f.read()
#         return ResponseModel(
#             response=prompt_response,
#             audio=None
#         )
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail=str(e))

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
