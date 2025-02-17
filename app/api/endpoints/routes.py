# endpoints/routes.py

from typing import Dict, Optional
from api.endpoints.models import GoogleLoginRequest, Question, ResponseModel, RegisterModel, LoginModel, Token, AudioResponseModel
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
from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends, HTTPException, Request # Import Request e Session
from fastapi.logger import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy import insert
from fastapi.security import OAuth2PasswordRequestForm
from api.controllers.auth import create_access_token, get_current_user, create_google_flow, credentials_to_dict, dict_to_credentials # Import funções auth
from utils import SECRET_EDUCATOR_CODE
from database.mongo_database_manager import MongoDatabaseManager
from datetime import datetime, timezone
from agent.calendar_agent import CalendarAgent, CalendarOrchestrator
from chains.chain_setup import ClassificationChain
from audio.text_to_speech import AudioService
from utils import OPENAI_API_KEY
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
from api.controllers.classroom_api_client import ClassroomAPIClient  # Importe ClassroomAPIClient
import os
import json


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
    try:
        if register_model.senha == "" or register_model.senha == None:
            raise HTTPException(status_code=400, detail="Senha não pode ser vazia")

        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = CredentialsDispatcher(sql_database_manager)
        mongo_manager = MongoDatabaseManager()

        if register_model.tipo_usuario == "student":
            sql_database_controller.create_account(
                name=register_model.nome,
                email=register_model.email,
                password=register_model.senha,
                user_type=register_model.tipo_usuario,
                matricula=register_model.matricula,
                instituicao=register_model.instituicao
            )
            profile_data = {
                "Nome": register_model.nome,
                "Email": register_model.email,
                "EstiloAprendizagem": None,  # Inicialmente como None
                "Feedback": None,
                "PreferenciaAprendizado": None,
                "created_at": datetime.now(timezone.utc)
            }

            profile_id = await mongo_manager.create_student_profile(
                email=register_model.email,
                profile_data=profile_data
            )

        elif register_model.tipo_usuario == "educator":
            if register_model.special_code != SECRET_EDUCATOR_CODE:
                raise HTTPException(status_code=400, detail="Código especial inválido")

            sql_database_controller.create_account(
                name=register_model.nome,
                email=register_model.email,
                password=register_model.senha,
                user_type=register_model.tipo_usuario,
                matricula=None,
                instituicao=register_model.instituicao
            )
        else:
            raise HTTPException(status_code=400, detail="Tipo de usuário inválido.")

        return {"message": "Conta e perfil criados com sucesso"}

    except HTTPException as e:
        if e.status_code == 409:
            raise HTTPException(status_code=e.status_code, detail="Email já cadastrado.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def login(
    login_model: LoginModel,
):
    try:
        if login_model.senha == "" or login_model.senha == None:
            raise HTTPException(status_code=400, detail="Senha não pode ser vazia")

        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = CredentialsDispatcher(sql_database_manager)
        print("Tentando login")

        user = sql_database_controller.login(
            login_model.email,
            login_model.senha
        )
        print("Login efetuado")
        print(user)

        access_token = create_access_token(data={"sub": user.Email})
        print("Login efetuado com sucesso")
        print(access_token)
        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException as e:
        if e.status_code == 401:
            raise HTTPException(status_code=e.status_code, detail="Email ou senha inválidos")
        if e.status_code == 404:
            raise HTTPException(status_code=e.status_code, detail="Email ou senha inválidos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user_email")
def read_user_email(current_user: dict = Depends(get_current_user)):
    return {"user_email": current_user["sub"]}

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


@router.get("/auth/google/initiate")  # Endpoint para iniciar o fluxo OAuth2
async def google_login_initiate(request: Request):
    try:
        flow = create_google_flow(request)
        authorization_url, state = flow.authorization_url(prompt='consent')  # 'consent' para forçar o prompt sempre
        print("Authorization URL gerada:", authorization_url)
        print("State gerado:", state)
        return {"auth_url": authorization_url}
    except Exception as e:
        print("Erro no google_login_initiate:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/google/callback")  # Endpoint de callback do Google
async def google_login_callback(request: Request, code: str, session: Session = Depends(DatabaseManager.get_db)):
    try:
        print("Callback do Google recebido com code:", code)
        flow = create_google_flow(request)
        flow.fetch_token(code=code)
        print("Flow credentials após fetch_token:", flow.credentials)

        credentials_obj = flow.credentials
        id_token = credentials_obj.id_token
        print("ID Token (raw):", id_token)

        from jose import jwt
        user_info = jwt.get_unverified_claims(id_token)
        print("User info decodificado:", user_info)

        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = CredentialsDispatcher(sql_database_manager)
        user = await sql_database_controller.google_login(user_info)
        print("Usuário retornado do google_login:", user)

        access_token_jwt = create_access_token(data={"sub": user.Email})
        print("JWT de acesso criado:", access_token_jwt)

        request.session['google_credentials'] = credentials_to_dict(credentials_obj)
        print("Credenciais do Google armazenadas na sessão:", request.session['google_credentials'])

        # *** Handler para buscar cursos e SALVAR em arquivo JSON ***
        credentials_dict = request.session.get('google_credentials') # Recupera novamente as credenciais da sessão
        creds = dict_to_credentials(credentials_dict)
        classroom_client = ClassroomAPIClient(creds) # Instancia ClassroomAPIClient com as credenciais

        try:
            classroom_courses_response = classroom_client.list_courses() # Chama a função para listar cursos

            # *** Salvar a resposta em arquivo JSON ***
            filename = "classroom_courses_response.json" # Nome do arquivo JSON
            filepath = os.path.join(".", filename) # Salvar na raiz do projeto (pode ajustar o caminho se necessário)

            try:
                with open(filepath, 'w', encoding='utf-8') as f: # Abre o arquivo para escrita com encoding UTF-8
                    json.dump(classroom_courses_response, f, ensure_ascii=False, indent=4) # Salva a resposta em JSON formatado
                print(f"\n*** Resposta da API do Google Classroom (cursos) SALVA em: {filepath} ***")
            except Exception as e_save_json:
                print(f"Erro ao salvar resposta em arquivo JSON: {e_save_json}")

            classroom_courses_materials_response = classroom_client.list_course_materials(classroom_courses_response.get("courses")[0].get("id"))
            print(classroom_courses_materials_response)
            filename = "classroom_courses_materials_response.json" # Nome do arquivo JSON
            filepath = os.path.join(".", filename) # Salvar na raiz do projeto (pode ajustar o caminho se necessário)

            try:
                with open(filepath, 'w', encoding='utf-8') as f: # Abre o arquivo para escrita com encoding UTF-8
                    json.dump(classroom_courses_materials_response, f, ensure_ascii=False, indent=4) # Salva a resposta em JSON formatado
                print(f"\n*** Resposta da API do Google Classroom (cursos) SALVA em: {filepath} ***")
            except Exception as e_save_json:
                print(f"Erro ao salvar resposta em arquivo JSON: {e_save_json}")

        except HTTPException as classroom_error:
            print(f"Erro ao buscar cursos do Google Classroom no handler: {classroom_error}")
            # Decide como lidar com o erro aqui - logar, retornar um erro específico ao frontend, etc.
        except Exception as e_classroom:
            print(f"Erro inesperado ao buscar cursos do Google Classroom no handler: {e_classroom}")
            # Lidar com outras exceções inesperadas


        # Redirecionar diretamente para /home-student, passando o access_token
        frontend_redirect_url = f"http://localhost:8080/home-student?accessToken={access_token_jwt}"  # Redireciona para /home-student
        return RedirectResponse(url=frontend_redirect_url, status_code=303)

    except Exception as e:
        print(f"Erro no google_login_callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/me") # Rota de exemplo para obter informações do usuário logado (usando JWT)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"user_info": current_user}


@router.get("/classroom/courses") # Rota para buscar cursos do Classroom
async def get_classroom_courses(request: Request, session: Session = Depends(DatabaseManager.get_db), current_user: dict = Depends(get_current_user)): # Injeta Request, Session e current_user
    credentials_dict = request.session.get('google_credentials') # Recupera credenciais da sessão (DEMONSTRAÇÃO)
    if not credentials_dict:
        raise HTTPException(status_code=401, detail="Credenciais Google não encontradas. Faça login com o Google.")

    creds = dict_to_credentials(credentials_dict) # Desserializa as credenciais

    from googleapiclient.discovery import build
    try:
        classroom_service = build('classroom', 'v1', credentials=creds) # 'v1' é a versão da Classroom API
        courses_result = classroom_service.courses().list().execute()
        courses = courses_result.get('courses', [])
        return {"courses": courses}
    except Exception as e:
        print(f"Erro ao acessar Classroom API: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao acessar a API do Google Classroom: {e}")

