# endpoints/routes.py

from typing import Dict, Optional
from api.endpoints.models import (
    GoogleLoginRequest,
    Question,
    ResponseModel,
    RegisterModel,
    LoginModel,
    Token,
    AudioResponseModel,
    ResetPasswordModel,
    ForgotPasswordModel,
    RefreshTokenRequest,
    )
from api.controllers.controller import (
    code_confirmation,
    build_chain,
    build_sql_chain,
    route_request,
    #insertDocsInVectorDatabase
    )
from logg import logger
from api.dispatchers.login_dispatcher import CredentialsDispatcher
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
from api.controllers.calendar_controller import CalendarController
from database.sql_database_manager import DatabaseManager, session, metadata
from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends, HTTPException, Request

from api.controllers.auth import (
    create_access_token,
    create_refresh_token,
    get_current_user,
    create_google_flow,
    credentials_to_dict,
    dict_to_credentials,
    decode_reset_token,
    send_reset_email
    )
from jose import JWTError, jwt
from api.controllers.constants import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from utils import SECRET_EDUCATOR_CODE
from database.mongo_database_manager import MongoDatabaseManager
from datetime import datetime, timedelta, timezone
from agent.calendar_agent import CalendarOrchestrator
from chains.chain_setup import ClassificationChain
from audio.text_to_speech import AudioService
from utils import OPENAI_API_KEY
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
from api.controllers.classroom_api_client import ClassroomAPIClient
import os
import json
from utils import APP_URL


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

        logger.info(f"Nova conta: {register_model.email} (tipo: {register_model.tipo_usuario})")

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
                "EstiloAprendizagem": None,
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

        user = sql_database_controller.login(
            login_model.email,
            login_model.senha
        )

        # Criar access token e refresh token
        access_token = create_access_token(data={"sub": user.Email})
        refresh_token = create_refresh_token(data={"sub": user.Email})
        logger.info(f"[LOGIN] Usuário autenticado: {user.Email}")
        return {
            "access_token": access_token, 
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # converter para segundos
        }

    except HTTPException as e:
        if e.status_code == 401:
            raise HTTPException(status_code=e.status_code, detail="Email ou senha inválidos")
        if e.status_code == 404:
            raise HTTPException(status_code=e.status_code, detail="Email ou senha inválidos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        
@router.post("/refresh-token", response_model=Token)
async def refresh_token(request: RefreshTokenRequest):
    """
    Endpoint para renovar o token de acesso usando um refresh token.
    Recebe o refresh token e retorna um novo access token.
    """
    try:
        refresh_token = request.refresh_token
        
        # Decodifica o refresh token
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Verifica se é realmente um refresh token
        if payload.get("token_type") != "refresh":
            raise HTTPException(status_code=401, detail="Token inválido")
        
        # Extrai o email do usuário do token
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Token inválido")
            
        # Gera um novo access token
        new_access_token = create_access_token(data={"sub": email})
        
        logger.info(f"[REFRESH_TOKEN] Token renovado para usuário: {email}")
        
        # Retorna o novo access token
        return {
            "access_token": new_access_token,
            "refresh_token": refresh_token,  # Mantém o mesmo refresh token
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user_email")
def read_user_email(current_user: dict = Depends(get_current_user)):
    return {"user_email": current_user["sub"]}


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
        flow = create_google_flow(request)
        flow.fetch_token(code=code)

        credentials_obj = flow.credentials
        id_token = credentials_obj.id_token

        from jose import jwt
        user_info = jwt.get_unverified_claims(id_token)

        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = CredentialsDispatcher(sql_database_manager)
        user = await sql_database_controller.google_login(user_info)

        access_token_jwt = create_access_token(data={"sub": user.Email})

        request.session['google_credentials'] = credentials_to_dict(credentials_obj)

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
            except Exception as e_save_json:
                logger.error(f"Erro ao salvar resposta em arquivo JSON: {e_save_json}")

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

        frontend_redirect_url = f"{APP_URL}/home-student?accessToken={access_token_jwt}"  # Redireciona para /home-student
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


@router.post("/forgot-password")
async def forgot_password(forgotPassword: ForgotPasswordModel):
    """
    Endpoint para solicitar o reset da senha.
    Se o usuário for autenticado, gera um token de reset e envia por email.
    """
    try:
        sql_database_manager = DatabaseManager(session, metadata)
        sql_database_controller = CredentialsDispatcher(sql_database_manager)
        user_email = forgotPassword.user_email
        reset_token = sql_database_controller.generate_reset_token(user_email)

        if reset_token is None:
            return {
            "message": "Instruções para reset de senha foram enviadas para o seu email."
        }

        send_reset_email(user_email, reset_token)

        return {
            "message": "Instruções para reset de senha foram enviadas para o seu email."
        }
    except Exception as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.detail))


@router.post("/reset-password")
async def reset_password(reset_data: ResetPasswordModel):
    """
    Endpoint para resetar a senha do usuário.
    Recebe o token de reset e a nova senha (com confirmação).
    """
    if reset_data.new_password != reset_data.confirm_password:
        raise HTTPException(status_code=400, detail="As senhas não coincidem.")

    payload = decode_reset_token(reset_data.reset_token)
    if payload is None:
        raise HTTPException(status_code=400, detail="Token inválido ou expirado")

    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=400, detail="Token inválido")

    try:
        sql_database_manager = DatabaseManager(session, metadata)
        credentials_controller = CredentialsDispatcher(sql_database_manager)
        credentials_controller.reset_password(email, reset_data.new_password)

        return {"message": "Senha atualizada com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
