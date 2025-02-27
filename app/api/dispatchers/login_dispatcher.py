# dispatchers/login_dispatcher.py

from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_interface.sql_tables import tabela_usuarios, tabela_estudantes, tabela_educadores, tabela_perfil_aprendizado_aluno, tabela_cursos
from datetime import datetime, timezone
from api.controllers.auth import (
    hash_password,
    verify_password,
    create_reset_token,
    decode_reset_token,
    )
from database.mongo_database_manager import MongoDatabaseManager

class CredentialsDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def create_account(self, name: str, email: str, password: str, user_type: str, 
                       matricula: str = None, instituicao: str = None):
        # Realiza o hash da senha fornecida
        password_hash = hash_password(password)
        try:
            session = self.database_manager.session

            # Monta o dicionário de dados para inserção em Usuarios
            user_data = {
                'Nome': name,
                'Email': email,
                'SenhaHash': password_hash,
                'TipoUsuario': user_type,
                'TipoDeConta': 'email',
                'CriadoEm': datetime.now(timezone.utc),
                'AtualizadoEm': datetime.now(timezone.utc)
            }

            if instituicao:
                user_data['Instituicao'] = instituicao

            user_id = self.database_manager.inserir_dado_retorna_id(
                tabela_usuarios,
                user_data,
                'IdUsuario'
            )
            print(f"Usuário inserido com ID: {user_id}")

            if user_type == 'student':
                print("Inserindo registro de estudante no banco de dados.")
                student_id = self.database_manager.inserir_dado_retorna_id(
                    tabela_estudantes,
                    {
                        'IdUsuario': user_id,
                        'Matricula': matricula
                    },
                    'IdEstudante'
                )
                if not student_id:
                    raise HTTPException(
                        status_code=400,
                        detail="Erro ao inserir o estudante. Não foi possível criar o perfil de aprendizado."
                    )
                print("Registro de estudante criado com sucesso.")

                # Cria o perfil de aprendizado para o estudante (assegure que o relacionamento use o user_id)
                self.database_manager.inserir_dado(
                    tabela_perfil_aprendizado_aluno,
                    {
                        'IdUsuario': user_id,
                        'DadosPerfil': {},  # Inicialmente vazio; pode ser atualizado posteriormente
                        'IdPerfilFelderSilverman': None
                    }
                )
                print("Perfil de aprendizado criado com sucesso.")

            # Para educadores, não há inserção adicional em outra tabela, pois os dados específicos já estão em Usuarios
            session.commit()
            session.close()
            return {"message": "Conta criada com sucesso"}

        except IntegrityError as e:
            session.rollback()
            raise HTTPException(status_code=409, detail="Email já cadastrado.")
        except HTTPException as e:
            session.rollback()
            raise e
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail="Internal server error.")

    def login(self, email: str, password: str):
        user = self.database_manager.get_user_by_email(email)
        print("user: ", user)
        if not user:
            raise HTTPException(status_code=404, detail="Email ou senha inválidos")

        if not verify_password(password, user.SenhaHash):
            raise HTTPException(status_code=404, detail="Email ou senha inválidos")

        return user

    async def google_login(self, google_data: dict):
        try:
            # Extrai as informações necessárias do dicionário retornado pelo Google
            email = google_data.get('email')
            name = google_data.get('name')

            print("Tentando buscar usuário pelo email:", email)
            user = self.database_manager.get_user_by_email(email)

            if user:
                print("Usuário já existe para o email:", email)
                return user
            else:
                print("Criando novo usuário sem senha (acesso Google) para o email:", email)

                user_id = self.database_manager.inserir_dado_retorna_id(
                    tabela_usuarios,
                    {
                        'Nome': name,
                        'Email': email,
                        'SenhaHash': None,
                        'TipoUsuario': 'student', #possivel bug no futuro
                        'TipoDeConta': 'google',
                        'CriadoEm': datetime.now(timezone.utc),
                        'AtualizadoEm': datetime.now(timezone.utc)
                    },
                    'IdUsuario'
                )
                print(f"Novo usuário criado com ID: {user_id}")

                # Cria o registro do estudante na tabela Estudantes
                student_id = self.database_manager.inserir_dado_retorna_id(
                    tabela_estudantes,
                    {
                        'IdUsuario': user_id,
                        'Matricula': None  # Pode ser atualizado futuramente
                    },
                    'IdEstudante'
                )
                print(f"Novo estudante criado com ID: {student_id}")

                # Cria o perfil de aprendizado para o estudante utilizando o user_id
                self.database_manager.inserir_dado(
                    tabela_perfil_aprendizado_aluno,
                    {
                        'IdUsuario': user_id,
                        'DadosPerfil': {},  # Inicialmente vazio; poderá ser atualizado depois
                        'IdPerfilFelderSilverman': None
                    }
                )
                print("Perfil de aprendizado criado com sucesso.")

                # Busca e retorna o usuário criado (registro completo)
                user = self.database_manager.get_user_by_email(email)
                print("Usuário após criação:", user)

                profile_data = {
                    "Nome": name,
                    "Email": email,
                    "EstiloAprendizagem": None,
                    "Feedback": None,
                    "PreferenciaAprendizado": None,
                    "created_at": datetime.now(timezone.utc)
                }

                mongo_manager = MongoDatabaseManager()
                await mongo_manager.create_student_profile(
                    email=email,
                    profile_data=profile_data
                )

                return user

        except Exception as e:
            print(f"Erro no google_login: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def generate_reset_token(self, user_email: str):
        user = self.database_manager.get_user_by_email(user_email)

        if not user or (user.TipoDeConta == 'google' and user.SenhaHash is None):
            raise HTTPException(
                status_code=400,
                detail="Se o email informado estiver cadastrado, você receberá um email com instruções para redefinir sua senha."
            )

        return create_reset_token(user_email)

    def reset_password(self, email: str, new_password: str):
        user_obj = self.database_manager.get_user_by_email(email)
        if not user_obj:
            raise HTTPException(status_code=404, detail="Email não encontrado")

        password_hash = hash_password(new_password)
        is_updated = self.database_manager.update_user_password(email, password_hash)
        return is_updated
