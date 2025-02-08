# dispatchers/login_dispatcher.py

from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_usuarios, tabela_estudantes, tabela_educadores, tabela_perfil_aprendizado_aluno, tabela_cursos
from datetime import datetime, timezone
from api.controllers.auth import hash_password, verify_password

class CredentialsDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def create_account(self, name: str, email: str, password: str, user_type: str, matricula: str = None, instituicao: str = None):
        password_hash = hash_password(password)
        try:
            # Iniciando uma transação explícita
            print("Iniciando transação para criação da conta.")
            session = self.database_manager.session
            #session.begin()  # Iniciando uma transação

            print("Inserting user into database")

            # Inserir o usuário na tabela Usuarios e obter o ID gerado
            user_id = self.database_manager.inserir_dado_retorna_id(
                tabela_usuarios,
                {
                    'Nome': name,
                    'Email': email,
                    'SenhaHash': password_hash,
                    'TipoUsuario': user_type,
                    'CriadoEm': datetime.now(timezone.utc),
                    'AtualizadoEm': datetime.now(timezone.utc),
                },
                'IdUsuario'
            )  # Retorna o IdUsuario recém-criado

            print(f"User inserted with ID: {user_id}")

            # Verificar o tipo de usuário e inserir nas tabelas apropriadas
            if user_type == 'student':
                print("Inserting student into database")

                # Inserir o estudante na tabela Estudantes
                student_id = self.database_manager.inserir_dado_retorna_id(
                    tabela_estudantes,
                    {
                        'IdUsuario': user_id,
                        'Matricula': matricula
                    },
                    'IdEstudante'
                )

                if not student_id:
                    raise HTTPException(status_code=400, detail="Erro ao inserir o estudante. Não foi possível criar um perfil de aprendizado.")

                # Criar um perfil de aprendizado com um curso nulo (não associado ainda)
                print(f"Creating learning profile for student ID: {student_id}")
                self.database_manager.inserir_dado(
                    tabela_perfil_aprendizado_aluno,
                    {
                        'IdUsuario': student_id,
                        'DadosPerfil': {},  # Inicialmente vazio, você pode atualizar depois
                        'IdPerfilFelderSilverman': None
                    }
                )
                print("Learning profile created successfully.")

            elif user_type == 'educator':
                print("Inserting educator into database")
                # Inserir o educador na tabela Educadores
                self.database_manager.inserir_dado_retorna_id(
                    tabela_educadores,
                    {
                        'IdUsuario': user_id,
                        'Instituicao': instituicao or "CESAR School",
                    },
                    'IdEducador'
                )
            else:
                raise HTTPException(status_code=400, detail="Tipo de usuário inválido.")

            # Commit da transação ao final de todas as operações bem-sucedidas
            session.commit()
            session.close()
            return {"message": "Conta criada com sucesso"}

        except IntegrityError as e:
            session.rollback()  # Reverter a transação em caso de erro
            print(f"IntegrityError encountered: {e}")
            raise HTTPException(status_code=400, detail="Email já cadastrado.")
        except Exception as e:
            session.rollback()  # Reverter a transação em caso de erro geral
            print(f"General exception encountered: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")


    def login(self, email: str, password: str):
        user = self.database_manager.get_user_by_email(email)
        print("user: ", user)
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado.")

        if not verify_password(password, user.SenhaHash):
            raise HTTPException(status_code=400, detail="Senha incorreta.")

        return user

    async def google_login(self, google_data: dict):
        try:
            # Extrai as informações necessárias do dicionário retornado pelo Google
            email = google_data.get('email')
            name = google_data.get('name')

            print("Tentando buscar usuário pelo email:", email)
            # Procura o usuário no banco de dados usando o email
            user = self.database_manager.get_user_by_email(email)

            if user:
                print("Usuário já existe para o email:", email)
                return user
            else:
                # Se o usuário não existir, cria um novo registro
                dummy_password = "GoogleAccount"  # Senha dummy (fixa)
                hashed_dummy = hash_password(dummy_password)
                print("Criando novo usuário com senha dummy para o email:", email)

                # Insere o usuário na tabela Usuarios
                user_id = self.database_manager.inserir_dado_retorna_id(
                    tabela_usuarios,
                    {
                        'Nome': name,
                        'Email': email,
                        'SenhaHash': hashed_dummy,  # utiliza a senha dummy
                        'TipoUsuario': 'student',   # define o tipo padrão (ou adapte conforme necessário)
                        'CriadoEm': datetime.now(timezone.utc),
                        'AtualizadoEm': datetime.now(timezone.utc),
                    },
                    'IdUsuario'
                )
                print(f"Novo usuário criado com ID: {user_id}")

                # Cria o perfil de estudante (na tabela Estudantes)
                student_id = self.database_manager.inserir_dado_retorna_id(
                    tabela_estudantes,
                    {
                        'IdUsuario': user_id,
                        'Matricula': None  # Pode ser atualizado futuramente
                    },
                    'IdEstudante'
                )
                print(f"Novo estudante criado com ID: {student_id}")

                # Cria o perfil de aprendizado para o estudante
                self.database_manager.inserir_dado(
                    tabela_perfil_aprendizado_aluno,
                    {
                        'IdUsuario': student_id,
                        'DadosPerfil': {},  # Inicialmente vazio; poderá ser atualizado depois
                        'IdPerfilFelderSilverman': None
                    }
                )
                print("Perfil de aprendizado criado com sucesso.")

                # Busca o usuário criado (para retornar o registro completo)
                user = self.database_manager.get_user_by_email(email)
                print("Usuário após criação:", user)
                return user

        except Exception as e:
            print(f"Erro no google_login: {e}")
            raise HTTPException(status_code=500, detail=str(e))