# dispatchers/login_dispatcher.py

from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_usuarios, tabela_estudantes, tabela_educadores
from datetime import datetime, timezone
from api.controllers.auth import hash_password, verify_password

class CredentialsDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def create_account(self, name: str, email: str, password: str, user_type: str, matricula: str = None, instituicao: str = None, especializacao: str = None):
        password_hash = hash_password(password)
        try:
            print("Inserting user into database")

            # Inserir o usuário na tabela Usuarios
            user_id = self.database_manager.inserir_dado(tabela_usuarios, {
                'Nome': name,
                'Email': email,
                'SenhaHash': password_hash,
                'TipoUsuario': user_type,
                'CriadoEm': datetime.now(timezone.utc),
                'AtualizadoEm': datetime.now(timezone.utc),
            })[0]  # user_id é retornado

            # Verificar o tipo de usuário e inserir nas tabelas apropriadas
            if user_type == 'student':
                print("Inserting student into database")
                # Inserir o estudante na tabela Estudantes
                self.database_manager.inserir_dado(tabela_estudantes, {
                    'IdUsuario': user_id,
                    'Matricula': matricula  # Matricula pode ser None (nulo)
                })

            elif user_type == 'educator':
                if not instituicao or not especializacao:
                    raise HTTPException(status_code=400, detail="Instituição e Especialização são obrigatórias para educadores.")

                print("Inserting educator into database")
                # Inserir o educador na tabela Educadores
                self.database_manager.inserir_dado(tabela_educadores, {
                    'IdUsuario': user_id,
                    'Instituicao': instituicao,
                    'EspecializacaoDisciplina': especializacao
                })

            else:
                raise HTTPException(status_code=400, detail="Tipo de usuário inválido.")

            return True

        except IntegrityError:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=400, detail="Email já cadastrado.")
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.database_manager.session.close()


    def login(self, email: str, password: str):
        user = self.database_manager.get_user_by_email(email)
        print("user: ", user)
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado.")

        if not verify_password(password, user.SenhaHash):
            raise HTTPException(status_code=400, detail="Senha incorreta.")

        return user
