from sqlalchemy.exc import IntegrityError
from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain, RetrievalChain
from database.sql_database_manager import DatabaseManager
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from passlib.context import CryptContext
from sql_test.sql_test_create import tabela_usuarios
from datetime import datetime, timezone

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DatabaseController:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def create_account(self, name: str, email: str, password: str, user_type: str):
        password_hash = pwd_context.hash(password)
        try:
            print("inserting to database")
            self.database_manager.inserir_dado(tabela_usuarios, {
                'Nome': name,
                'Email': email,
                'SenhaHash': password_hash,
                'TipoUsuario': user_type,
                'CriadoEm': datetime.now(timezone.utc),
                'AtualizadoEm': datetime.now(timezone.utc),
            })

            return True

        except IntegrityError:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=400, detail="Email já cadastrado.")
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.database_manager.session.close()
