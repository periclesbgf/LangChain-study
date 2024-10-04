from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from sql_test.sql_test_create import tabela_estudantes
from database.sql_database_manager import DatabaseManager

class StudentDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get_all_students(self, current_user: str):
        try:
            # Buscar todos os estudantes
            students = self.database_manager.selecionar_dados(tabela_estudantes)

            # Transformar o resultado em uma lista de dicionários
            student_list = [
                {
                    "IdEstudante": student.IdEstudante,
                    "Nome": student.Nome,
                    "Matricula": student.Matricula,
                    "IdUsuario": student.IdUsuario
                }
                for student in students
            ]
            return student_list

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching students: {e}")

    def create_student(self, student_data: dict):
        try:
            # Criar um novo estudante no banco de dados
            self.database_manager.inserir_dado(tabela_estudantes, student_data)
            return True
        except IntegrityError:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=400, detail="Student already exists.")
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating student: {e}")

    def update_student(self, student_id: int, updated_data: dict, current_user: str):
        try:
            # Atualizar o estudante no banco de dados
            self.database_manager.atualizar_dado(
                tabela_estudantes,
                tabela_estudantes.c.IdEstudante == student_id,
                updated_data
            )
            return True
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating student: {e}")

    def delete_student(self, student_id: int, current_user: str):
        try:
            # Deletar o estudante do banco de dados
            self.database_manager.deletar_dado(
                tabela_estudantes,
                tabela_estudantes.c.IdEstudante == student_id
            )
            return True
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error deleting student: {e}")
