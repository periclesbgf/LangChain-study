from fastapi import APIRouter, HTTPException, Depends
from fastapi.logger import logger
from sqlalchemy.orm import Session
from database.sql_database_manager import DatabaseManager, session, metadata
from api.controllers.auth import get_current_user
from api.controllers.student_controller import StudentController
from api.dispatchers.student_dispatcher import StudentDispatcher
from models import StudentCreate, StudentUpdate

router_student = APIRouter()


@router_student.get("/students")
async def get_all_students(current_user: dict = Depends(get_current_user)):
    logger.info(f"Fetching all students for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = StudentDispatcher(sql_database_manager)
        controller = StudentController(dispatcher)

        # Obter todos os estudantes
        students = controller.get_all_students(current_user['sub'])

        logger.info(f"Students fetched successfully for user: {current_user['sub']}")
        return {"students": students}
    except Exception as e:
        logger.error(f"Error fetching students for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_student.post("/students")
async def create_student(
    student: StudentCreate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Creating new student for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = StudentDispatcher(sql_database_manager)
        controller = StudentController(dispatcher)

        # Criar um novo estudante
        controller.create_student(student.name, student.matricula, current_user['sub'])

        logger.info(f"New student created successfully for user: {current_user['sub']}")
        return {"message": "Student created successfully"}
    except Exception as e:
        logger.error(f"Error creating student for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_student.put("/students/{student_id}")
async def update_student(
    student_id: int,
    student_data: StudentUpdate,
    current_user: dict = Depends(get_current_user)
):
    logger.info(f"Updating student {student_id} for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = StudentDispatcher(sql_database_manager)
        controller = StudentController(dispatcher)

        # Atualizar o estudante
        controller.update_student(student_id, student_data.name, student_data.matricula, current_user['sub'])

        logger.info(f"Student {student_id} updated successfully for user: {current_user['sub']}")
        return {"message": "Student updated successfully"}
    except Exception as e:
        logger.error(f"Error updating student {student_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router_student.delete("/students/{student_id}")
async def delete_student(student_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"Deleting student {student_id} for user: {current_user['sub']}")
    try:
        # Instanciar o DatabaseManager
        sql_database_manager = DatabaseManager(session, metadata)

        dispatcher = StudentDispatcher(sql_database_manager)
        controller = StudentController(dispatcher)

        # Deletar o estudante
        controller.delete_student(student_id, current_user['sub'])

        logger.info(f"Student {student_id} deleted successfully for user: {current_user['sub']}")
        return {"message": "Student deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting student {student_id} for user: {current_user['sub']} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
