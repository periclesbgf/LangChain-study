# api/endpoints/classroom.py
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException, Depends, Request

from api.controllers.auth import get_current_user, dict_to_credentials
from api.controllers.classroom_api_client import ClassroomAPIClient

router_classroom = APIRouter()
classroom_client = ClassroomAPIClient()


@router_classroom.get("/classroom/courses", response_model=Optional[Dict])
async def get_classroom_courses(request: Request, current_user: dict = Depends(get_current_user)):
    """
    Retorna a lista de cursos do Google Classroom do utilizador autenticado.
    """
    credentials_dict = request.session.get('google_credentials')
    if not credentials_dict:
        raise HTTPException(status_code=401, detail="Credenciais Google não encontradas. Faça login com o Google.")

    creds = dict_to_credentials(credentials_dict)

    try:

        classroom_data = classroom_client.list_courses(creds)
        return classroom_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erro ao processar pedido de cursos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar cursos do Google Classroom: {e}")


@router_classroom.get("/classroom/courses/{course_id}/assignments", response_model=Optional[Dict])
async def get_classroom_assignments(course_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Retorna a lista de trabalhos (assignments) de um curso específico do Google Classroom.
    """
    credentials_dict = request.session.get('google_credentials')
    if not credentials_dict:
        raise HTTPException(status_code=401, detail="Credenciais Google não encontradas. Faça login com o Google.")

    creds = dict_to_credentials(credentials_dict)

    try:
        assignments_data = classroom_client.list_course_work(creds, course_id)
        return assignments_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erro ao processar pedido de trabalhos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar trabalhos do curso do Google Classroom: {e}")


@router_classroom.get("/classroom/courses/{course_id}/students", response_model=Optional[Dict])
async def get_classroom_students(course_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Retorna a lista de alunos de um curso específico do Google Classroom (requer permissões de professor/admin).
    """
    credentials_dict = request.session.get('google_credentials')
    if not credentials_dict:
        raise HTTPException(status_code=401, detail="Credenciais Google não encontradas. Faça login com o Google.")

    creds = dict_to_credentials(credentials_dict)

    try:
        students_data = classroom_client.list_students(creds, course_id)
        return students_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erro ao processar pedido de alunos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar alunos do curso do Google Classroom: {e}")


@router_classroom.get("/classroom/courses/{course_id}/materials", response_model=Optional[Dict])
async def get_classroom_course_materials(course_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Retorna a lista de materiais de um curso específico do Google Classroom.
    """
    credentials_dict = request.session.get('google_credentials')
    if not credentials_dict:
        raise HTTPException(status_code=401, detail="Credenciais Google não encontradas. Faça login com o Google.")

    creds = dict_to_credentials(credentials_dict)

    try:
        materials_data = classroom_client.list_course_materials(creds, course_id)
        return materials_data
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions
    except Exception as e:
        print(f"Erro ao processar pedido de materiais: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar materiais do curso do Google Classroom: {e}")


@router_classroom.get("/classroom/courses/{course_id}", response_model=Optional[Dict])
async def get_classroom_course_detail(course_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Retorna detalhes de um curso específico do Google Classroom.
    """
    credentials_dict = request.session.get('google_credentials')
    if not credentials_dict:
        raise HTTPException(status_code=401, detail="Credenciais Google não encontradas. Faça login com o Google.")

    creds = dict_to_credentials(credentials_dict)
    classroom_client = ClassroomAPIClient(creds) # Instancia ClassroomAPIClient com as credenciais

    try:
        course_detail = classroom_client.get_course(course_id) # Usa o novo método get_course
        return course_detail
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions para serem tratadas pelo FastAPI
    except Exception as e:
        print(f"Erro ao processar pedido de detalhes do curso: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter detalhes do curso do Google Classroom com ID: {course_id}: {e}")