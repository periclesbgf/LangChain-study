# api/controllers/classroom_api_client.py
from typing import List, Dict, Optional

from fastapi import HTTPException
from googleapiclient.discovery import build
from google.oauth2 import credentials

from api.controllers.constants import CLASSROOM_API_VERSION


class ClassroomAPIClient:
    """
    Classe para encapsular a interação com a API do Google Classroom.
    Inicializada com as credenciais do utilizador.
    """

    def __init__(self, creds: credentials.Credentials):  # Modificado para receber 'creds' no construtor
        """
        Inicializa o cliente da API do Classroom com as credenciais do Google.

        Args:
            creds: Credenciais de autenticação do Google do utilizador.
        """
        self.creds = creds  # Armazena as credenciais como atributo da instância
        self.api_version = CLASSROOM_API_VERSION

    def _build_service(self):  # Método _build_service agora usa self.creds
        """
        Constrói o serviço da API do Google Classroom usando as credenciais da instância.

        Returns:
            Um objeto de serviço da API do Google Classroom.
        """
        try:
            service = build('classroom', self.api_version, credentials=self.creds)  # Usa self.creds
            return service
        except Exception as e:
            print(f"Erro ao construir o serviço da API do Google Classroom: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao construir o serviço da API do Google Classroom: {e}")

    def list_courses(self) -> Optional[Dict]:  # Removido 'creds' como parâmetro
        """
        Lista os cursos do Google Classroom do utilizador.

        Returns:
            Um dicionário contendo a lista de cursos ou None em caso de erro.
        """
        try:
            classroom_service = self._build_service()  # _build_service agora usa self.creds
            courses_result = classroom_service.courses().list().execute()
            courses = courses_result.get('courses', [])
            return {"courses": courses}
        except Exception as e:
            print(f"Erro ao aceder à Classroom API (courses) na classe ClassroomAPIClient: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao listar cursos do Google Classroom: {e}")

    def get_course(self, course_id: str) -> Optional[Dict]:
            """
            Recupera informações de um curso específico do Google Classroom.

            Args:
                course_id: O ID do curso do Google Classroom.

            Returns:
                Um dicionário contendo as informações do curso ou None em caso de erro.
            """
            try:
                classroom_service = self._build_service()
                course_result = classroom_service.courses().get(id=course_id).execute()
                return course_result
            except Exception as e:
                print(f"Erro ao aceder à Classroom API (get course) na classe ClassroomAPIClient: {e}")
                raise HTTPException(status_code=500, detail=f"Erro ao obter informações do curso do Google Classroom com ID: {course_id}: {e}")

    def list_course_work(self, course_id: str) -> Optional[Dict]:  # Removido 'creds' como parâmetro
        """
        Lista os trabalhos de um curso específico do Google Classroom.

        Args:
            course_id: O ID do curso do Google Classroom.

        Returns:
            Um dicionário contendo a lista de trabalhos ou None em caso de erro.
        """
        try:
            classroom_service = self._build_service()  # _build_service agora usa self.creds
            assignments_result = classroom_service.courses().courseWork().list(id=course_id).execute()
            assignments = assignments_result.get('courseWork', [])
            return {"assignments": assignments}
        except Exception as e:
            print(f"Erro ao aceder à Classroom API (trabalhos) na classe ClassroomAPIClient: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao listar trabalhos do curso do Google Classroom: {e}")

    def list_course_materials(self, course_id: str) -> Optional[Dict]:
        """
        Lista os materiais de um curso específico do Google Classroom.

        Args:
            course_id: O ID do curso do Google Classroom.

        Returns:
            Um dicionário contendo a lista de materiais do curso ou None em caso de erro.
        """
        try:
            classroom_service = self._build_service()
            # Corrigido para utilizar courseWorkMaterials() em vez de courseMaterials()
            materials_result = classroom_service.courses().courseWorkMaterials().list(courseId=course_id).execute()
            # Note que a chave do resultado também muda para 'courseWorkMaterials'
            materials = materials_result.get('courseWorkMaterials', [])
            return {"materials": materials}
        except Exception as e:
            print(f"Erro ao aceder à Classroom API (materiais) na classe ClassroomAPIClient: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao listar materiais do curso do Google Classroom: {e}")


    def get_course_work_material(self, course_id: str, material_id: str) -> Optional[Dict]:
        """
        Recupera informações de um material de curso específico do Google Classroom.

        Args:
            course_id: O ID do curso do Google Classroom.
            material_id: O ID do material do curso (courseWorkMaterial).

        Returns:
            Um dicionário contendo as informações do material do curso ou None em caso de erro.
        """
        try:
            classroom_service = self._build_service()
            # Chamada para o endpoint correto da API: courseWorkMaterials().get()
            material_result = classroom_service.courses().courseWorkMaterials().get(
                courseId=course_id,
                id=material_id # 'id' aqui refere-se ao material_id (courseWorkMaterial id)
            ).execute()
            return material_result
        except Exception as e:
            print(f"Erro ao aceder à Classroom API (get courseWorkMaterial) na classe ClassroomAPIClient: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao obter informações do material de curso do Google Classroom com ID: {material_id} do curso {course_id}: {e}")
