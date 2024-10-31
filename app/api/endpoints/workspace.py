from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, status
from typing import Optional, List
from agent.image_handler import ImageHandler
from database.vector_db import QdrantHandler, Embeddings
from database.sql_database_manager import DatabaseManager
from database.workspace_handler import WorkspaceHandler
from api.controllers.auth import get_current_user
from datetime import datetime
from api.endpoints.models import AccessLevel, Material
from database.mongo_database_manager import MongoDatabaseManager
from utils import QDRANT_URL, OPENAI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback

router_workspace = APIRouter()

def get_workspace_handler():
    """Factory para criar instância do WorkspaceHandler com dependências."""
    try:
        # Inicializa embeddings
        embeddings = Embeddings().get_embeddings()
        
        # Configura Qdrant
        qdrant_handler = QdrantHandler(
            url=QDRANT_URL,
            collection_name="student_documents",
            embeddings=embeddings
        )
        
        # Configura handlers
        image_handler = ImageHandler(OPENAI_API_KEY)
        mongo_manager = MongoDatabaseManager()
        
        # Configura text splitter para chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Cria WorkspaceHandler
        return WorkspaceHandler(
            mongo_manager=mongo_manager,
            qdrant_handler=qdrant_handler,
            image_handler=image_handler,
            text_splitter=text_splitter
        )
    except Exception as e:
        print(f"[ERROR] Falha ao criar WorkspaceHandler: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao inicializar serviços"
        )

@router_workspace.post(
    "/workspace/upload",
    response_model=Material,
    status_code=status.HTTP_201_CREATED
)
async def upload_material(
    file: UploadFile = File(...),
    access_level: AccessLevel = Form(...),
    discipline_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    current_user = Depends(get_current_user)
):
    """
    Upload de novo material com processamento automático de conteúdo.
    
    Args:
        file: Arquivo a ser processado (PDF, DOC, ou imagem)
        access_level: Nível de acesso do material
        discipline_id: ID opcional da disciplina
        session_id: ID opcional da sessão
        
    Returns:
        Material: Objeto com metadados do material processado
    """
    try:
        # Valida tipo do arquivo
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nome do arquivo não fornecido"
            )
            
        # Valida tamanho máximo (10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Arquivo muito grande. Limite de 10MB"
            )
            
        # Valida contexto para níveis específicos
        if access_level == AccessLevel.DISCIPLINE and not discipline_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="discipline_id obrigatório para acesso DISCIPLINE"
            )
            
        if access_level == AccessLevel.SESSION and not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id obrigatório para acesso SESSION"
            )
            
        # Processa material
        workspace_handler = get_workspace_handler()
        material = await workspace_handler.add_material(
            file_content=content,
            filename=file.filename,
            student_email=current_user["sub"],
            access_level=access_level,
            discipline_id=discipline_id,
            session_id=session_id
        )
        
        return material
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"[ERROR] Falha no upload: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar arquivo"
        )

@router_workspace.get(
    "/workspace/materials",
    response_model=List[Material]
)
async def get_materials(
    discipline_id: Optional[str] = None,
    session_id: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """
    Recupera materiais baseado no contexto e nível de acesso.
    
    Args:
        discipline_id: Filtro opcional por disciplina
        session_id: Filtro opcional por sessão
        
    Returns:
        List[Material]: Lista de materiais disponíveis
    """
    try:
        workspace_handler = get_workspace_handler()
        materials = await workspace_handler.get_materials(
            student_email=current_user["sub"],
            discipline_id=discipline_id,
            session_id=session_id
        )
        print(f"[INFO] Materiais recuperados: {materials}")
        print(f"[INFO] Materiais recuperados: {len(materials)}")

        return materials
        
    except Exception as e:
        print(f"[ERROR] Falha ao recuperar materiais: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao recuperar materiais"
        )

@router_workspace.put(
    "/workspace/material/{material_id}/access",
    status_code=status.HTTP_200_OK
)
async def update_material_access(
    material_id: str,
    access_level: AccessLevel,
    discipline_id: Optional[str] = None,
    session_id: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """
    Atualiza nível de acesso de um material.
    
    Args:
        material_id: ID do material
        access_level: Novo nível de acesso
        discipline_id: Nova disciplina opcional
        session_id: Nova sessão opcional
    """
    try:
        # Valida contexto para níveis específicos
        if access_level == AccessLevel.DISCIPLINE and not discipline_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="discipline_id obrigatório para acesso DISCIPLINE"
            )
            
        if access_level == AccessLevel.SESSION and not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id obrigatório para acesso SESSION"
            )
            
        workspace_handler = get_workspace_handler()
        await workspace_handler.update_material_access(
            material_id=material_id,
            new_access_level=access_level,
            discipline_id=discipline_id,
            session_id=session_id
        )
        
        return {"message": "Material access updated successfully"}
        
    except Exception as e:
        print(f"[ERROR] Falha ao atualizar acesso: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao atualizar acesso do material"
        )

@router_workspace.delete(
    "/workspace/material/{material_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_material(
    material_id: str,
    current_user = Depends(get_current_user)
):
    """
    Remove um material e seus dados associados.
    
    Args:
        material_id: ID do material a ser removido
    """
    try:
        workspace_handler = get_workspace_handler()
        await workspace_handler.delete_material(material_id)
        
    except Exception as e:
        print(f"[ERROR] Falha ao deletar material: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao remover material"
        )