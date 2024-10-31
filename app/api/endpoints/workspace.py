from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, status
from typing import Optional, List
from agent.image_handler import ImageHandler
from database.vector_db import QdrantHandler, Embeddings, Material
from api.controllers.auth import get_current_user
from datetime import datetime
from api.endpoints.models import AccessLevel
from utils import QDRANT_URL, OPENAI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback

router_workspace = APIRouter()

def get_qdrant_handler():
    """Factory para criar instância do QdrantHandler com dependências."""
    try:
        # Inicializa embeddings
        embeddings = Embeddings().get_embeddings()
        
        # Configura image handler
        image_handler = ImageHandler(OPENAI_API_KEY)
        
        # Configura text splitter para chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Cria QdrantHandler
        return QdrantHandler(
            url=QDRANT_URL,
            collection_name="student_documents",
            embeddings=embeddings,
            image_handler=image_handler,
            text_splitter=text_splitter
        )
    except Exception as e:
        print(f"[ERROR] Falha ao criar QdrantHandler: {str(e)}")
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
            
        # Processa material usando QdrantHandler
        qdrant_handler = get_qdrant_handler()
        material = await qdrant_handler.add_material(
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
    """
    try:
        qdrant_handler = get_qdrant_handler()
        materials = await qdrant_handler.get_materials(
            student_email=current_user["sub"],
            discipline_id=discipline_id,
            session_id=session_id
        )
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
            
        qdrant_handler = get_qdrant_handler()
        await qdrant_handler.update_material_access(
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
    """
    try:
        qdrant_handler = get_qdrant_handler()
        await qdrant_handler.delete_material(material_id)
        
    except Exception as e:
        print(f"[ERROR] Falha ao deletar material: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao remover material"
        )