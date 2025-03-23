from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, status
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import hashlib
import traceback
from fastapi.responses import StreamingResponse
from io import BytesIO

# Supondo que o AccessLevel seja um Enum definido em api.endpoints.models
from api.endpoints.models import AccessLevel
from api.controllers.auth import get_current_user

# Importe o gerenciador do Mongo e o handler para PDFs
from database.mongo_database_manager import MongoDatabaseManager, MongoPDFHandler

router_pdf = APIRouter()

def get_pdf_handler() -> MongoPDFHandler:
    """
    Factory para criar uma instância do MongoPDFHandler.
    """
    try:
        mongo_manager = MongoDatabaseManager()
        return MongoPDFHandler(mongo_manager)
    except Exception as e:
        print(f"[ERROR] Falha ao criar MongoPDFHandler: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao inicializar serviço de PDF"
        )

@router_pdf.post(
    "/pdf/upload",
    status_code=status.HTTP_201_CREATED
)
async def upload_pdf(
    file: UploadFile = File(...),
    access_level: AccessLevel = Form(...),
    discipline_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload de um novo PDF, armazenando-o no MongoDB.
    Valida o tipo de arquivo, tamanho (máximo 10MB) e contexto de acesso.
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nome do arquivo não fornecido"
            )

        # Valida se o arquivo é PDF
        if not (file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O arquivo enviado não é um PDF"
            )

        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Arquivo muito grande. Limite de 10MB"
            )

        # Valida o contexto de acesso
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

        pdf_handler = get_pdf_handler()
        pdf_uuid = str(uuid.uuid4())
        content_hash = hashlib.md5(content).hexdigest()

        stored_pdf = await pdf_handler.store_pdf(
            pdf_uuid=pdf_uuid,
            pdf_bytes=content,
            student_email=current_user["sub"],
            disciplina=discipline_id if discipline_id else "",
            session_id=session_id if session_id else "",
            filename=file.filename,
            content_hash=content_hash,
            access_level=access_level.value  # Supondo que AccessLevel seja um Enum com atributo .value
        )

        # Opcional: remova o campo binário antes de retornar para não sobrecarregar a resposta
        if stored_pdf:
            stored_pdf["pdf_data"] = None

        return stored_pdf

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"[ERROR] Falha no upload de PDF: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar PDF"
        )

@router_pdf.get(
    "/pdf/materials",
    response_model=List[Dict[str, Any]]
)
async def get_pdf_materials(
    discipline_id: Optional[str] = None,
    session_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Recupera os PDFs armazenados no MongoDB para o usuário atual.
    É possível filtrar por discipline_id e session_id.
    """
    try:
        pdf_handler = get_pdf_handler()
        query = {"student_email": current_user["sub"]}
        if discipline_id:
            query["disciplina"] = discipline_id
        if session_id:
            query["session_id"] = session_id

        # Consulta a coleção 'pdf_file'
        cursor = pdf_handler.mongo_manager.db['pdf_file'].find(query)
        pdfs = []
        async for doc in cursor:
            # Opcional: remova o campo binário da resposta
            doc["pdf_data"] = None
            pdfs.append(doc)
        return pdfs

    except Exception as e:
        print(f"[ERROR] Falha ao recuperar PDFs: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao recuperar PDFs"
        )

@router_pdf.get(
    "/pdf/material/{pdf_id}",
    response_class=StreamingResponse,
    summary="Recupera um PDF para visualização",
    description="Retorna o conteúdo do PDF para que o usuário possa visualizá-lo."
)
async def get_pdf_by_id(
    pdf_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Recupera um PDF armazenado no MongoDB com base no seu ID e no email do usuário.
    Retorna o conteúdo binário do PDF utilizando StreamingResponse.
    """
    try:
        pdf_handler = get_pdf_handler()
        # Busca o documento que pertence ao usuário atual
        pdf_doc = await pdf_handler.pdf_collection.find_one({
            "_id": pdf_id,
            "student_email": current_user["sub"]
        })
        if not pdf_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF não encontrado"
            )
        
        # Obtém os dados binários do PDF
        pdf_bytes = pdf_doc.get("pdf_data")
        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dados do PDF não disponíveis"
            )

        filename = pdf_doc.get("filename", "arquivo.pdf")
        headers = {"Content-Disposition": f"inline; filename={filename}"}

        # Retorna o PDF como StreamingResponse
        return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)

    except Exception as e:
        print(f"[ERROR] Falha ao recuperar PDF: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao recuperar PDF"
        )

@router_pdf.put(
    "/pdf/material/{pdf_id}/access",
    status_code=status.HTTP_200_OK
)
async def update_pdf_access(
    pdf_id: str,
    access_level: AccessLevel = Form(...),
    discipline_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Atualiza o nível de acesso de um PDF armazenado.
    Valida o contexto conforme o nível de acesso definido.
    """
    try:
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

        pdf_handler = get_pdf_handler()
        update_fields = {"access_level": access_level.value}
        if discipline_id:
            update_fields["disciplina"] = discipline_id
        if session_id:
            update_fields["session_id"] = session_id

        result = await pdf_handler.pdf_collection.update_one(
            {"_id": pdf_id, "student_email": current_user["sub"]},
            {"$set": update_fields}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF não encontrado ou nenhum campo atualizado"
            )

        return {"message": "Nível de acesso atualizado com sucesso"}

    except Exception as e:
        print(f"[ERROR] Falha ao atualizar acesso do PDF: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao atualizar acesso do PDF"
        )

@router_pdf.delete(
    "/pdf/material/{pdf_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_pdf_material(
    pdf_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Remove um PDF armazenado no MongoDB.
    """
    try:
        pdf_handler = get_pdf_handler()
        result = await pdf_handler.pdf_collection.delete_one(
            {"_id": pdf_id, "student_email": current_user["sub"]}
        )
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF não encontrado"
            )
        return  # Retorna 204 No Content

    except Exception as e:
        print(f"[ERROR] Falha ao deletar PDF: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao remover PDF"
        )
