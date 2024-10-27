# app/api/workspace.py
from fastapi import APIRouter, Depends, File, UploadFile, Form
from typing import Optional
from agent.image_handler import ImageHandler
from database.vector_db import QdrantHandler, Embeddings
from database.sql_database_manager import DatabaseManager
from database.workspace_handler import WorkspaceHandler
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from datetime import datetime, timedelta
from api.controllers.auth import get_current_user
from sqlalchemy import text
import json
from datetime import datetime
from qdrant_client.http import models
from endpoints.models import AccessLevel, Material
from database.mongo_database_manager import MongoDatabaseManager
from utils import QDRANT_URL, OPENAI_API_KEY
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
)

router_workspace = APIRouter()

@router_workspace.post("/workspace/upload")
async def upload_material(
    file: UploadFile = File(...),
    access_level: AccessLevel = Form(...),
    discipline_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    current_user = Depends(get_current_user)
):
    embeddings = Embeddings().get_embeddings()
    qdrant_handler = QdrantHandler(
        url=QDRANT_URL,
        collection_name="student_documents",
        embeddings=embeddings
    )
    image_handler = ImageHandler(OPENAI_API_KEY)

    mongo_manager = MongoDatabaseManager()
    qdrant_handler = QdrantHandler()
    content = await file.read()
    workspace_handler = WorkspaceHandler(mongo_manager, qdrant_handler, image_handler)
    
    material = await workspace_handler.add_material(
        file_content=content,
        filename=file.filename,
        student_email=current_user["sub"],
        access_level=access_level,
        discipline_id=discipline_id,
        session_id=session_id
    )
    
    return material

@router_workspace.get("/workspace/materials")
async def get_materials(
    discipline_id: Optional[str] = None,
    session_id: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    workspace_handler = WorkspaceHandler(mongo_manager, qdrant_handler)
    materials = await workspace_handler.get_materials(
        student_email=current_user["sub"],
        discipline_id=discipline_id,
        session_id=session_id
    )
    return materials

@router_workspace.put("/workspace/material/{material_id}/access")
async def update_material_access(
    material_id: str,
    access_level: AccessLevel,
    discipline_id: Optional[str] = None,
    session_id: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    workspace_handler = WorkspaceHandler(mongo_manager, qdrant_handler)
    await workspace_handler.update_material_access(
        material_id=material_id,
        new_access_level=access_level,
        discipline_id=discipline_id,
        session_id=session_id
    )
    return {"message": "Material access updated successfully"}