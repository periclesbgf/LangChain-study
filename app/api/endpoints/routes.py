from api.endpoints.models import Question
from api.controller import code_confirmation, build_chain, build_sql_chain
from fastapi import APIRouter, HTTPException, Depends


router = APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/prompt")
def read_prompt(question: Question):

    if not code_confirmation(question.code):
        raise HTTPException(status_code=400, detail="Invalid code")

    try:
        text = question.question
        response = build_chain(text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sql")
def read_sql(question: Question):

    if not code_confirmation(question.code):
        raise HTTPException(status_code=400, detail="Invalid code")

    try:
        text = question.question
        response = build_sql_chain(text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))