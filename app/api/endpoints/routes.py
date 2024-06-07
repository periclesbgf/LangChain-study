from api.endpoints.models import Question
from api.controller import code_confirmation, build_chain, build_sql_chain, route_request
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from agent.chat import ConversationHistory

history = ConversationHistory()

router = APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/prompt")
def read_prompt(
    question: str = Form(...),
    code: str = Form(...),
):
    if not code_confirmation(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    #history = ConversationHistory()
    print("history: ", history.get_history())
    try:
        response = build_chain(question, history)
        return response, None
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

@router.post("/route")
async def read_route(
    question: str = Form(...),
    code: str = Form(...),
    file: UploadFile = File(None)
):
    if not code_confirmation(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    try:
        if file is None:
            response = route_request(question)
            return response
        file_bytes = await file.read()
        response = route_request(question, file_bytes)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))