from api.endpoints.models import Question, ResponseModel
from api.controller import code_confirmation, build_chain, build_sql_chain, route_request
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from agent.chat import ConversationHistory
from fastapi.logger import logger

history = ConversationHistory()

router = APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/prompt", response_model=ResponseModel)
async def read_prompt(
    question: str = Form(...),
    code: str = Form(...),
) -> ResponseModel:
    if not code_confirmation(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    print("history: ", history.get_history())

    try:
        speech_file_path, prompt_response = build_chain(question, history)
        if not speech_file_path:
            return ResponseModel(response=prompt_response, audio=None)

        print("speech_file_path: ", speech_file_path)
        with open(speech_file_path, 'rb') as f:
            wav_data = f.read()
        print("response: ", prompt_response)
        return ResponseModel(
            response=prompt_response,
            audio=wav_data.decode("latin1")
        )
    except Exception as e:
        print(e)
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
