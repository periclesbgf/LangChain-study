from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from api.endpoints.routes import router
from api.endpoints.study_sessions import router_study_sessions
from api.endpoints.calendar import router_calendar
from api.endpoints.discipline import router_disciplines
from api.endpoints.educator import router_educator
from api.endpoints.student import router_student
from api.endpoints.chat import router_chat
from api.endpoints.student_profile import router_profiles
from api.endpoints.plan import router_study_plan
from api.endpoints.workspace import router_workspace
from api.endpoints.websocket_manager import router_websocket
from api.endpoints.files import router_pdf

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", 
        "https://localhost:8080", 
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(router_study_sessions)
app.include_router(router_calendar)
app.include_router(router_disciplines)
app.include_router(router_educator)
app.include_router(router_student)
app.include_router(router_chat)
app.include_router(router_profiles)
app.include_router(router_study_plan)
app.include_router(router_workspace)
app.include_router(router_websocket)
app.include_router(router_pdf)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
