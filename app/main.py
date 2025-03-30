from fastapi import FastAPI, Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
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
from api.endpoints.support import router_support
from api.endpoints.files import router_pdf
from utils import SECRET_KEY
from logg import logger

load_dotenv()

app = FastAPI()

# Classe middleware personalizada para rastrear requisições
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        path = request.url.path
        method = request.method

        if not (path == "/docs" or path == "/redoc" or path == "/openapi.json" or path == "/favicon.ico"):
            logger.info(f"[REQUEST] {method} {path} completado em {process_time:.2f}s")

        return response

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

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
app.include_router(router_pdf)
app.include_router(router_support)

logger.info("[STARTUP] Aplicação FastAPI iniciada com logging para Loki.")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
