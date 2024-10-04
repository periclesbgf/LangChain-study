from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from api.endpoints.routes import router
from api.endpoints.study_sessions import router_study_sessions
from api.endpoints.calendar import router_calendar
from api.endpoints.discipline import router_disciplines

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(router_study_sessions)
app.include_router(router_calendar)
app.include_router(router_disciplines)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
