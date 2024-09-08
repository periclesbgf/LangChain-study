from pydantic import BaseModel
from typing import Optional


class Question(BaseModel):
    question: str
    code: str

class ResponseModel(BaseModel):
    response: str
    audio: Optional[str] = None

class LoginModel(BaseModel):
    Email: str
    SenhaHash: str

class Token(BaseModel):
    access_token: str
    token_type: str
