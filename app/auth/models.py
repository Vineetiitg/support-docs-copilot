from pydantic import BaseModel


class UserContext(BaseModel):
    role: str
    user_id: str


class LoginRequest(BaseModel):
    api_key: str


class LoginResponse(BaseModel):
    role: str
    token_type: str = "api_key"
