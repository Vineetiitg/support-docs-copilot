from fastapi import Header, HTTPException

from app.auth.models import UserContext
from app.core.config import settings


def resolve_user(x_api_key: str | None = Header(default=None)) -> UserContext:
    if not settings.AUTH_ENABLED:
        return UserContext(role="admin", user_id="local-dev")
    if x_api_key == settings.ADMIN_API_KEY:
        return UserContext(role="admin", user_id="admin")
    if x_api_key == settings.USER_API_KEY:
        return UserContext(role="user", user_id="user")
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def require_admin(user: UserContext) -> None:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
