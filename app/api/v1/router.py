from fastapi import APIRouter
from app.api.v1 import analysis, chat, health

api_router = APIRouter()

api_router.include_router(analysis.router, prefix="/analyze", tags=["analysis"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(health.router, prefix="/health", tags=["health"])