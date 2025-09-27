from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.ai_models import AIModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize AI models
    ai_manager = AIModelManager()
    await ai_manager.initialize()
    app.state.ai_manager = ai_manager

    yield

    # Shutdown: Cleanup
    if hasattr(app.state, 'ai_manager'):
        await app.state.ai_manager.cleanup()


def create_application() -> FastAPI:
    app = FastAPI(
        title="CreativeIQ API",
        description="AI-powered design intelligence platform using Vision Language Models",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Mount static files for frontend
    if os.path.exists("frontend/dist"):
        app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

    return app


app = create_application()


@app.get("/")
async def root():
    return {
        "message": "CreativeIQ Design Intelligence API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "creativeiq-api"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )