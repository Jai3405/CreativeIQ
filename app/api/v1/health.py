from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse
from app.core.config import settings
import torch

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Comprehensive health check including AI model status
    """
    # Check AI model status
    model_status = "unavailable"
    if hasattr(request.app.state, 'ai_manager'):
        if request.app.state.ai_manager.initialized:
            model_status = "ready"
        else:
            model_status = "initializing"

    return HealthResponse(
        status="healthy",
        service="creativeiq-api",
        model_status=model_status,
        version=settings.VERSION
    )


@router.get("/models")
async def model_status(request: Request):
    """
    Detailed model status and capabilities
    """
    if not hasattr(request.app.state, 'ai_manager'):
        return {"status": "AI manager not available"}

    ai_manager = request.app.state.ai_manager

    return {
        "initialized": ai_manager.initialized,
        "model_name": ai_manager.model_name,
        "device": ai_manager.device,
        "cuda_available": torch.cuda.is_available(),
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    }


@router.get("/capabilities")
async def system_capabilities():
    """
    System capabilities and feature availability
    """
    return {
        "features": {
            "design_analysis": True,
            "color_analysis": True,
            "typography_analysis": True,
            "layout_analysis": True,
            "performance_prediction": True,
            "chat_interface": True,
            "batch_processing": True,
            "variant_generation": True
        },
        "supported_formats": settings.ALLOWED_EXTENSIONS,
        "max_file_size": settings.MAX_FILE_SIZE,
        "max_concurrent_analyses": settings.MAX_CONCURRENT_ANALYSES,
        "analysis_timeout": settings.ANALYSIS_TIMEOUT
    }