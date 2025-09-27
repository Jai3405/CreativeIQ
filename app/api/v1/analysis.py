from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import List
import uuid
import asyncio
from PIL import Image
import io

from app.models.schemas import (
    AnalysisRequest, DesignAnalysisResult, BatchAnalysisRequest, AnalysisStatus
)
from app.services.design_analyzer import DesignAnalyzer
from app.core.config import settings
from app.utils.file_handler import validate_image_file, process_uploaded_image

router = APIRouter()

# Global analyzer instance
analyzer = DesignAnalyzer()


@router.post("/", response_model=DesignAnalysisResult)
async def analyze_single_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = "Analyze this design comprehensively",
    analysis_type: str = "comprehensive",
    target_platform: str = "general"
):
    """
    Analyze a single design image using Vision Language Models
    """
    # Validate file
    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Generate analysis ID
    analysis_id = str(uuid.uuid4())

    try:
        # Process uploaded image
        image = await process_uploaded_image(file)

        # Create analysis request
        request = AnalysisRequest(
            prompt=prompt,
            analysis_type=analysis_type,
            target_platform=target_platform
        )

        # Perform analysis
        result = await analyzer.analyze_design(analysis_id, image, request)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch", response_model=List[DesignAnalysisResult])
async def analyze_batch_images(
    files: List[UploadFile] = File(...),
    requests: List[AnalysisRequest] = None
):
    """
    Analyze multiple images in batch
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    # Validate all files first
    for file in files:
        if not validate_image_file(file):
            raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")

    try:
        # Process all images
        results = []
        for i, file in enumerate(files):
            analysis_id = str(uuid.uuid4())
            image = await process_uploaded_image(file)

            # Use corresponding request or default
            request = requests[i] if requests and i < len(requests) else AnalysisRequest()

            # Analyze image
            result = await analyzer.analyze_design(analysis_id, image, request)
            results.append(result)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/{analysis_id}", response_model=DesignAnalysisResult)
async def get_analysis_result(analysis_id: str):
    """
    Retrieve analysis results by ID
    """
    result = await analyzer.get_analysis_result(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@router.post("/predict", response_model=dict)
async def predict_performance(
    file: UploadFile = File(...),
    target_platform: str = "instagram"
):
    """
    Predict design performance for specific platform
    """
    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image = await process_uploaded_image(file)
        prediction = await analyzer.predict_performance(image, target_platform)
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance prediction failed: {str(e)}")


@router.post("/variants", response_model=List[dict])
async def generate_variants(
    file: UploadFile = File(...),
    variant_count: int = 3
):
    """
    Generate A/B test variants of the design
    """
    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail="Invalid image file")

    if variant_count > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 variants allowed")

    try:
        image = await process_uploaded_image(file)
        variants = await analyzer.generate_variants(image, variant_count)
        return variants

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Variant generation failed: {str(e)}")