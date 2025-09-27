from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ColorPalette(BaseModel):
    dominant_colors: List[str] = Field(description="Hex codes of dominant colors")
    color_scheme: str = Field(description="Type of color scheme (complementary, triadic, etc.)")
    harmony_score: float = Field(ge=0, le=100, description="Color harmony score 0-100")
    accessibility_score: float = Field(ge=0, le=100, description="WCAG contrast score")


class TypographyAnalysis(BaseModel):
    fonts_detected: List[str] = Field(description="Detected font families")
    font_pairing_score: float = Field(ge=0, le=100, description="Font pairing quality score")
    readability_score: float = Field(ge=0, le=100, description="Text readability score")
    text_hierarchy_score: float = Field(ge=0, le=100, description="Typography hierarchy score")


class LayoutAnalysis(BaseModel):
    composition_score: float = Field(ge=0, le=100, description="Overall composition score")
    balance_score: float = Field(ge=0, le=100, description="Visual balance score")
    grid_alignment: float = Field(ge=0, le=100, description="Grid alignment score")
    white_space_usage: float = Field(ge=0, le=100, description="White space optimization")
    focal_points: List[Dict[str, float]] = Field(description="Detected focal points with coordinates")


class VisualHierarchy(BaseModel):
    hierarchy_score: float = Field(ge=0, le=100, description="Visual hierarchy effectiveness")
    eye_flow_pattern: str = Field(description="Detected eye flow pattern (Z, F, etc.)")
    saliency_map: Optional[str] = Field(description="Base64 encoded saliency map")
    attention_areas: List[Dict[str, Any]] = Field(description="Key attention-grabbing areas")


class PerformancePrediction(BaseModel):
    engagement_score: float = Field(ge=0, le=100, description="Predicted engagement potential")
    platform_optimization: Dict[str, float] = Field(description="Platform-specific scores")
    improvement_potential: float = Field(ge=0, le=100, description="Potential for improvement")
    confidence_interval: float = Field(ge=0, le=100, description="Prediction confidence")


class DesignRecommendation(BaseModel):
    category: str = Field(description="Recommendation category")
    priority: str = Field(description="Priority level (high, medium, low)")
    description: str = Field(description="Human-readable recommendation")
    technical_details: str = Field(description="Technical implementation details")
    impact_score: float = Field(ge=0, le=100, description="Expected impact of implementing")


class DesignAnalysisResult(BaseModel):
    analysis_id: str = Field(description="Unique analysis identifier")
    status: AnalysisStatus
    overall_score: float = Field(ge=0, le=100, description="Overall design quality score")

    # Core Analysis Components
    color_analysis: Optional[ColorPalette] = None
    typography_analysis: Optional[TypographyAnalysis] = None
    layout_analysis: Optional[LayoutAnalysis] = None
    visual_hierarchy: Optional[VisualHierarchy] = None

    # Intelligence Features
    performance_prediction: Optional[PerformancePrediction] = None
    recommendations: List[DesignRecommendation] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = Field(description="Analysis time in seconds")
    model_version: str = Field(default="1.0.0")


class AnalysisRequest(BaseModel):
    prompt: Optional[str] = Field(
        default="Analyze this design and provide detailed feedback on color, typography, layout, and visual hierarchy. Include specific recommendations for improvement.",
        description="Custom analysis prompt"
    )
    analysis_type: str = Field(
        default="comprehensive",
        description="Type of analysis (comprehensive, quick, color-only, etc.)"
    )
    target_platform: Optional[str] = Field(
        default="general",
        description="Target platform (instagram, linkedin, tiktok, web, etc.)"
    )
    brand_context: Optional[str] = Field(
        description="Brand guidelines or context for consistency analysis"
    )


class BatchAnalysisRequest(BaseModel):
    requests: List[AnalysisRequest] = Field(description="List of analysis requests")
    comparison_analysis: bool = Field(
        default=False,
        description="Whether to perform comparative analysis across images"
    )


class ChatRequest(BaseModel):
    message: str = Field(description="User's design question or feedback request")
    analysis_id: Optional[str] = Field(description="Related analysis ID for context")
    image_context: Optional[str] = Field(description="Base64 encoded image for reference")


class ChatResponse(BaseModel):
    response: str = Field(description="AI response to the design question")
    suggestions: List[str] = Field(default_factory=list, description="Additional suggestions")
    related_analyses: List[str] = Field(default_factory=list, description="Related analysis IDs")


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_status: str = Field(description="AI model availability status")
    version: str = Field(default="1.0.0")