import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from PIL import Image
import logging

from app.models.schemas import (
    AnalysisRequest, DesignAnalysisResult, ColorPalette, TypographyAnalysis,
    LayoutAnalysis, VisualHierarchy, PerformancePrediction, DesignRecommendation,
    AnalysisStatus
)
from app.services.image_processor import ImageProcessor
from app.services.typography_analyzer import TypographyAnalyzer
from app.services.performance_predictor import PerformancePredictor
from app.services.recommendation_engine import RecommendationEngine
from app.core.ai_models import ai_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class DesignAnalyzer:
    """
    Main design analysis orchestrator that coordinates all analysis components
    """

    def __init__(self):
        self.image_processor = ImageProcessor()
        self.typography_analyzer = TypographyAnalyzer()
        self.performance_predictor = PerformancePredictor()
        self.recommendation_engine = RecommendationEngine()

        # In-memory storage for analysis results (in production, use database)
        self.analysis_cache: Dict[str, DesignAnalysisResult] = {}

    async def analyze_design(self, analysis_id: str, image: Image.Image, request: AnalysisRequest) -> DesignAnalysisResult:
        """
        Perform comprehensive design analysis
        """
        start_time = time.time()

        try:
            # Initialize result
            result = DesignAnalysisResult(
                analysis_id=analysis_id,
                status=AnalysisStatus.PROCESSING,
                overall_score=0.0
            )

            # Store initial result
            self.analysis_cache[analysis_id] = result

            # Step 1: Preprocess image
            processed_data = self.image_processor.preprocess_image(image)

            # Step 2: Parallel analysis of different components
            color_task = self._analyze_color_harmony(image)
            typography_task = self._analyze_typography(image)
            layout_task = self._analyze_layout_composition(image)
            hierarchy_task = self._analyze_visual_hierarchy(image)

            # Run VLM analysis
            vlm_task = self._get_vlm_analysis(image, request.prompt)

            # Wait for all analyses to complete
            color_analysis, typography_analysis, layout_analysis, hierarchy_analysis, vlm_analysis = await asyncio.gather(
                color_task, typography_task, layout_task, hierarchy_task, vlm_task
            )

            # Step 3: Performance prediction
            performance_prediction = await self._predict_performance(
                image, request.target_platform, color_analysis, typography_analysis, layout_analysis
            )

            # Step 4: Generate recommendations
            recommendations = await self._generate_recommendations(
                vlm_analysis, color_analysis, typography_analysis, layout_analysis, hierarchy_analysis
            )

            # Step 5: Calculate overall score
            overall_score = self._calculate_overall_score(
                color_analysis, typography_analysis, layout_analysis, hierarchy_analysis, performance_prediction
            )

            # Update result
            processing_time = time.time() - start_time
            result = DesignAnalysisResult(
                analysis_id=analysis_id,
                status=AnalysisStatus.COMPLETED,
                overall_score=overall_score,
                color_analysis=color_analysis,
                typography_analysis=typography_analysis,
                layout_analysis=layout_analysis,
                visual_hierarchy=hierarchy_analysis,
                performance_prediction=performance_prediction,
                recommendations=recommendations,
                processing_time=processing_time
            )

            # Cache result
            self.analysis_cache[analysis_id] = result

            logger.info(f"Analysis {analysis_id} completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {str(e)}")
            error_result = DesignAnalysisResult(
                analysis_id=analysis_id,
                status=AnalysisStatus.FAILED,
                overall_score=0.0,
                processing_time=time.time() - start_time
            )
            self.analysis_cache[analysis_id] = error_result
            raise

    async def get_analysis_result(self, analysis_id: str) -> Optional[DesignAnalysisResult]:
        """
        Retrieve analysis result by ID
        """
        return self.analysis_cache.get(analysis_id)

    async def predict_performance(self, image: Image.Image, target_platform: str) -> Dict[str, Any]:
        """
        Quick performance prediction for specific platform
        """
        return await self.performance_predictor.predict_engagement(image, target_platform)

    async def generate_variants(self, image: Image.Image, variant_count: int) -> List[Dict[str, Any]]:
        """
        Generate A/B test variants
        """
        return await self.recommendation_engine.generate_variants(image, variant_count)

    # Private analysis methods

    async def _analyze_color_harmony(self, image: Image.Image) -> ColorPalette:
        """
        Analyze color harmony and palette
        """
        # Extract color palette
        colors = self.image_processor.extract_color_palette(image)

        # Analyze harmony
        harmony_data = self.image_processor.analyze_color_harmony(colors)

        # Calculate accessibility score (WCAG contrast ratios)
        accessibility_score = self._calculate_accessibility_score(colors)

        return ColorPalette(
            dominant_colors=colors,
            color_scheme=harmony_data["scheme_type"],
            harmony_score=harmony_data["harmony_score"],
            accessibility_score=accessibility_score
        )

    async def _analyze_typography(self, image: Image.Image) -> TypographyAnalysis:
        """
        Analyze typography and text elements
        """
        typography_data = self.typography_analyzer.analyze_typography(image)

        return TypographyAnalysis(
            fonts_detected=typography_data["fonts_detected"],
            font_pairing_score=typography_data["font_pairing_score"],
            readability_score=typography_data["readability_score"],
            text_hierarchy_score=typography_data["text_hierarchy_score"]
        )

    async def _analyze_layout_composition(self, image: Image.Image) -> LayoutAnalysis:
        """
        Analyze layout and composition
        """
        composition_data = self.image_processor.analyze_composition(image)

        return LayoutAnalysis(
            composition_score=composition_data["rule_of_thirds_score"],
            balance_score=composition_data["balance_score"],
            grid_alignment=self._calculate_grid_alignment(image),
            white_space_usage=self._calculate_white_space_usage(image),
            focal_points=composition_data["focal_points"]
        )

    async def _analyze_visual_hierarchy(self, image: Image.Image) -> VisualHierarchy:
        """
        Analyze visual hierarchy and eye flow
        """
        # Generate saliency map
        saliency_map = self.image_processor.generate_saliency_map(image)

        # Detect eye flow pattern
        eye_flow_pattern = self._detect_eye_flow_pattern(image)

        # Calculate hierarchy effectiveness
        hierarchy_score = self._calculate_hierarchy_effectiveness(image, saliency_map)

        # Get attention areas from composition analysis
        composition_data = self.image_processor.analyze_composition(image)
        attention_areas = self._convert_focal_points_to_attention_areas(composition_data["focal_points"])

        return VisualHierarchy(
            hierarchy_score=hierarchy_score,
            eye_flow_pattern=eye_flow_pattern,
            saliency_map=saliency_map,
            attention_areas=attention_areas
        )

    async def _get_vlm_analysis(self, image: Image.Image, prompt: str) -> str:
        """
        Get Vision Language Model analysis
        """
        try:
            analysis = await ai_manager.analyze_design(image, prompt)
            return analysis
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return "VLM analysis unavailable"

    async def _predict_performance(
        self, image: Image.Image, target_platform: str,
        color_analysis: ColorPalette, typography_analysis: TypographyAnalysis,
        layout_analysis: LayoutAnalysis
    ) -> PerformancePrediction:
        """
        Predict design performance
        """
        prediction_data = await self.performance_predictor.predict_comprehensive(
            image, target_platform, color_analysis, typography_analysis, layout_analysis
        )

        return PerformancePrediction(
            engagement_score=prediction_data["engagement_score"],
            platform_optimization=prediction_data["platform_scores"],
            improvement_potential=prediction_data["improvement_potential"],
            confidence_interval=prediction_data["confidence"]
        )

    async def _generate_recommendations(
        self, vlm_analysis: str, color_analysis: ColorPalette,
        typography_analysis: TypographyAnalysis, layout_analysis: LayoutAnalysis,
        hierarchy_analysis: VisualHierarchy
    ) -> List[DesignRecommendation]:
        """
        Generate actionable design recommendations
        """
        return await self.recommendation_engine.generate_recommendations(
            vlm_analysis, color_analysis, typography_analysis, layout_analysis, hierarchy_analysis
        )

    # Helper methods

    def _calculate_overall_score(
        self, color_analysis: ColorPalette, typography_analysis: TypographyAnalysis,
        layout_analysis: LayoutAnalysis, hierarchy_analysis: VisualHierarchy,
        performance_prediction: PerformancePrediction
    ) -> float:
        """
        Calculate weighted overall design score
        """
        # Component weights based on design importance
        weights = {
            "color": 0.20,
            "typography": 0.25,
            "layout": 0.25,
            "hierarchy": 0.20,
            "performance": 0.10
        }

        scores = {
            "color": color_analysis.harmony_score,
            "typography": (typography_analysis.font_pairing_score + typography_analysis.readability_score) / 2,
            "layout": (layout_analysis.composition_score + layout_analysis.balance_score) / 2,
            "hierarchy": hierarchy_analysis.hierarchy_score,
            "performance": performance_prediction.engagement_score
        }

        overall_score = sum(scores[component] * weights[component] for component in weights)
        return round(overall_score, 1)

    def _calculate_accessibility_score(self, colors: List[str]) -> float:
        """
        Calculate WCAG accessibility score for color palette
        """
        if len(colors) < 2:
            return 80.0  # Single color is neutral

        # Convert colors to luminance and calculate contrast ratios
        contrast_scores = []

        for i, color1 in enumerate(colors[:3]):  # Check top 3 colors
            for color2 in colors[i+1:4]:  # Against next colors
                luminance1 = self._get_relative_luminance(color1)
                luminance2 = self._get_relative_luminance(color2)

                lighter = max(luminance1, luminance2)
                darker = min(luminance1, luminance2)

                contrast_ratio = (lighter + 0.05) / (darker + 0.05)

                # WCAG scoring
                if contrast_ratio >= 7.0:  # AAA
                    contrast_scores.append(100)
                elif contrast_ratio >= 4.5:  # AA
                    contrast_scores.append(80)
                elif contrast_ratio >= 3.0:  # AA Large
                    contrast_scores.append(60)
                else:
                    contrast_scores.append(30)

        return sum(contrast_scores) / len(contrast_scores) if contrast_scores else 80.0

    def _get_relative_luminance(self, hex_color: str) -> float:
        """
        Calculate relative luminance for WCAG contrast calculation
        """
        # Remove # if present
        hex_color = hex_color.lstrip('#')

        # Convert to RGB
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        # Apply gamma correction
        def gamma_correct(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r_linear = gamma_correct(r)
        g_linear = gamma_correct(g)
        b_linear = gamma_correct(b)

        # Calculate luminance
        return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

    def _calculate_grid_alignment(self, image: Image.Image) -> float:
        """
        Calculate grid alignment score
        """
        # Simplified grid alignment calculation
        # In production, use more sophisticated grid detection
        width, height = image.size

        # Check if elements align to common grid systems (12-column, 16-column)
        grid_score = 75.0  # Default neutral score

        # Analyze edge distributions for alignment
        import cv2
        import numpy as np

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check vertical alignment
        vertical_projection = np.sum(edges, axis=0)
        vertical_peaks = self._find_peaks(vertical_projection)

        # Check horizontal alignment
        horizontal_projection = np.sum(edges, axis=1)
        horizontal_peaks = self._find_peaks(horizontal_projection)

        # Score based on peak regularity
        if len(vertical_peaks) >= 3:
            vertical_spacing = np.diff(vertical_peaks)
            vertical_consistency = 1 - (np.std(vertical_spacing) / np.mean(vertical_spacing)) if np.mean(vertical_spacing) > 0 else 0
            grid_score += vertical_consistency * 15

        if len(horizontal_peaks) >= 3:
            horizontal_spacing = np.diff(horizontal_peaks)
            horizontal_consistency = 1 - (np.std(horizontal_spacing) / np.mean(horizontal_spacing)) if np.mean(horizontal_spacing) > 0 else 0
            grid_score += horizontal_consistency * 10

        return min(100, max(0, grid_score))

    def _calculate_white_space_usage(self, image: Image.Image) -> float:
        """
        Calculate white space usage effectiveness
        """
        import cv2
        import numpy as np

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Identify white/empty space (bright areas with low variation)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_space_ratio = np.sum(binary == 255) / (binary.shape[0] * binary.shape[1])

        # Optimal white space is typically 20-40%
        if 0.20 <= white_space_ratio <= 0.40:
            return 100.0
        elif 0.15 <= white_space_ratio <= 0.50:
            return 80.0
        elif 0.10 <= white_space_ratio <= 0.60:
            return 60.0
        else:
            return 40.0

    def _detect_eye_flow_pattern(self, image: Image.Image) -> str:
        """
        Detect eye flow pattern (Z, F, or circular)
        """
        # Simplified pattern detection based on saliency distribution
        saliency_map = self.image_processor._calculate_saliency(
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        )

        height, width = saliency_map.shape

        # Divide into regions for pattern analysis
        top_left = np.mean(saliency_map[:height//2, :width//2])
        top_right = np.mean(saliency_map[:height//2, width//2:])
        bottom_left = np.mean(saliency_map[height//2:, :width//2])
        bottom_right = np.mean(saliency_map[height//2:, width//2:])

        # Z-pattern: high top-left, top-right, bottom-left
        z_score = top_left + top_right + bottom_left - bottom_right

        # F-pattern: high left side
        f_score = top_left + bottom_left - top_right - bottom_right

        # Determine pattern
        if z_score > f_score and z_score > 0:
            return "Z-pattern"
        elif f_score > 0:
            return "F-pattern"
        else:
            return "circular"

    def _calculate_hierarchy_effectiveness(self, image: Image.Image, saliency_map: str) -> float:
        """
        Calculate visual hierarchy effectiveness
        """
        # Extract base64 saliency data and analyze distribution
        import base64
        import cv2
        import numpy as np
        from io import BytesIO

        try:
            # Decode base64 saliency map
            saliency_data = base64.b64decode(saliency_map.split(',')[1])
            saliency_img = Image.open(BytesIO(saliency_data))
            saliency_array = np.array(saliency_img.convert('L'))

            # Calculate hierarchy based on saliency distribution
            # Good hierarchy has clear peaks and valleys
            saliency_variance = np.var(saliency_array)
            max_variance = 255**2 / 4  # Theoretical maximum

            hierarchy_score = min(100, (saliency_variance / max_variance) * 100)
            return hierarchy_score

        except Exception:
            # Fallback score
            return 75.0

    def _convert_focal_points_to_attention_areas(self, focal_points: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Convert focal points to attention areas format
        """
        attention_areas = []
        for i, point in enumerate(focal_points):
            attention_areas.append({
                "id": i,
                "x": point["x"],
                "y": point["y"],
                "strength": point["strength"],
                "type": "focal_point",
                "description": f"High attention area {i+1}"
            })
        return attention_areas

    def _find_peaks(self, data: np.ndarray, min_distance: int = 20) -> List[int]:
        """
        Find peaks in 1D data for grid analysis
        """
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > np.mean(data):
                # Check minimum distance from existing peaks
                if not peaks or min(abs(i - peak) for peak in peaks) >= min_distance:
                    peaks.append(i)
        return peaks