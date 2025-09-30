"""
Production Performance Predictor
Uses trained ML models for real-time design performance prediction
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging
from PIL import Image

from app.services.image_processor import ImageProcessor
from app.services.typography_analyzer import TypographyAnalyzer
from app.models.schemas import ColorPalette, TypographyAnalysis, LayoutAnalysis

logger = logging.getLogger(__name__)


class ProductionPerformancePredictor:
    """
    Production-ready performance predictor using trained models
    """

    def __init__(self):
        self.models_dir = Path("models/trained")
        self.image_processor = ImageProcessor()
        self.typography_analyzer = TypographyAnalyzer()

        # Loaded models cache
        self.models = {}
        self.scalers = {}
        self.feature_names = {}

        # Load all available models
        self._load_models()

    def _load_models(self):
        """
        Load all trained models from disk
        """
        if not self.models_dir.exists():
            logger.warning("No trained models directory found. Using fallback models.")
            return

        # Find all model files
        model_files = list(self.models_dir.glob("*_performance_model.joblib"))

        for model_file in model_files:
            platform = model_file.stem.replace("_performance_model", "")

            try:
                # Load model
                self.models[platform] = joblib.load(model_file)

                # Load scaler
                scaler_file = self.models_dir / f"{platform}_scaler.joblib"
                if scaler_file.exists():
                    self.scalers[platform] = joblib.load(scaler_file)

                # Load feature names
                features_file = self.models_dir / f"{platform}_features.json"
                if features_file.exists():
                    with open(features_file, "r") as f:
                        self.feature_names[platform] = json.load(f)

                logger.info(f"Loaded model for platform: {platform}")

            except Exception as e:
                logger.error(f"Failed to load model for {platform}: {e}")

        if not self.models:
            logger.warning("No models loaded. Using synthetic prediction.")

    async def predict_comprehensive(
        self, image: Image.Image, target_platform: str,
        color_analysis: ColorPalette, typography_analysis: TypographyAnalysis,
        layout_analysis: LayoutAnalysis
    ) -> Dict[str, Any]:
        """
        Comprehensive performance prediction using trained models
        """
        # Extract features
        features = self._extract_production_features(
            image, color_analysis, typography_analysis, layout_analysis, target_platform
        )

        # Get predictions for all platforms
        platform_scores = {}
        confidence_scores = {}

        for platform in ["general", "instagram", "linkedin", "facebook", "tiktok"]:
            if platform in self.models:
                score, confidence = await self._predict_for_platform(features, platform)
            else:
                # Fallback to general model or synthetic
                score, confidence = await self._predict_fallback(features, platform)

            platform_scores[platform] = score
            confidence_scores[platform] = confidence

        # Main platform prediction
        main_platform = target_platform if target_platform in platform_scores else "general"
        main_score = platform_scores.get(main_platform, 75.0)
        main_confidence = confidence_scores.get(main_platform, 70.0)

        # Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(features, main_score)

        return {
            "engagement_score": main_score,
            "platform_scores": platform_scores,
            "improvement_potential": improvement_potential,
            "confidence": main_confidence,
            "feature_insights": self._get_feature_insights(features, main_platform),
            "prediction_metadata": {
                "model_used": main_platform,
                "features_count": len(features),
                "prediction_timestamp": pd.Timestamp.now().isoformat()
            }
        }

    def _extract_production_features(
        self, image: Image.Image, color_analysis: ColorPalette,
        typography_analysis: TypographyAnalysis, layout_analysis: LayoutAnalysis,
        platform: str
    ) -> Dict[str, float]:
        """
        Extract features for production prediction (optimized version)
        """
        features = {}

        # 1. Image properties
        width, height = image.size
        features["image_width"] = width
        features["image_height"] = height
        features["aspect_ratio"] = width / height
        features["image_area"] = width * height

        # 2. Color features
        features["num_dominant_colors"] = len(color_analysis.dominant_colors)
        features["color_harmony_score"] = color_analysis.harmony_score
        features["accessibility_score"] = color_analysis.accessibility_score

        # Color scheme encoding
        color_scheme_map = {"complementary": 1, "triadic": 2, "analogous": 3, "monochromatic": 4, "custom": 5}
        features["color_scheme_encoded"] = color_scheme_map.get(color_analysis.color_scheme, 5)

        # 3. Typography features
        features["font_pairing_score"] = typography_analysis.font_pairing_score
        features["readability_score"] = typography_analysis.readability_score
        features["text_hierarchy_score"] = typography_analysis.text_hierarchy_score
        features["num_fonts_detected"] = len(typography_analysis.fonts_detected)

        # 4. Layout features
        features["composition_score"] = layout_analysis.composition_score
        features["balance_score"] = layout_analysis.balance_score
        features["grid_alignment"] = layout_analysis.grid_alignment
        features["white_space_usage"] = layout_analysis.white_space_usage
        features["num_focal_points"] = len(layout_analysis.focal_points)

        # 5. Platform encoding
        features["platform_instagram"] = 1 if platform == "instagram" else 0
        features["platform_linkedin"] = 1 if platform == "linkedin" else 0
        features["platform_facebook"] = 1 if platform == "facebook" else 0
        features["platform_tiktok"] = 1 if platform == "tiktok" else 0

        # 6. Derived features
        features["overall_design_quality"] = (
            features["color_harmony_score"] * 0.3 +
            features["readability_score"] * 0.3 +
            features["composition_score"] * 0.4
        )

        features["visual_complexity"] = (
            features["num_dominant_colors"] / 8 * 0.3 +
            features["num_fonts_detected"] / 3 * 0.3 +
            features["num_focal_points"] / 5 * 0.4
        ) * 100

        features["accessibility_friendly"] = 1 if features["accessibility_score"] > 80 else 0
        features["is_minimal_design"] = 1 if features["visual_complexity"] < 30 else 0

        return features

    async def _predict_for_platform(self, features: Dict[str, float], platform: str) -> tuple[float, float]:
        """
        Make prediction using trained model for specific platform
        """
        try:
            model = self.models[platform]
            scaler = self.scalers.get(platform)
            feature_names = self.feature_names.get(platform, list(features.keys()))

            # Prepare feature vector
            feature_vector = []
            for feature_name in feature_names:
                feature_vector.append(features.get(feature_name, 0.0))

            feature_array = np.array(feature_vector).reshape(1, -1)

            # Scale features if scaler available
            if scaler:
                feature_array = scaler.transform(feature_array)

            # Make prediction
            prediction = model.predict(feature_array)[0]

            # Calculate confidence based on model uncertainty
            confidence = self._calculate_model_confidence(model, feature_array, platform)

            # Ensure prediction is in valid range
            prediction = max(0, min(100, prediction))

            return float(prediction), float(confidence)

        except Exception as e:
            logger.error(f"Prediction error for {platform}: {e}")
            return self._predict_fallback(features, platform)

    async def _predict_fallback(self, features: Dict[str, float], platform: str) -> tuple[float, float]:
        """
        Fallback prediction when trained model is not available
        """
        # Rule-based prediction based on features
        base_score = features.get("overall_design_quality", 75.0)

        # Platform-specific adjustments
        platform_adjustments = {
            "instagram": {
                "visual_weight": 0.4,
                "color_weight": 0.3,
                "typography_weight": 0.3
            },
            "linkedin": {
                "visual_weight": 0.2,
                "color_weight": 0.2,
                "typography_weight": 0.6
            },
            "tiktok": {
                "visual_weight": 0.5,
                "color_weight": 0.3,
                "typography_weight": 0.2
            },
            "facebook": {
                "visual_weight": 0.35,
                "color_weight": 0.35,
                "typography_weight": 0.3
            }
        }

        weights = platform_adjustments.get(platform, {"visual_weight": 0.33, "color_weight": 0.33, "typography_weight": 0.34})

        # Calculate weighted score
        visual_score = (features.get("composition_score", 75) + features.get("balance_score", 75)) / 2
        color_score = features.get("color_harmony_score", 75)
        typography_score = (features.get("readability_score", 75) + features.get("font_pairing_score", 75)) / 2

        adjusted_score = (
            visual_score * weights["visual_weight"] +
            color_score * weights["color_weight"] +
            typography_score * weights["typography_weight"]
        )

        # Add some variance based on platform preferences
        if platform == "instagram" and features.get("visual_complexity", 50) > 70:
            adjusted_score += 5  # Instagram likes visually rich content

        if platform == "linkedin" and features.get("accessibility_score", 75) > 85:
            adjusted_score += 8  # LinkedIn values accessibility

        confidence = 65.0  # Lower confidence for rule-based predictions

        return max(0, min(100, adjusted_score)), confidence

    def _calculate_model_confidence(self, model, feature_array: np.ndarray, platform: str) -> float:
        """
        Calculate prediction confidence based on model uncertainty
        """
        try:
            # For ensemble models, use prediction variance
            if hasattr(model, "predict_proba"):
                # For classification models (if any)
                proba = model.predict_proba(feature_array)[0]
                confidence = max(proba) * 100
            elif hasattr(model, "estimators_"):
                # For ensemble regression models
                predictions = [estimator.predict(feature_array)[0] for estimator in model.estimators_[:10]]
                std_dev = np.std(predictions)
                confidence = max(0, min(100, 100 - std_dev * 2))
            else:
                # Default confidence for other models
                confidence = 80.0

            return confidence

        except Exception as e:
            logger.warning(f"Could not calculate confidence for {platform}: {e}")
            return 75.0  # Default confidence

    def _calculate_improvement_potential(self, features: Dict[str, float], current_score: float) -> float:
        """
        Calculate improvement potential based on weak areas
        """
        weak_areas = []
        thresholds = {
            "color_harmony_score": 75,
            "readability_score": 75,
            "composition_score": 75,
            "accessibility_score": 80
        }

        for feature, threshold in thresholds.items():
            if features.get(feature, 100) < threshold:
                weak_areas.append(feature)

        if not weak_areas:
            return min(20, 100 - current_score)  # Always some room for improvement

        # Calculate potential based on number and severity of issues
        improvement = len(weak_areas) * 15

        # Additional potential for specific issues
        if features.get("accessibility_score", 100) < 60:
            improvement += 20  # Accessibility is critical

        if features.get("visual_complexity", 50) > 80:
            improvement += 15  # Simplification potential

        return min(80, improvement)

    def _get_feature_insights(self, features: Dict[str, float], platform: str) -> Dict[str, Any]:
        """
        Provide insights about key features affecting performance
        """
        insights = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        # Identify strengths
        if features.get("color_harmony_score", 0) > 85:
            insights["strengths"].append("Excellent color harmony")

        if features.get("readability_score", 0) > 85:
            insights["strengths"].append("High text readability")

        if features.get("composition_score", 0) > 85:
            insights["strengths"].append("Strong composition")

        # Identify weaknesses
        if features.get("accessibility_score", 100) < 70:
            insights["weaknesses"].append("Poor accessibility compliance")
            insights["recommendations"].append("Increase color contrast for better accessibility")

        if features.get("visual_complexity", 50) > 75:
            insights["weaknesses"].append("High visual complexity")
            insights["recommendations"].append("Simplify design by reducing elements")

        if features.get("font_pairing_score", 100) < 60:
            insights["weaknesses"].append("Poor font pairing")
            insights["recommendations"].append("Use fewer, better-paired fonts")

        # Platform-specific insights
        if platform == "instagram":
            if features.get("aspect_ratio", 1) < 0.8:
                insights["recommendations"].append("Consider square or vertical format for Instagram")

        elif platform == "linkedin":
            if features.get("readability_score", 100) < 80:
                insights["recommendations"].append("Improve text readability for professional audience")

        return insights

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        """
        info = {
            "models_loaded": list(self.models.keys()),
            "models_directory": str(self.models_dir),
            "feature_counts": {platform: len(features) for platform, features in self.feature_names.items()},
            "fallback_available": True
        }

        return info

    async def retrain_model(self, platform: str, new_data: pd.DataFrame):
        """
        Retrain model with new data (for continuous learning)
        """
        try:
            from app.ml.model_trainer import PerformanceModelTrainer

            trainer = PerformanceModelTrainer()

            # Prepare training data
            X, y, feature_cols = trainer.prepare_training_data(new_data)

            # Retrain model
            results = trainer.train_models(X, y, platform=platform)

            # Reload the updated model
            self._load_models()

            logger.info(f"Model retrained for platform: {platform}")
            return True

        except Exception as e:
            logger.error(f"Retraining failed for {platform}: {e}")
            return False


# Global instance for production use
production_predictor = ProductionPerformancePredictor()