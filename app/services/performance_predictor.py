import numpy as np
from PIL import Image
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import cv2

from app.models.schemas import ColorPalette, TypographyAnalysis, LayoutAnalysis
from app.ml.production_predictor import production_predictor


class PerformancePredictor:
    """
    ML-based performance prediction for design engagement
    Now uses production-trained models when available, falls back to synthetic models
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.platform_weights = {
            "instagram": {"visual": 0.4, "color": 0.3, "layout": 0.2, "typography": 0.1},
            "linkedin": {"typography": 0.4, "layout": 0.3, "visual": 0.2, "color": 0.1},
            "tiktok": {"visual": 0.5, "color": 0.25, "layout": 0.15, "typography": 0.1},
            "facebook": {"visual": 0.35, "color": 0.25, "layout": 0.25, "typography": 0.15},
            "twitter": {"typography": 0.35, "visual": 0.3, "layout": 0.2, "color": 0.15},
            "general": {"visual": 0.3, "color": 0.25, "layout": 0.25, "typography": 0.2}
        }

        # Initialize pre-trained models (in production, load from files)
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize ML models for performance prediction
        """
        # Create dummy models for demo (in production, load trained models)
        for platform in self.platform_weights.keys():
            # Random Forest for engagement prediction
            self.models[platform] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scalers[platform] = StandardScaler()

            # Train on synthetic data for demo
            self._train_synthetic_model(platform)

    def _train_synthetic_model(self, platform: str):
        """
        Train model on synthetic data for demonstration
        """
        # Generate synthetic training data
        n_samples = 1000
        features = np.random.rand(n_samples, 20)  # 20 features

        # Simulate platform-specific performance relationships
        weights = self.platform_weights[platform]

        # Create synthetic engagement scores based on weighted features
        engagement = (
            features[:, 0:5].mean(axis=1) * weights["visual"] +
            features[:, 5:10].mean(axis=1) * weights["color"] +
            features[:, 10:15].mean(axis=1) * weights["layout"] +
            features[:, 15:20].mean(axis=1) * weights["typography"]
        ) * 100

        # Add some noise
        engagement += np.random.normal(0, 10, n_samples)
        engagement = np.clip(engagement, 0, 100)

        # Fit scaler and model
        features_scaled = self.scalers[platform].fit_transform(features)
        self.models[platform].fit(features_scaled, engagement)

    async def predict_engagement(self, image: Image.Image, target_platform: str = "general") -> Dict[str, Any]:
        """
        Predict engagement score for specific platform
        """
        # Extract features from image
        features = self._extract_features(image)

        # Get platform-specific prediction
        platform = target_platform if target_platform in self.models else "general"

        scaled_features = self.scalers[platform].transform([features])
        engagement_score = self.models[platform].predict(scaled_features)[0]

        # Calculate confidence based on model uncertainty
        confidence = self._calculate_prediction_confidence(scaled_features, platform)

        return {
            "engagement_score": max(0, min(100, float(engagement_score))),
            "confidence": confidence,
            "platform": platform,
            "feature_importance": self._get_feature_importance(platform)
        }

    async def predict_comprehensive(
        self, image: Image.Image, target_platform: str,
        color_analysis: ColorPalette, typography_analysis: TypographyAnalysis,
        layout_analysis: LayoutAnalysis
    ) -> Dict[str, Any]:
        """
        Comprehensive performance prediction using all analysis components
        Uses production-trained models when available
        """
        try:
            # Try to use production predictor with trained models
            if hasattr(production_predictor, 'models') and production_predictor.models:
                return await production_predictor.predict_comprehensive(
                    image, target_platform, color_analysis, typography_analysis, layout_analysis
                )
        except Exception as e:
            # Fall back to synthetic prediction
            pass

        # Fallback to original synthetic prediction
        # Extract comprehensive features
        features = self._extract_comprehensive_features(
            image, color_analysis, typography_analysis, layout_analysis
        )

        # Predict for multiple platforms
        platform_scores = {}
        main_platform = target_platform if target_platform in self.models else "general"

        # Predict for main platform
        main_prediction = await self.predict_engagement(image, main_platform)
        platform_scores[main_platform] = main_prediction["engagement_score"]

        # Predict for other major platforms
        for platform in ["instagram", "linkedin", "facebook"]:
            if platform != main_platform:
                pred = await self.predict_engagement(image, platform)
                platform_scores[platform] = pred["engagement_score"]

        # Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(features, main_platform)

        # Calculate overall confidence
        confidence = main_prediction["confidence"]

        return {
            "engagement_score": main_prediction["engagement_score"],
            "platform_scores": platform_scores,
            "improvement_potential": improvement_potential,
            "confidence": confidence,
            "detailed_features": features
        }

    def _extract_features(self, image: Image.Image) -> List[float]:
        """
        Extract basic features from image for ML prediction
        """
        features = []

        # Convert to numpy array
        img_array = np.array(image)
        cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 1. Color features (5 features)
        # Average RGB values
        features.extend([
            np.mean(img_array[:, :, 0]) / 255.0,  # Red
            np.mean(img_array[:, :, 1]) / 255.0,  # Green
            np.mean(img_array[:, :, 2]) / 255.0,  # Blue
        ])

        # Color variance
        features.append(np.var(img_array) / (255**2))

        # Saturation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        features.append(np.mean(hsv[:, :, 1]) / 255.0)

        # 2. Visual complexity features (5 features)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)

        # Texture features
        features.extend([
            np.std(gray) / 255.0,  # Contrast
            np.mean(gray) / 255.0,  # Brightness
            len(np.unique(gray)) / 256.0,  # Color diversity
        ])

        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(gradient_magnitude) / 255.0)

        # 3. Layout features (5 features)
        height, width = gray.shape

        # Aspect ratio
        features.append(width / height)

        # Symmetry
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry = np.corrcoef(
            left_half[:, :min_width].flatten(),
            right_half[:, :min_width].flatten()
        )[0, 1]
        features.append(max(0, symmetry) if not np.isnan(symmetry) else 0)

        # Rule of thirds alignment
        third_y = height // 3
        third_x = width // 3
        roi_strength = np.mean([
            np.mean(gray[third_y:2*third_y, third_x:2*third_x]),
            np.mean(gray[:third_y, :third_x]),
            np.mean(gray[:third_y, 2*third_x:]),
            np.mean(gray[2*third_y:, :third_x]),
            np.mean(gray[2*third_y:, 2*third_x:])
        ])
        features.append(roi_strength / 255.0)

        # White space ratio
        white_space = np.sum(gray > 240) / gray.size
        features.append(white_space)

        # Center weight
        center_region = gray[height//4:3*height//4, width//4:3*width//4]
        center_weight = np.mean(center_region) / 255.0
        features.append(center_weight)

        # 4. Typography-related features (5 features)
        # These are simplified - in production, use actual text detection

        # High contrast regions (potential text)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        text_ratio = np.sum(binary == 0) / binary.size  # Dark regions
        features.append(text_ratio)

        # Horizontal line density (text indicator)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        features.append(np.sum(horizontal_lines > 0) / horizontal_lines.size)

        # Vertical line density
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        features.append(np.sum(vertical_lines > 0) / vertical_lines.size)

        # Text region estimate
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 0.1 < w/h < 20 and w > 10 and h > 5:  # Text-like aspect ratio
                text_regions += 1
        features.append(min(1.0, text_regions / 10))  # Normalize

        # Line uniformity
        if len(contours) > 0:
            heights = [cv2.boundingRect(c)[3] for c in contours]
            height_std = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0
            features.append(max(0, 1 - height_std))  # Lower variation = higher score
        else:
            features.append(0.5)

        # Ensure we have exactly 20 features
        while len(features) < 20:
            features.append(0.0)

        return features[:20]

    def _extract_comprehensive_features(
        self, image: Image.Image, color_analysis: ColorPalette,
        typography_analysis: TypographyAnalysis, layout_analysis: LayoutAnalysis
    ) -> Dict[str, float]:
        """
        Extract comprehensive features including analysis results
        """
        basic_features = self._extract_features(image)

        comprehensive_features = {
            # Basic image features
            "color_variance": basic_features[3],
            "edge_density": basic_features[5],
            "brightness": basic_features[7],
            "contrast": basic_features[6],

            # Color analysis features
            "color_harmony_score": color_analysis.harmony_score,
            "accessibility_score": color_analysis.accessibility_score,
            "color_scheme_quality": self._score_color_scheme(color_analysis.color_scheme),

            # Typography features
            "font_pairing_score": typography_analysis.font_pairing_score,
            "readability_score": typography_analysis.readability_score,
            "text_hierarchy_score": typography_analysis.text_hierarchy_score,

            # Layout features
            "composition_score": layout_analysis.composition_score,
            "balance_score": layout_analysis.balance_score,
            "grid_alignment": layout_analysis.grid_alignment,
            "white_space_usage": layout_analysis.white_space_usage,

            # Derived features
            "focal_point_strength": self._calculate_focal_point_strength(layout_analysis.focal_points),
            "visual_complexity": basic_features[5] + basic_features[6],  # Edge + contrast
            "professional_appearance": (typography_analysis.font_pairing_score + layout_analysis.grid_alignment) / 2
        }

        return comprehensive_features

    def _score_color_scheme(self, scheme_type: str) -> float:
        """
        Score color scheme type for engagement
        """
        scheme_scores = {
            "complementary": 90.0,
            "triadic": 85.0,
            "analogous": 80.0,
            "monochromatic": 70.0,
            "custom": 60.0
        }
        return scheme_scores.get(scheme_type, 60.0)

    def _calculate_focal_point_strength(self, focal_points: List[Dict[str, float]]) -> float:
        """
        Calculate average focal point strength
        """
        if not focal_points:
            return 50.0

        strengths = [point["strength"] for point in focal_points]
        return np.mean(strengths) * 100

    def _calculate_prediction_confidence(self, features: np.ndarray, platform: str) -> float:
        """
        Calculate prediction confidence based on model uncertainty
        """
        # Use ensemble predictions for confidence estimation
        model = self.models[platform]

        # Get predictions from individual trees
        tree_predictions = [tree.predict(features)[0] for tree in model.estimators_]

        # Calculate standard deviation as uncertainty measure
        prediction_std = np.std(tree_predictions)

        # Convert to confidence (higher std = lower confidence)
        confidence = max(0, min(100, 100 - prediction_std))

        return confidence

    def _get_feature_importance(self, platform: str) -> Dict[str, float]:
        """
        Get feature importance for the platform model
        """
        model = self.models[platform]
        feature_names = [
            "red_avg", "green_avg", "blue_avg", "color_variance", "saturation",
            "edge_density", "contrast", "brightness", "color_diversity", "gradient",
            "aspect_ratio", "symmetry", "rule_of_thirds", "white_space", "center_weight",
            "text_ratio", "horizontal_lines", "vertical_lines", "text_regions", "line_uniformity"
        ]

        importance_dict = {}
        for i, importance in enumerate(model.feature_importances_):
            if i < len(feature_names):
                importance_dict[feature_names[i]] = float(importance)

        return importance_dict

    def _calculate_improvement_potential(self, features: Dict[str, float], platform: str) -> float:
        """
        Calculate potential for improvement based on current scores
        """
        # Identify weak areas
        weak_areas = []

        if features.get("color_harmony_score", 100) < 70:
            weak_areas.append("color")
        if features.get("readability_score", 100) < 70:
            weak_areas.append("typography")
        if features.get("composition_score", 100) < 70:
            weak_areas.append("layout")
        if features.get("accessibility_score", 100) < 70:
            weak_areas.append("accessibility")

        # Calculate improvement potential
        if not weak_areas:
            return 20.0  # Already good, small improvement potential

        # More weak areas = higher improvement potential
        base_potential = len(weak_areas) * 20

        # Platform-specific adjustments
        platform_weights = self.platform_weights.get(platform, self.platform_weights["general"])

        weighted_potential = 0
        for area in weak_areas:
            if area in platform_weights:
                weighted_potential += platform_weights[area] * 100

        improvement_potential = min(80, max(base_potential, weighted_potential))

        return improvement_potential