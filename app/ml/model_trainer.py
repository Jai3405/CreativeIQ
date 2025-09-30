"""
ML Model Training Pipeline for CreativeIQ
Trains performance prediction models using collected design data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import joblib
import json
from datetime import datetime
import logging

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Computer Vision for Feature Extraction
import cv2
from PIL import Image
from app.services.image_processor import ImageProcessor
from app.services.typography_analyzer import TypographyAnalyzer

logger = logging.getLogger(__name__)


class PerformanceModelTrainer:
    """
    Trains ML models to predict design performance based on visual features
    """

    def __init__(self):
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.image_processor = ImageProcessor()
        self.typography_analyzer = TypographyAnalyzer()

        # Available models
        self.model_configs = {
            "random_forest": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor,
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.05, 0.1, 0.15],
                    "max_depth": [3, 5, 7]
                }
            },
            "xgboost": {
                "model": xgb.XGBRegressor,
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.05, 0.1, 0.15],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0]
                }
            },
            "neural_network": {
                "model": MLPRegressor,
                "params": {
                    "hidden_layer_sizes": [(100,), (100, 50), (200, 100, 50)],
                    "alpha": [0.001, 0.01, 0.1],
                    "learning_rate": ["constant", "adaptive"]
                }
            }
        }

    def extract_features_from_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features from images in the dataset
        """
        print("🔍 Extracting visual features from images...")

        features_list = []

        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Progress: {idx}/{len(df)} images processed")

            try:
                # Load image
                image_path = Path(row["image_path"])
                if not image_path.exists():
                    print(f"⚠️ Image not found: {image_path}")
                    continue

                image = Image.open(image_path)

                # Extract visual features
                features = self._extract_comprehensive_features(image, row)
                features["sample_id"] = idx
                features_list.append(features)

            except Exception as e:
                logger.error(f"Error processing image {row.get('image_path', 'unknown')}: {e}")
                continue

        features_df = pd.DataFrame(features_list)
        print(f"✅ Feature extraction complete: {len(features_df)} samples with {len(features_df.columns)} features")

        return features_df

    def _extract_comprehensive_features(self, image: Image.Image, metadata: pd.Series) -> Dict[str, float]:
        """
        Extract comprehensive visual and contextual features
        """
        features = {}

        # 1. Basic Image Properties
        width, height = image.size
        features["image_width"] = width
        features["image_height"] = height
        features["aspect_ratio"] = width / height
        features["image_area"] = width * height

        # 2. Color Features
        try:
            colors = self.image_processor.extract_color_palette(image, num_colors=8)
            color_analysis = self.image_processor.analyze_color_harmony(colors)

            features["num_dominant_colors"] = len(colors)
            features["color_harmony_score"] = color_analysis["harmony_score"]
            features["dominant_hue"] = color_analysis["dominant_hue"]
            features["saturation_range"] = color_analysis["saturation_range"][1] - color_analysis["saturation_range"][0]
            features["brightness_range"] = color_analysis["brightness_range"][1] - color_analysis["brightness_range"][0]

            # Color scheme encoding
            color_scheme_map = {"complementary": 1, "triadic": 2, "analogous": 3, "monochromatic": 4, "custom": 5}
            features["color_scheme_encoded"] = color_scheme_map.get(color_analysis["scheme_type"], 5)

        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            for key in ["num_dominant_colors", "color_harmony_score", "dominant_hue", "saturation_range", "brightness_range", "color_scheme_encoded"]:
                features[key] = 0

        # 3. Typography Features
        try:
            typography_analysis = self.typography_analyzer.analyze_typography(image)

            features["num_fonts_detected"] = len(typography_analysis["fonts_detected"])
            features["font_pairing_score"] = typography_analysis["font_pairing_score"]
            features["readability_score"] = typography_analysis["readability_score"]
            features["text_hierarchy_score"] = typography_analysis["text_hierarchy_score"]
            features["num_text_regions"] = len(typography_analysis["text_regions"])

        except Exception as e:
            logger.warning(f"Typography analysis failed: {e}")
            for key in ["num_fonts_detected", "font_pairing_score", "readability_score", "text_hierarchy_score", "num_text_regions"]:
                features[key] = 0

        # 4. Composition Features
        try:
            composition_data = self.image_processor.analyze_composition(image)

            features["rule_of_thirds_score"] = composition_data["rule_of_thirds_score"]
            features["balance_score"] = composition_data["balance_score"]
            features["num_focal_points"] = len(composition_data["focal_points"])
            features["focal_point_strength"] = np.mean([fp["strength"] for fp in composition_data["focal_points"]]) if composition_data["focal_points"] else 0

            # Visual weight distribution
            weight_dist = composition_data["visual_weight_distribution"]
            features["weight_top"] = weight_dist["top"]
            features["weight_center"] = weight_dist["center"]
            features["weight_bottom"] = weight_dist["bottom"]

        except Exception as e:
            logger.warning(f"Composition analysis failed: {e}")
            for key in ["rule_of_thirds_score", "balance_score", "num_focal_points", "focal_point_strength", "weight_top", "weight_center", "weight_bottom"]:
                features[key] = 0

        # 5. Basic CV Features
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features["edge_density"] = np.sum(edges > 0) / edges.size

            # Texture features
            features["image_contrast"] = np.std(gray) / 255.0
            features["image_brightness"] = np.mean(gray) / 255.0

            # Color variance
            features["color_variance"] = np.var(np.array(image)) / (255**2)

        except Exception as e:
            logger.warning(f"CV analysis failed: {e}")
            for key in ["edge_density", "image_contrast", "image_brightness", "color_variance"]:
                features[key] = 0

        # 6. Contextual Features from Metadata
        features["platform_instagram"] = 1 if metadata.get("platform") == "instagram" else 0
        features["platform_linkedin"] = 1 if metadata.get("platform") == "linkedin" else 0
        features["platform_dribbble"] = 1 if metadata.get("platform") == "dribbble" else 0

        # Time-based features
        if "posting_time" in metadata and isinstance(metadata["posting_time"], dict):
            posting_time = metadata["posting_time"]
            features["posting_hour"] = posting_time.get("hour", 12)
            features["is_weekend"] = 1 if posting_time.get("is_weekend", False) else 0
            features["is_morning"] = 1 if posting_time.get("time_of_day") == "morning" else 0
            features["is_evening"] = 1 if posting_time.get("time_of_day") == "evening" else 0
        else:
            features.update({"posting_hour": 12, "is_weekend": 0, "is_morning": 0, "is_evening": 0})

        # Content features
        features["has_text"] = 1 if features["num_text_regions"] > 0 else 0
        features["is_high_contrast"] = 1 if features.get("image_contrast", 0) > 0.5 else 0

        return features

    def prepare_training_data(self, df: pd.DataFrame, target_metric: str = "performance_score") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target variables for training
        """
        print(f"📊 Preparing training data with target: {target_metric}")

        # Extract features from images
        features_df = self.extract_features_from_dataset(df)

        # Merge with original metadata
        df_reset = df.reset_index(drop=True)
        combined_df = pd.concat([df_reset, features_df], axis=1)

        # Define target variable
        if target_metric == "performance_score":
            # Calculate performance score if not exists
            if "performance_score" not in combined_df.columns:
                combined_df["performance_score"] = self._calculate_performance_score(combined_df)
        elif target_metric == "engagement_rate":
            if "derived_engagement_rate" not in combined_df.columns:
                combined_df["derived_engagement_rate"] = self._calculate_engagement_rate(combined_df)
            target_metric = "derived_engagement_rate"

        # Remove rows with missing target
        combined_df = combined_df.dropna(subset=[target_metric])

        # Feature columns (exclude metadata and target)
        exclude_cols = [
            "sample_id", "id", "image_path", "timestamp", "caption", "content", "description",
            target_metric, "platform", "user_id", "company_id"
        ]
        feature_cols = [col for col in combined_df.columns if col not in exclude_cols and not col.startswith("engagement_metrics")]

        X = combined_df[feature_cols]
        y = combined_df[target_metric]

        print(f"✅ Training data prepared: {len(X)} samples, {len(feature_cols)} features")
        print(f"Target distribution: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")

        return X, y, feature_cols

    def train_models(self, X: pd.DataFrame, y: pd.Series, platform: str = "general") -> Dict[str, Any]:
        """
        Train multiple models and select the best one
        """
        print(f"🚀 Training models for platform: {platform}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}
        best_model = None
        best_score = -np.inf

        for model_name, config in self.model_configs.items():
            print(f"🔄 Training {model_name}...")

            try:
                # Create pipeline
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", config["model"](random_state=42))
                ])

                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid={f"model__{k}": v for k, v in config["params"].items()},
                    cv=5,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train, y_train)

                # Best model predictions
                y_pred = grid_search.predict(X_test)

                # Calculate metrics
                metrics = {
                    "r2_score": r2_score(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
                }

                # Cross-validation score
                cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring="r2")
                metrics["cv_r2_mean"] = cv_scores.mean()
                metrics["cv_r2_std"] = cv_scores.std()

                results[model_name] = {
                    "model": grid_search.best_estimator_,
                    "best_params": grid_search.best_params_,
                    "metrics": metrics,
                    "feature_importance": self._get_feature_importance(grid_search.best_estimator_, X.columns)
                }

                print(f"✅ {model_name}: R² = {metrics['r2_score']:.3f}, RMSE = {metrics['rmse']:.3f}")

                # Track best model
                if metrics["r2_score"] > best_score:
                    best_score = metrics["r2_score"]
                    best_model = model_name

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        if best_model:
            print(f"🏆 Best model: {best_model} (R² = {best_score:.3f})")

            # Save best model
            model_path = self.models_dir / f"{platform}_performance_model.joblib"
            joblib.dump(results[best_model]["model"], model_path)

            # Save scaler
            scaler_path = self.models_dir / f"{platform}_scaler.joblib"
            joblib.dump(scaler, scaler_path)

            # Save feature names
            features_path = self.models_dir / f"{platform}_features.json"
            with open(features_path, "w") as f:
                json.dump(list(X.columns), f)

            # Save training results
            results_path = self.models_dir / f"{platform}_training_results.json"
            serializable_results = {}
            for name, result in results.items():
                serializable_results[name] = {
                    "best_params": result["best_params"],
                    "metrics": result["metrics"],
                    "feature_importance": result["feature_importance"]
                }

            with open(results_path, "w") as f:
                json.dump(serializable_results, f, indent=2)

            print(f"💾 Models saved to {self.models_dir}")

        return results

    def train_platform_specific_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Train separate models for each platform
        """
        all_results = {}

        # Train general model (all platforms)
        print("🌍 Training general model...")
        X, y, feature_cols = self.prepare_training_data(df)
        general_results = self.train_models(X, y, platform="general")
        all_results["general"] = general_results

        # Train platform-specific models
        platforms = df["platform"].unique()
        for platform in platforms:
            if platform and df[df["platform"] == platform].shape[0] >= 50:  # Minimum samples
                print(f"🎯 Training {platform}-specific model...")
                platform_df = df[df["platform"] == platform]
                X_platform, y_platform, _ = self.prepare_training_data(platform_df)

                if len(X_platform) >= 20:  # Minimum for training
                    platform_results = self.train_models(X_platform, y_platform, platform=platform)
                    all_results[platform] = platform_results
                else:
                    print(f"⚠️ Insufficient data for {platform}: {len(X_platform)} samples")

        return all_results

    def _calculate_performance_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite performance score
        """
        # Normalize engagement metrics
        likes = df.get("engagement_metrics.likes", 0).fillna(0)
        comments = df.get("engagement_metrics.comments", 0).fillna(0)
        shares = df.get("engagement_metrics.shares", 0).fillna(0)
        impressions = df.get("engagement_metrics.impressions", 1).fillna(1)

        # Calculate engagement rate
        engagement_rate = (likes + comments * 2 + shares * 3) / impressions * 100

        # Calculate performance score (0-100)
        performance_score = np.clip(engagement_rate * 10, 0, 100)

        return performance_score

    def _calculate_engagement_rate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate platform-specific engagement rate
        """
        engagement_rate = pd.Series(0.0, index=df.index)

        for platform in df["platform"].unique():
            mask = df["platform"] == platform

            if platform == "instagram":
                likes = df.loc[mask, "engagement_metrics.likes"].fillna(0)
                comments = df.loc[mask, "engagement_metrics.comments"].fillna(0)
                followers = df.loc[mask, "audience_size"].fillna(1000)
                engagement_rate.loc[mask] = (likes + comments) / followers * 100

            elif platform == "linkedin":
                engagement_rate.loc[mask] = df.loc[mask, "engagement_metrics.engagement_rate"].fillna(0) * 100

            # Add other platforms as needed

        return engagement_rate

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from trained model
        """
        try:
            # Get the actual model from pipeline
            if hasattr(model, "named_steps"):
                actual_model = model.named_steps["model"]
            else:
                actual_model = model

            if hasattr(actual_model, "feature_importances_"):
                importance = actual_model.feature_importances_
            elif hasattr(actual_model, "coef_"):
                importance = np.abs(actual_model.coef_)
            else:
                return {}

            return dict(zip(feature_names, importance.tolist()))

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}

    def evaluate_model(self, model_path: Path, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate trained model on test data
        """
        # Load model
        model = joblib.load(model_path)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }

        return metrics


def train_production_models():
    """
    Main training script for production models
    """
    print("🚀 Starting CreativeIQ Model Training Pipeline")

    # Initialize trainer
    trainer = PerformanceModelTrainer()

    # Load dataset
    data_dir = Path("data/training/metadata")
    if not data_dir.exists():
        print("❌ No training data found. Run dataset collection first.")
        return

    # Combine all datasets
    all_data = []
    for json_file in data_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            all_data.extend(data)

    if not all_data:
        print("❌ No training data found in JSON files.")
        return

    # Convert to DataFrame
    df = pd.json_normalize(all_data)
    print(f"📊 Loaded {len(df)} training samples")

    # Train models
    results = trainer.train_platform_specific_models(df)

    # Print summary
    print("\n🎯 Training Summary:")
    for platform, platform_results in results.items():
        if platform_results:
            best_model = max(platform_results.keys(), key=lambda k: platform_results[k]["metrics"]["r2_score"])
            best_r2 = platform_results[best_model]["metrics"]["r2_score"]
            print(f"  {platform}: {best_model} (R² = {best_r2:.3f})")

    print("\n✅ Model training complete!")
    print(f"Models saved to: {trainer.models_dir}")


if __name__ == "__main__":
    train_production_models()