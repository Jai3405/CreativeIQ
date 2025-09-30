"""
End-to-End Training Pipeline for CreativeIQ
Orchestrates data collection, model training, and deployment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import json

from app.ml.dataset_collector import DatasetCollector
from app.ml.model_trainer import PerformanceModelTrainer
from app.ml.production_predictor import production_predictor

logger = logging.getLogger(__name__)


class CreativeIQTrainingPipeline:
    """
    Complete training pipeline from data collection to model deployment
    """

    def __init__(self):
        self.collector = DatasetCollector()
        self.trainer = PerformanceModelTrainer()
        self.pipeline_dir = Path("data/pipeline")
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

    async def run_full_pipeline(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        """
        if config is None:
            config = self._get_default_config()

        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = {"pipeline_id": pipeline_id, "start_time": datetime.now().isoformat()}

        logger.info(f"🚀 Starting CreativeIQ Training Pipeline: {pipeline_id}")

        try:
            # Step 1: Data Collection
            if config.get("collect_data", True):
                logger.info("📊 Step 1: Data Collection")
                collection_results = await self._run_data_collection(config["data_sources"])
                results["data_collection"] = collection_results
            else:
                logger.info("⏭️ Skipping data collection")

            # Step 2: Data Validation and Preprocessing
            logger.info("🔍 Step 2: Data Validation")
            validation_results = await self._validate_data()
            results["data_validation"] = validation_results

            if not validation_results["is_valid"]:
                raise ValueError("Data validation failed")

            # Step 3: Model Training
            logger.info("🧠 Step 3: Model Training")
            training_results = await self._run_model_training(config["training"])
            results["model_training"] = training_results

            # Step 4: Model Evaluation
            logger.info("📈 Step 4: Model Evaluation")
            evaluation_results = await self._evaluate_models()
            results["model_evaluation"] = evaluation_results

            # Step 5: Model Deployment
            if config.get("deploy_models", True):
                logger.info("🚀 Step 5: Model Deployment")
                deployment_results = await self._deploy_models()
                results["model_deployment"] = deployment_results

            # Step 6: Pipeline Validation
            logger.info("✅ Step 6: Pipeline Validation")
            validation_results = await self._validate_pipeline()
            results["pipeline_validation"] = validation_results

            results["status"] = "success"
            results["end_time"] = datetime.now().isoformat()

            # Save pipeline results
            self._save_pipeline_results(results)

            logger.info(f"✅ Pipeline completed successfully: {pipeline_id}")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()

        return results

    async def _run_data_collection(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run production data collection with real APIs
        """
        collection_results = {"sources": {}, "total_samples": 0, "api_status": {}}

        # Check API status first
        logger.info("🔑 Checking API credentials...")
        api_status = await self.collector.validate_all_tokens()
        collection_results["api_status"] = api_status

        valid_apis = [platform for platform, valid in api_status.items() if valid]
        if not valid_apis:
            logger.warning("⚠️ No valid API credentials found. Data collection will be limited.")

        # Collect Instagram data (REAL API)
        if data_sources.get("instagram", {}).get("enabled", False) and api_status.get("instagram", False):
            logger.info("📸 Collecting real Instagram data...")
            instagram_config = data_sources["instagram"]
            instagram_data = await self.collector.collect_instagram_data(
                user_ids=instagram_config.get("user_ids"),
                business_accounts=instagram_config.get("business_accounts", ["me"]),
                limit=instagram_config.get("limit", 100)
            )
            if instagram_data:
                self.collector.save_dataset(instagram_data, "instagram_production")
                collection_results["sources"]["instagram"] = len(instagram_data)
                logger.info(f"✅ Collected {len(instagram_data)} Instagram posts")
            else:
                logger.warning("⚠️ No Instagram data collected")
        elif data_sources.get("instagram", {}).get("enabled", False):
            logger.warning("⚠️ Instagram enabled but API token invalid")

        # Collect LinkedIn data (REAL API)
        if data_sources.get("linkedin", {}).get("enabled", False) and api_status.get("linkedin", False):
            logger.info("💼 Collecting real LinkedIn data...")
            linkedin_config = data_sources["linkedin"]
            linkedin_data = await self.collector.collect_linkedin_data(
                company_ids=linkedin_config.get("company_ids"),
                user_ids=linkedin_config.get("user_ids"),
                days_back=linkedin_config.get("days_back", 30)
            )
            if linkedin_data:
                self.collector.save_dataset(linkedin_data, "linkedin_production")
                collection_results["sources"]["linkedin"] = len(linkedin_data)
                logger.info(f"✅ Collected {len(linkedin_data)} LinkedIn posts")
            else:
                logger.warning("⚠️ No LinkedIn data collected")
        elif data_sources.get("linkedin", {}).get("enabled", False):
            logger.warning("⚠️ LinkedIn enabled but API token invalid")

        # Collect Facebook data (REAL API)
        if data_sources.get("facebook", {}).get("enabled", False) and api_status.get("facebook", False):
            logger.info("📘 Collecting real Facebook data...")
            facebook_config = data_sources["facebook"]
            facebook_data = await self.collector.collect_facebook_data(
                page_ids=facebook_config.get("page_ids"),
                ad_account_ids=facebook_config.get("ad_account_ids"),
                limit=facebook_config.get("limit", 75)
            )
            if facebook_data:
                self.collector.save_dataset(facebook_data, "facebook_production")
                collection_results["sources"]["facebook"] = len(facebook_data)
                logger.info(f"✅ Collected {len(facebook_data)} Facebook posts")
            else:
                logger.warning("⚠️ No Facebook data collected")
        elif data_sources.get("facebook", {}).get("enabled", False):
            logger.warning("⚠️ Facebook enabled but API token invalid")

        # Collect supplementary design platform data (if enabled)
        if data_sources.get("design_platforms", {}).get("enabled", False):
            logger.info("🎨 Collecting supplementary design platform data...")
            design_config = data_sources["design_platforms"]
            design_data = await self.collector.collect_design_platform_data(
                platforms=design_config.get("platforms", ["dribbble"])
            )
            if design_data:
                self.collector.save_dataset(design_data, "design_platforms_supplementary")
                collection_results["sources"]["design_platforms"] = len(design_data)
                logger.info(f"✅ Collected {len(design_data)} design platform samples")

        collection_results["total_samples"] = sum(collection_results["sources"].values())

        if collection_results["total_samples"] == 0:
            logger.error("❌ No data collected! Check API credentials and configuration.")

        return collection_results

    async def _validate_data(self) -> Dict[str, Any]:
        """
        Validate collected data quality
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "statistics": {},
            "recommendations": []
        }

        try:
            # Load and combine all datasets
            df, stats = self.collector.create_training_dataset()

            # Check minimum sample requirements
            min_samples = 100
            if len(df) < min_samples:
                validation_results["issues"].append(f"Insufficient samples: {len(df)} < {min_samples}")
                validation_results["is_valid"] = False

            # Check platform distribution
            platform_counts = df["platform"].value_counts()
            for platform, count in platform_counts.items():
                if count < 20:
                    validation_results["issues"].append(f"Low sample count for {platform}: {count}")

            # Check for missing engagement metrics
            engagement_cols = [col for col in df.columns if "engagement_metrics" in col]
            for col in engagement_cols:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > 50:
                    validation_results["issues"].append(f"High missing data in {col}: {missing_pct:.1f}%")

            # Check image file existence
            missing_images = 0
            for image_path in df["image_path"].dropna():
                if not Path(image_path).exists():
                    missing_images += 1

            if missing_images > 0:
                validation_results["issues"].append(f"Missing image files: {missing_images}")

            # Set validation status
            if len(validation_results["issues"]) > 3:
                validation_results["is_valid"] = False

            # Statistics
            validation_results["statistics"] = {
                "total_samples": len(df),
                "platforms": platform_counts.to_dict(),
                "date_range": stats["date_range"],
                "missing_images": missing_images,
                "data_quality_score": max(0, 100 - len(validation_results["issues"]) * 10)
            }

            # Recommendations
            if missing_images > 0:
                validation_results["recommendations"].append("Re-run data collection for missing images")

            if len(df) < 500:
                validation_results["recommendations"].append("Collect more data for better model performance")

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")

        return validation_results

    async def _run_model_training(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run model training step
        """
        training_results = {"models_trained": {}, "best_models": {}}

        try:
            # Load training data
            df, _ = self.collector.create_training_dataset()

            # Train platform-specific models
            if training_config.get("platform_specific", True):
                all_results = self.trainer.train_platform_specific_models(df)
                training_results["models_trained"] = all_results

                # Identify best models
                for platform, platform_results in all_results.items():
                    if platform_results:
                        best_model = max(
                            platform_results.keys(),
                            key=lambda k: platform_results[k]["metrics"]["r2_score"]
                        )
                        training_results["best_models"][platform] = {
                            "model_type": best_model,
                            "r2_score": platform_results[best_model]["metrics"]["r2_score"],
                            "rmse": platform_results[best_model]["metrics"]["rmse"]
                        }

            # Train general model
            if training_config.get("general_model", True):
                X, y, feature_cols = self.trainer.prepare_training_data(df)
                general_results = self.trainer.train_models(X, y, platform="general")
                training_results["models_trained"]["general"] = general_results

        except Exception as e:
            training_results["error"] = str(e)
            logger.error(f"Training failed: {e}")

        return training_results

    async def _evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate trained models
        """
        evaluation_results = {"platform_performance": {}, "overall_quality": "unknown"}

        try:
            # Load test data (most recent 20%)
            df, _ = self.collector.create_training_dataset()
            test_size = max(20, int(len(df) * 0.2))
            test_df = df.tail(test_size)

            # Evaluate each platform model
            models_dir = Path("models/trained")
            for model_file in models_dir.glob("*_performance_model.joblib"):
                platform = model_file.stem.replace("_performance_model", "")

                try:
                    # Prepare test data for this platform
                    if platform != "general":
                        platform_test = test_df[test_df["platform"] == platform]
                        if len(platform_test) < 5:
                            continue
                    else:
                        platform_test = test_df

                    X, y, _ = self.trainer.prepare_training_data(platform_test)
                    if len(X) == 0:
                        continue

                    # Evaluate model
                    metrics = self.trainer.evaluate_model(model_file, X, y)
                    evaluation_results["platform_performance"][platform] = metrics

                except Exception as e:
                    logger.warning(f"Evaluation failed for {platform}: {e}")

            # Determine overall quality
            if evaluation_results["platform_performance"]:
                avg_r2 = np.mean([
                    metrics["r2_score"] for metrics in evaluation_results["platform_performance"].values()
                ])

                if avg_r2 > 0.7:
                    evaluation_results["overall_quality"] = "excellent"
                elif avg_r2 > 0.5:
                    evaluation_results["overall_quality"] = "good"
                elif avg_r2 > 0.3:
                    evaluation_results["overall_quality"] = "fair"
                else:
                    evaluation_results["overall_quality"] = "poor"

        except Exception as e:
            evaluation_results["error"] = str(e)

        return evaluation_results

    async def _deploy_models(self) -> Dict[str, Any]:
        """
        Deploy models to production
        """
        deployment_results = {"deployed_models": [], "deployment_status": "unknown"}

        try:
            # Reload production predictor to pick up new models
            production_predictor._load_models()

            # Get model info
            model_info = production_predictor.get_model_info()
            deployment_results["deployed_models"] = model_info["models_loaded"]

            if model_info["models_loaded"]:
                deployment_results["deployment_status"] = "success"
            else:
                deployment_results["deployment_status"] = "no_models"

        except Exception as e:
            deployment_results["deployment_status"] = "failed"
            deployment_results["error"] = str(e)

        return deployment_results

    async def _validate_pipeline(self) -> Dict[str, Any]:
        """
        Validate the complete pipeline
        """
        validation_results = {"pipeline_health": "unknown", "tests_passed": []}

        try:
            # Test 1: Model loading
            model_info = production_predictor.get_model_info()
            if model_info["models_loaded"]:
                validation_results["tests_passed"].append("model_loading")

            # Test 2: Prediction functionality
            from PIL import Image
            test_image = Image.new("RGB", (500, 500), "red")

            # Mock analysis results for testing
            from app.models.schemas import ColorPalette, TypographyAnalysis, LayoutAnalysis

            mock_color = ColorPalette(
                dominant_colors=["#FF0000", "#FFFFFF"],
                color_scheme="complementary",
                harmony_score=85.0,
                accessibility_score=90.0
            )

            mock_typography = TypographyAnalysis(
                fonts_detected=["sans-serif"],
                font_pairing_score=80.0,
                readability_score=85.0,
                text_hierarchy_score=75.0
            )

            mock_layout = LayoutAnalysis(
                composition_score=80.0,
                balance_score=75.0,
                grid_alignment=70.0,
                white_space_usage=65.0,
                focal_points=[]
            )

            # Test prediction
            prediction = await production_predictor.predict_comprehensive(
                test_image, "general", mock_color, mock_typography, mock_layout
            )

            if prediction and "engagement_score" in prediction:
                validation_results["tests_passed"].append("prediction_test")

            # Determine overall health
            if len(validation_results["tests_passed"]) >= 2:
                validation_results["pipeline_health"] = "healthy"
            elif len(validation_results["tests_passed"]) >= 1:
                validation_results["pipeline_health"] = "partial"
            else:
                validation_results["pipeline_health"] = "unhealthy"

        except Exception as e:
            validation_results["pipeline_health"] = "error"
            validation_results["error"] = str(e)

        return validation_results

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default pipeline configuration
        """
        return {
            "collect_data": True,
            "deploy_models": True,
            "data_sources": {
                "instagram": {
                    "enabled": False,  # Requires API keys
                    "user_ids": [],
                    "days_back": 30
                },
                "linkedin": {
                    "enabled": False,  # Requires API keys
                    "company_ids": [],
                    "days_back": 30
                },
                "design_platforms": {
                    "enabled": True,
                    "platforms": ["dribbble"]
                },
                "ab_tests": {
                    "enabled": True,
                    "test_configs": [
                        {"test_id": "demo_test_1", "platform": "instagram"},
                        {"test_id": "demo_test_2", "platform": "linkedin"}
                    ]
                }
            },
            "training": {
                "platform_specific": True,
                "general_model": True,
                "cross_validation": True
            }
        }

    def _save_pipeline_results(self, results: Dict[str, Any]):
        """
        Save pipeline results to disk
        """
        results_file = self.pipeline_dir / f"{results['pipeline_id']}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Pipeline results saved to: {results_file}")

    async def run_incremental_training(self, new_data_days: int = 7) -> Dict[str, Any]:
        """
        Run incremental training with new data
        """
        logger.info("🔄 Starting incremental training...")

        # Collect only recent data
        recent_data = await self._collect_recent_data(new_data_days)

        if not recent_data:
            return {"status": "no_new_data"}

        # Retrain models with combined data
        results = {}
        for platform in ["general", "instagram", "linkedin"]:
            if platform in production_predictor.models:
                success = await production_predictor.retrain_model(platform, recent_data)
                results[platform] = "success" if success else "failed"

        return {"status": "completed", "retrained_models": results}

    async def _collect_recent_data(self, days: int) -> pd.DataFrame:
        """
        Collect data from recent days only
        """
        # This would collect only recent data
        # For demo purposes, return empty DataFrame
        return pd.DataFrame()


# Convenience functions for common operations

async def run_training_pipeline(config: Dict[str, Any] = None):
    """
    Run the complete training pipeline
    """
    pipeline = CreativeIQTrainingPipeline()
    return await pipeline.run_full_pipeline(config)


async def run_quick_training():
    """
    Run training with minimal data collection
    """
    config = {
        "collect_data": True,
        "deploy_models": True,
        "data_sources": {
            "design_platforms": {"enabled": True, "platforms": ["dribbble"]},
            "ab_tests": {"enabled": True, "test_configs": [{"test_id": "quick_test", "platform": "general"}]}
        },
        "training": {"platform_specific": False, "general_model": True}
    }

    pipeline = CreativeIQTrainingPipeline()
    return await pipeline.run_full_pipeline(config)


if __name__ == "__main__":
    # Run training pipeline
    asyncio.run(run_quick_training())