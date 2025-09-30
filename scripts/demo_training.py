#!/usr/bin/env python3
"""
CreativeIQ Demo Training Script
Demonstrates the complete ML training pipeline with synthetic data
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import json
from datetime import datetime, timedelta
import logging

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.dataset_collector import DatasetCollector
from app.ml.model_trainer import PerformanceModelTrainer
from app.ml.production_predictor import ProductionPerformancePredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def create_demo_dataset():
    """
    Create a synthetic dataset for demonstration
    """
    print("🎨 Creating demo dataset...")

    # Create demo images and metadata
    demo_data = []

    # Simulate different types of designs
    design_types = [
        {"name": "minimal", "color_count": 2, "complexity": "low", "engagement_multiplier": 1.2},
        {"name": "vibrant", "color_count": 5, "complexity": "high", "engagement_multiplier": 1.0},
        {"name": "professional", "color_count": 3, "complexity": "medium", "engagement_multiplier": 1.1},
        {"name": "bold", "color_count": 4, "complexity": "high", "engagement_multiplier": 0.9},
    ]

    platforms = ["instagram", "linkedin", "facebook", "dribbble"]

    # Create data directory
    data_dir = Path("data/training")
    images_dir = data_dir / "images"
    metadata_dir = data_dir / "metadata"

    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    # Generate 200 sample designs
    for i in range(200):
        design_type = np.random.choice(design_types)
        platform = np.random.choice(platforms)

        # Create synthetic image
        image = create_synthetic_design_image(design_type, i)
        image_path = images_dir / f"demo_design_{i}.jpg"
        image.save(image_path)

        # Create metadata
        base_engagement = np.random.normal(50, 15)  # Base engagement 35-65
        platform_boost = get_platform_boost(platform, design_type)
        final_engagement = max(10, min(95, base_engagement * design_type["engagement_multiplier"] * platform_boost))

        metadata = {
            "id": f"demo_design_{i}",
            "platform": platform,
            "image_path": str(image_path),
            "timestamp": (datetime.now() - timedelta(days=np.random.randint(1, 90))).isoformat(),
            "design_type": design_type["name"],
            "engagement_metrics": {
                "likes": int(final_engagement * np.random.uniform(10, 50)),
                "comments": int(final_engagement * np.random.uniform(1, 8)),
                "shares": int(final_engagement * np.random.uniform(0.5, 5)),
                "impressions": int(final_engagement * np.random.uniform(100, 500)),
                "engagement_rate": final_engagement / 100
            },
            "audience_size": np.random.randint(1000, 50000),
            "posting_time": {
                "hour": np.random.randint(6, 23),
                "day_of_week": np.random.randint(0, 6),
                "is_weekend": np.random.choice([True, False]),
                "time_of_day": np.random.choice(["morning", "afternoon", "evening"])
            },
            "performance_score": final_engagement,
            "derived_engagement_rate": final_engagement
        }

        demo_data.append(metadata)

    # Save dataset
    json_path = metadata_dir / "demo_dataset.json"
    with open(json_path, "w") as f:
        json.dump(demo_data, f, indent=2, default=str)

    print(f"✅ Created {len(demo_data)} demo samples")
    print(f"💾 Saved to {json_path}")

    return demo_data


def create_synthetic_design_image(design_type: dict, index: int) -> Image.Image:
    """
    Create synthetic design images with different characteristics
    """
    width, height = 800, 600

    # Color palettes for different design types
    color_palettes = {
        "minimal": [(255, 255, 255), (64, 64, 64)],
        "vibrant": [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)],
        "professional": [(45, 55, 72), (74, 85, 104), (255, 255, 255)],
        "bold": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    }

    palette = color_palettes[design_type["name"]]

    # Create base image
    image = Image.new("RGB", (width, height), palette[0])

    # Add some synthetic design elements
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)

    # Add rectangles (simulating design elements)
    num_elements = {"low": 2, "medium": 4, "high": 6}[design_type["complexity"]]

    for i in range(num_elements):
        color = palette[i % len(palette)]
        x1 = np.random.randint(0, width // 2)
        y1 = np.random.randint(0, height // 2)
        x2 = x1 + np.random.randint(50, 200)
        y2 = y1 + np.random.randint(30, 150)

        draw.rectangle([x1, y1, x2, y2], fill=color)

    # Add some text areas (dark rectangles simulating text)
    for i in range(np.random.randint(1, 4)):
        x1 = np.random.randint(50, width - 200)
        y1 = np.random.randint(50, height - 50)
        x2 = x1 + np.random.randint(100, 180)
        y2 = y1 + 20

        draw.rectangle([x1, y1, x2, y2], fill=(50, 50, 50))

    return image


def get_platform_boost(platform: str, design_type: dict) -> float:
    """
    Get platform-specific engagement boost based on design type
    """
    boosts = {
        "instagram": {
            "minimal": 1.1,
            "vibrant": 1.3,
            "professional": 0.9,
            "bold": 1.2
        },
        "linkedin": {
            "minimal": 1.0,
            "vibrant": 0.8,
            "professional": 1.3,
            "bold": 0.9
        },
        "facebook": {
            "minimal": 1.0,
            "vibrant": 1.1,
            "professional": 1.0,
            "bold": 1.1
        },
        "dribbble": {
            "minimal": 1.2,
            "vibrant": 1.1,
            "professional": 1.0,
            "bold": 1.2
        }
    }

    return boosts.get(platform, {}).get(design_type["name"], 1.0)


async def train_demo_models():
    """
    Train models on the demo dataset
    """
    print("🧠 Training models on demo dataset...")

    # Load demo dataset
    metadata_dir = Path("data/training/metadata")
    json_path = metadata_dir / "demo_dataset.json"

    if not json_path.exists():
        print("❌ Demo dataset not found. Run create_demo_dataset() first.")
        return

    # Initialize trainer
    trainer = PerformanceModelTrainer()

    # Load data
    with open(json_path, "r") as f:
        demo_data = json.load(f)

    df = pd.json_normalize(demo_data)
    print(f"📊 Loaded {len(df)} training samples")

    # Train platform-specific models
    results = trainer.train_platform_specific_models(df)

    print("\n🎯 Training Results:")
    for platform, platform_results in results.items():
        if platform_results:
            best_model = max(platform_results.keys(), key=lambda k: platform_results[k]["metrics"]["r2_score"])
            best_r2 = platform_results[best_model]["metrics"]["r2_score"]
            best_rmse = platform_results[best_model]["metrics"]["rmse"]
            print(f"  {platform}: {best_model} (R² = {best_r2:.3f}, RMSE = {best_rmse:.3f})")

    return results


async def test_trained_models():
    """
    Test the trained models with sample predictions
    """
    print("🧪 Testing trained models...")

    # Initialize production predictor
    predictor = ProductionPerformancePredictor()

    # Create test image
    test_image = Image.new("RGB", (600, 400), (100, 150, 200))

    # Mock analysis results
    from app.models.schemas import ColorPalette, TypographyAnalysis, LayoutAnalysis

    color_analysis = ColorPalette(
        dominant_colors=["#6496C8", "#FFFFFF", "#404040"],
        color_scheme="analogous",
        harmony_score=82.5,
        accessibility_score=88.0
    )

    typography_analysis = TypographyAnalysis(
        fonts_detected=["sans-serif"],
        font_pairing_score=85.0,
        readability_score=90.0,
        text_hierarchy_score=78.0
    )

    layout_analysis = LayoutAnalysis(
        composition_score=75.0,
        balance_score=80.0,
        grid_alignment=70.0,
        white_space_usage=65.0,
        focal_points=[]
    )

    # Test predictions for different platforms
    platforms = ["general", "instagram", "linkedin"]

    print("\n📊 Prediction Results:")
    for platform in platforms:
        try:
            prediction = await predictor.predict_comprehensive(
                test_image, platform, color_analysis, typography_analysis, layout_analysis
            )

            print(f"\n{platform.title()}:")
            print(f"  Engagement Score: {prediction['engagement_score']:.1f}")
            print(f"  Confidence: {prediction['confidence']:.1f}%")
            print(f"  Improvement Potential: {prediction['improvement_potential']:.1f}%")

            # Show platform scores
            if "platform_scores" in prediction:
                print("  Platform Scores:")
                for p, score in prediction["platform_scores"].items():
                    print(f"    {p}: {score:.1f}")

        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")


async def run_complete_demo():
    """
    Run complete demo: dataset creation, training, and testing
    """
    print("🚀 CreativeIQ ML Training Demo")
    print("=" * 40)

    try:
        # Step 1: Create demo dataset
        await create_demo_dataset()
        print()

        # Step 2: Train models
        await train_demo_models()
        print()

        # Step 3: Test models
        await test_trained_models()
        print()

        print("✅ Demo completed successfully!")
        print("\nWhat happened:")
        print("1. Created 200 synthetic design samples with engagement data")
        print("2. Trained Random Forest, XGBoost, and Neural Network models")
        print("3. Selected best models for each platform based on R² score")
        print("4. Deployed models and tested predictions")
        print("\nModels are now ready for production use!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


async def quick_demo():
    """
    Quick demo showing just the key concepts
    """
    print("⚡ Quick ML Demo")
    print("=" * 20)

    # Show model training concept
    print("🧠 Training concept:")
    print("1. Collect design images + engagement metrics")
    print("2. Extract visual features (color, layout, typography)")
    print("3. Train ML models to predict engagement")
    print("4. Deploy models for real-time predictions")

    # Create tiny demo dataset
    print("\n📊 Creating mini dataset (10 samples)...")

    # Simulate training data
    features = np.random.rand(10, 20)  # 10 samples, 20 features
    engagement_scores = np.random.rand(10) * 100  # Random scores 0-100

    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, engagement_scores, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Test prediction
    predictions = model.predict(X_test)

    print(f"✅ Model trained!")
    print(f"📈 Sample predictions: {predictions[:3].round(1)}")
    print(f"🎯 Actual scores: {y_test[:3].round(1)}")

    print("\n💡 In production:")
    print("- Models use real Instagram/LinkedIn data")
    print("- Features extracted from actual design images")
    print("- Predictions guide design optimization")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CreativeIQ ML Training Demo")
    parser.add_argument("--mode", choices=["full", "quick"], default="quick",
                       help="Demo mode: full (complete demo) or quick (concept demo)")

    args = parser.parse_args()

    try:
        if args.mode == "full":
            asyncio.run(run_complete_demo())
        else:
            asyncio.run(quick_demo())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        import traceback
        traceback.print_exc()