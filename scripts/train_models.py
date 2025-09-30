#!/usr/bin/env python3
"""
CreativeIQ Model Training Script
Trains ML models on collected design performance data
"""

import asyncio
import argparse
import sys
from pathlib import Path
import logging

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.training_pipeline import CreativeIQTrainingPipeline, run_quick_training, run_training_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main training script
    """
    parser = argparse.ArgumentParser(description="Train CreativeIQ performance prediction models")
    parser.add_argument("--mode", choices=["quick", "full", "incremental"], default="quick",
                       help="Training mode: quick (demo), full (production), or incremental")
    parser.add_argument("--platforms", nargs="+", default=["instagram", "linkedin", "facebook"],
                       help="Platforms to train models for")
    parser.add_argument("--collect-data", action="store_true",
                       help="Collect new data before training")
    parser.add_argument("--days-back", type=int, default=30,
                       help="Number of days of data to collect")

    args = parser.parse_args()

    print("🚀 CreativeIQ Model Training")
    print("=" * 50)

    if args.mode == "quick":
        print("🏃‍♂️ Running quick training (demo mode)")
        print("This will train models with synthetic data for demonstration")

        try:
            results = await run_quick_training()
            print_results(results)
        except Exception as e:
            print(f"❌ Quick training failed: {e}")
            return 1

    elif args.mode == "full":
        print("🏭 Running PRODUCTION training with real APIs")
        print("This requires valid API keys and will collect real social media data")
        print("Make sure your .env file contains valid API tokens\n")

        config = create_full_config(args)

        try:
            results = await run_training_pipeline(config)
            print_results(results)

            # Show API status in results
            if "data_collection" in results and "api_status" in results["data_collection"]:
                api_status = results["data_collection"]["api_status"]
                print(f"\n🔑 API Status:")
                for platform, valid in api_status.items():
                    emoji = "✅" if valid else "❌"
                    print(f"  {emoji} {platform}: {'Valid' if valid else 'Invalid/Missing'}")

        except Exception as e:
            print(f"❌ Full training failed: {e}")
            print("\nTroubleshooting:")
            print("1. Check your .env file has valid API tokens")
            print("2. Ensure API tokens have proper permissions")
            print("3. Check network connectivity")
            return 1

    elif args.mode == "incremental":
        print("🔄 Running incremental training")
        print("This will retrain existing models with new data")

        pipeline = CreativeIQTrainingPipeline()
        try:
            results = await pipeline.run_incremental_training(days=args.days_back)
            print_incremental_results(results)
        except Exception as e:
            print(f"❌ Incremental training failed: {e}")
            return 1

    print("\n✅ Training complete!")
    return 0


def create_full_config(args):
    """
    Create production training configuration with real APIs
    """
    return {
        "collect_data": args.collect_data,
        "deploy_models": True,
        "data_sources": {
            "instagram": {
                "enabled": "instagram" in args.platforms,
                "user_ids": get_instagram_accounts(),
                "business_accounts": get_instagram_business_accounts(),
                "limit": 100,
                "days_back": args.days_back
            },
            "linkedin": {
                "enabled": "linkedin" in args.platforms,
                "company_ids": get_linkedin_companies(),
                "user_ids": get_linkedin_users(),
                "days_back": args.days_back
            },
            "facebook": {
                "enabled": "facebook" in args.platforms,
                "page_ids": get_facebook_pages(),
                "ad_account_ids": get_facebook_ad_accounts(),
                "limit": 75
            },
            "design_platforms": {
                "enabled": True,  # Supplementary data
                "platforms": ["dribbble"]
            }
        },
        "training": {
            "platform_specific": True,
            "general_model": True,
            "cross_validation": True,
            "use_real_data": True
        }
    }


def get_instagram_accounts():
    """
    Get list of Instagram user accounts for data collection
    Configure these with actual Instagram user IDs you want to analyze
    """
    # Return empty list - users should configure their own accounts
    # Example: return ["your_instagram_user_id_1", "your_instagram_user_id_2"]
    return []


def get_instagram_business_accounts():
    """
    Get list of Instagram business accounts for data collection
    Use "me" for the authenticated user's account
    """
    return ["me"]  # Use authenticated user's account


def get_linkedin_companies():
    """
    Get list of LinkedIn company IDs for data collection
    Configure these with actual LinkedIn company IDs
    """
    # Return empty list - users should configure their own company IDs
    # Example: return ["12345", "67890"]
    return []


def get_linkedin_users():
    """
    Get list of LinkedIn user IDs for data collection
    """
    # Return empty list - users should configure their own user IDs
    return []


def get_facebook_pages():
    """
    Get list of Facebook page IDs for data collection
    Configure these with actual Facebook page IDs
    """
    # Return empty list - users should configure their own page IDs
    # Example: return ["your_page_id_1", "your_page_id_2"]
    return []


def get_facebook_ad_accounts():
    """
    Get list of Facebook ad account IDs for data collection
    """
    # Return empty list - users should configure their own ad account IDs
    return []


def print_results(results):
    """
    Print training results in a nice format
    """
    print(f"\n📊 Training Results")
    print("=" * 30)

    if results.get("status") == "success":
        print("✅ Pipeline Status: SUCCESS")
    else:
        print("❌ Pipeline Status: FAILED")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    # Data collection results
    if "data_collection" in results:
        data_stats = results["data_collection"]
        print(f"\n📈 Data Collection:")
        print(f"  Total samples: {data_stats.get('total_samples', 0)}")
        for source, count in data_stats.get("sources", {}).items():
            print(f"  {source}: {count} samples")

    # Model training results
    if "model_training" in results and "best_models" in results["model_training"]:
        print(f"\n🧠 Best Models:")
        for platform, model_info in results["model_training"]["best_models"].items():
            print(f"  {platform}: {model_info['model_type']} (R² = {model_info['r2_score']:.3f})")

    # Model evaluation
    if "model_evaluation" in results:
        eval_results = results["model_evaluation"]
        print(f"\n📊 Model Evaluation:")
        print(f"  Overall Quality: {eval_results.get('overall_quality', 'unknown').upper()}")

        for platform, metrics in eval_results.get("platform_performance", {}).items():
            print(f"  {platform}: R² = {metrics['r2_score']:.3f}, RMSE = {metrics['rmse']:.3f}")

    # Pipeline validation
    if "pipeline_validation" in results:
        validation = results["pipeline_validation"]
        print(f"\n✅ Pipeline Health: {validation.get('pipeline_health', 'unknown').upper()}")
        print(f"  Tests Passed: {len(validation.get('tests_passed', []))}")


def print_incremental_results(results):
    """
    Print incremental training results
    """
    print(f"\n🔄 Incremental Training Results")
    print("=" * 35)

    if results.get("status") == "no_new_data":
        print("ℹ️  No new data available for training")
        return

    if results.get("status") == "completed":
        print("✅ Incremental training completed")

        retrained = results.get("retrained_models", {})
        for platform, status in retrained.items():
            emoji = "✅" if status == "success" else "❌"
            print(f"  {emoji} {platform}: {status}")


def setup_api_keys():
    """
    Guide user through API key setup
    """
    print("\n🔑 API Key Setup Required")
    print("=" * 30)
    print("For full data collection, you need API keys for:")
    print("1. Instagram Basic Display API")
    print("2. LinkedIn Marketing API")
    print("3. Facebook Graph API")
    print("4. Dribbble API")
    print()
    print("Add these to your .env file:")
    print("INSTAGRAM_ACCESS_TOKEN=your_token")
    print("LINKEDIN_ACCESS_TOKEN=your_token")
    print("FACEBOOK_ACCESS_TOKEN=your_token")
    print("DRIBBBLE_ACCESS_TOKEN=your_token")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup-api":
        setup_api_keys()
        sys.exit(0)

    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)