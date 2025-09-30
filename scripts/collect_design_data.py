#!/usr/bin/env python3
"""
CreativeIQ Design Data Collection Script
Collects design inspiration data from Pinterest, Instagram design accounts, and design platforms
"""

import asyncio
import argparse
import sys
from pathlib import Path
import logging

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.data_collection.design_data_collector import collect_design_training_data
from app.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main design data collection script
    """
    parser = argparse.ArgumentParser(description="Collect design inspiration data for CreativeIQ training")
    parser.add_argument("--platforms", nargs="+", default=["instagram", "pinterest"],
                       help="Platforms to collect from: instagram, pinterest, dribbble, behance")
    parser.add_argument("--design-categories", nargs="+",
                       default=["graphic design", "logo design", "ui ux design", "typography"],
                       help="Design categories to focus on")
    parser.add_argument("--limit", type=int, default=100,
                       help="Maximum number of samples to collect per platform")
    parser.add_argument("--check-apis", action="store_true",
                       help="Only check API status without collecting data")

    args = parser.parse_args()

    print("🎨 CreativeIQ Design Data Collection")
    print("=====================================\n")

    # Check API credentials first
    if args.check_apis:
        await check_design_api_status()
        return 0

    # Show configuration
    print("📋 Collection Configuration:")
    print(f"  Platforms: {args.platforms}")
    print(f"  Design Categories: {args.design_categories}")
    print(f"  Sample Limit: {args.limit}")
    print()

    # Create custom configuration
    config = create_design_config(args)

    try:
        # Run design data collection
        result = await collect_design_training_data(config)

        if result["status"] == "success":
            stats = result["statistics"]
            print("🎉 Design data collection completed successfully!")
            print(f"📊 Collection Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Platforms: {dict(stats['platforms'])}")
            print(f"  Categories: {dict(stats['categories'])}")

            # Show next steps
            print(f"\n🎯 Next Steps:")
            print(f"  1. Review collected data in: data/training/metadata/")
            print(f"  2. Train models: python scripts/train_models.py --mode full")
            print(f"  3. Test with design analysis: python -m app.main")

            return 0
        else:
            print("\n❌ Design data collection failed!")
            print("Check your API credentials and configuration.")
            return 1

    except Exception as e:
        print(f"\n💥 Error during design data collection: {e}")
        print("\nTroubleshooting:")
        print("1. Verify API tokens in .env file")
        print("2. Check network connectivity")
        print("3. Run with --check-apis to verify credentials")
        return 1


async def check_design_api_status():
    """
    Check status of design-related API credentials
    """
    print("🔑 Checking Design API Credentials...")
    print("=" * 40)

    # Check environment variables
    credentials = {
        "Instagram (Design Hashtags)": settings.INSTAGRAM_ACCESS_TOKEN,
        "Pinterest (Design Inspiration)": settings.PINTEREST_ACCESS_TOKEN,
        "Dribbble (Design Portfolios)": settings.DRIBBBLE_ACCESS_TOKEN,
        "Behance (Creative Projects)": settings.BEHANCE_API_KEY
    }

    env_status = {}
    for platform, token in credentials.items():
        if token:
            # Mask token for security
            masked_token = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else "***"
            print(f"✅ {platform}: Token found ({masked_token})")
            env_status[platform] = True
        else:
            print(f"❌ {platform}: No token found")
            env_status[platform] = False

    print()

    # Test API connections
    if any(env_status.values()):
        print("🔄 Testing design API connections...")
        try:
            from app.data_collection.design_data_collector import DesignDataCollector
            collector = DesignDataCollector()
            api_validation = await collector.get_collection_status()

            platform_names = {
                "instagram": "Instagram (Design Hashtags)",
                "pinterest": "Pinterest (Design Inspiration)",
                "dribbble": "Dribbble (Design Portfolios)",
                "behance": "Behance (Creative Projects)"
            }

            for platform, valid in api_validation.items():
                platform_name = platform_names.get(platform, platform)
                emoji = "✅" if valid else "❌"
                status = "Valid" if valid else "Invalid/No permissions"
                print(f"  {emoji} {platform_name}: {status}")

        except Exception as e:
            print(f"❌ Error testing API connections: {e}")

    print()

    # Show configuration recommendations
    valid_apis = sum(1 for v in env_status.values() if v)
    if valid_apis == 0:
        print("⚠️  No design API credentials configured!")
        print("Add the following to your .env file:")
        print("INSTAGRAM_ACCESS_TOKEN=your_instagram_token")
        print("PINTEREST_ACCESS_TOKEN=your_pinterest_token")
        print("DRIBBBLE_ACCESS_TOKEN=your_dribbble_token")
        print("BEHANCE_API_KEY=your_behance_key")
    elif valid_apis < 2:
        print(f"⚠️  Only {valid_apis}/4 design APIs configured.")
        print("For best design data collection, configure Instagram + Pinterest at minimum.")
    else:
        print("✅ Good design API coverage!")

    # Show design data benefits
    print("\n🎨 Design Data Collection Benefits:")
    print("  • Instagram: Real engagement on design hashtags")
    print("  • Pinterest: Trending design inspiration and boards")
    print("  • Dribbble: Professional design portfolios")
    print("  • Behance: Creative project showcases")


def create_design_config(args) -> dict:
    """
    Create design-focused collection configuration
    """
    return {
        "pinterest": {
            "enabled": "pinterest" in args.platforms,
            "collect_inspiration": True,
            "collect_boards": True,
            "collect_trending": True,
            "design_categories": args.design_categories,
            "trending_limit": min(args.limit, 50)
        },
        "instagram": {
            "enabled": "instagram" in args.platforms,
            "collect_hashtags": True,
            "design_hashtags": [
                "graphicdesign", "logodesign", "branding", "typography",
                "uidesign", "uxdesign", "designinspiration", "minimalistdesign",
                "colorpalette", "layoutdesign", "posterdesign", "brandidentity"
            ],
            "hashtag_limit": min(args.limit // 10, 20),
            "design_accounts": []  # Add specific design account IDs if available
        },
        "design_platforms": {
            "enabled": any(p in args.platforms for p in ["dribbble", "behance"]),
            "include_simulation": True,  # Until real APIs are implemented
            "platforms": [p for p in args.platforms if p in ["dribbble", "behance"]]
        }
    }


def show_design_api_setup_guide():
    """
    Show detailed setup guide for design APIs
    """
    print("\n📖 Design API Setup Guide")
    print("=" * 30)
    print()
    print("🎨 To collect real design data, you need API access tokens:")
    print()
    print("📸 Instagram (for design hashtags and accounts):")
    print("  1. Go to: https://developers.facebook.com/")
    print("  2. Create a Facebook App with Instagram API")
    print("  3. Generate access token with permissions")
    print("  4. Focus on design hashtags: #graphicdesign, #logodesign, etc.")
    print()
    print("📌 Pinterest (for design inspiration):")
    print("  1. Go to: https://developers.pinterest.com/")
    print("  2. Create a Pinterest App")
    print("  3. Generate access token")
    print("  4. Access design boards and trending pins")
    print()
    print("🎯 Dribbble (for professional design portfolios):")
    print("  1. Go to: https://dribbble.com/account/applications")
    print("  2. Create a Dribbble App")
    print("  3. Get API access token")
    print("  4. Access popular shots and design trends")
    print()
    print("🎨 Behance (for creative project showcases):")
    print("  1. Go to: https://www.behance.net/dev")
    print("  2. Register for API key")
    print("  3. Access creative portfolios and featured work")
    print()
    print("🔑 Add tokens to your .env file:")
    print("INSTAGRAM_ACCESS_TOKEN=your_instagram_token")
    print("PINTEREST_ACCESS_TOKEN=your_pinterest_token")
    print("DRIBBBLE_ACCESS_TOKEN=your_dribbble_token")
    print("BEHANCE_API_KEY=your_behance_key")
    print()
    print("🎯 Focus Areas for Design Data:")
    print("  • Logo design and branding")
    print("  • UI/UX design patterns")
    print("  • Typography and color schemes")
    print("  • Layout and composition")
    print("  • Design trends and inspiration")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup-guide":
        show_design_api_setup_guide()
        sys.exit(0)

    print("🎨 CreativeIQ Design Data Collection")
    print("=====================================")
    print("Collecting design inspiration for AI training")
    print("Focus: Graphics, Logos, UI/UX, Typography\n")

    try:
        exit_code = asyncio.run(main())
        if exit_code == 0:
            print("\n🎉 Design data collection completed successfully!")
            print("Your CreativeIQ model now has real design training data.")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        print("\nRun with --setup-guide for API configuration help")
        sys.exit(1)