#!/usr/bin/env python3
"""
CreativeIQ Production Data Collection Script
Collects real social media data using production APIs
"""

import asyncio
import argparse
import sys
from pathlib import Path
import logging

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.dataset_collector import collect_production_dataset
from app.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main data collection script for production
    """
    parser = argparse.ArgumentParser(description="Collect real social media data for CreativeIQ training")
    parser.add_argument("--instagram-users", nargs="+", default=[],
                       help="Instagram user IDs to collect data from")
    parser.add_argument("--instagram-business", nargs="+", default=["me"],
                       help="Instagram business account IDs (use 'me' for authenticated user)")
    parser.add_argument("--linkedin-companies", nargs="+", default=[],
                       help="LinkedIn company IDs to collect data from")
    parser.add_argument("--linkedin-users", nargs="+", default=[],
                       help="LinkedIn user IDs to collect data from")
    parser.add_argument("--facebook-pages", nargs="+", default=[],
                       help="Facebook page IDs to collect data from")
    parser.add_argument("--facebook-ads", nargs="+", default=[],
                       help="Facebook ad account IDs to collect data from")
    parser.add_argument("--include-design-platforms", action="store_true",
                       help="Include supplementary data from design platforms")
    parser.add_argument("--check-apis", action="store_true",
                       help="Only check API status without collecting data")

    args = parser.parse_args()

    print("🚀 CreativeIQ Production Data Collection")
    print("========================================\n")

    # Check API credentials first
    if args.check_apis:
        await check_api_status()
        return 0

    # Show configuration
    print("📋 Collection Configuration:")
    print(f"  Instagram Users: {args.instagram_users if args.instagram_users else 'None'}")
    print(f"  Instagram Business: {args.instagram_business}")
    print(f"  LinkedIn Companies: {args.linkedin_companies if args.linkedin_companies else 'None'}")
    print(f"  LinkedIn Users: {args.linkedin_users if args.linkedin_users else 'None'}")
    print(f"  Facebook Pages: {args.facebook_pages if args.facebook_pages else 'None'}")
    print(f"  Facebook Ad Accounts: {args.facebook_ads if args.facebook_ads else 'None'}")
    print(f"  Design Platforms: {'Yes' if args.include_design_platforms else 'No'}")
    print()

    try:
        # Run production data collection
        df, stats = await collect_production_dataset(
            instagram_users=args.instagram_users if args.instagram_users else None,
            instagram_business_accounts=args.instagram_business,
            linkedin_companies=args.linkedin_companies if args.linkedin_companies else None,
            linkedin_users=args.linkedin_users if args.linkedin_users else None,
            facebook_pages=args.facebook_pages if args.facebook_pages else None,
            facebook_ad_accounts=args.facebook_ads if args.facebook_ads else None,
            include_design_platforms=args.include_design_platforms
        )

        if df is not None and stats is not None:
            print("\n🎉 Data collection completed successfully!")
            print(f"📊 Dataset Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Platforms: {dict(stats['platforms'])}")
            print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            print(f"  Average engagement rate: {stats['avg_engagement_rate']:.2f}%")

            # Show next steps
            print(f"\n🎯 Next Steps:")
            print(f"  1. Review collected data in: data/training/metadata/")
            print(f"  2. Train models: python scripts/train_models.py --mode full")
            print(f"  3. Deploy to production: make deploy-prod")

            return 0
        else:
            print("\n❌ Data collection failed!")
            print("Check your API credentials and configuration.")
            return 1

    except Exception as e:
        print(f"\n💥 Error during data collection: {e}")
        print("\nTroubleshooting:")
        print("1. Verify API tokens in .env file")
        print("2. Check token permissions and expiration")
        print("3. Ensure network connectivity")
        print("4. Run with --check-apis to verify credentials")
        return 1


async def check_api_status():
    """
    Check status of all API credentials
    """
    print("🔑 Checking API Credentials...")
    print("=" * 30)

    # Check environment variables
    credentials = {
        "Instagram": settings.INSTAGRAM_ACCESS_TOKEN,
        "LinkedIn": settings.LINKEDIN_ACCESS_TOKEN,
        "Facebook": settings.FACEBOOK_ACCESS_TOKEN
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
        print("🔄 Testing API connections...")
        try:
            from app.ml.dataset_collector import DatasetCollector
            collector = DatasetCollector()
            api_validation = await collector.validate_all_tokens()

            for platform, valid in api_validation.items():
                emoji = "✅" if valid else "❌"
                status = "Valid" if valid else "Invalid/No permissions"
                print(f"  {emoji} {platform}: {status}")

        except Exception as e:
            print(f"❌ Error testing API connections: {e}")

    print()

    # Show configuration recommendations
    valid_apis = sum(1 for v in env_status.values() if v)
    if valid_apis == 0:
        print("⚠️  No API credentials configured!")
        print("Add the following to your .env file:")
        print("INSTAGRAM_ACCESS_TOKEN=your_instagram_token")
        print("LINKEDIN_ACCESS_TOKEN=your_linkedin_token")
        print("FACEBOOK_ACCESS_TOKEN=your_facebook_token")
    elif valid_apis < 3:
        print(f"⚠️  Only {valid_apis}/3 APIs configured.")
        print("For best results, configure all three platforms.")
    else:
        print("✅ All API credentials configured!")


def show_api_setup_guide():
    """
    Show detailed API setup guide
    """
    print("\n📖 API Setup Guide")
    print("=" * 20)
    print()
    print("🔑 To collect real social media data, you need API access tokens:")
    print()
    print("📸 Instagram (Basic Display API or Graph API):")
    print("  1. Go to: https://developers.facebook.com/")
    print("  2. Create a Facebook App")
    print("  3. Add Instagram Basic Display product")
    print("  4. Generate access token with user_profile, user_media permissions")
    print()
    print("💼 LinkedIn (Marketing API):")
    print("  1. Go to: https://developer.linkedin.com/")
    print("  2. Create a LinkedIn App")
    print("  3. Request Marketing API access")
    print("  4. Generate access token with r_liteprofile, r_ads permissions")
    print()
    print("📘 Facebook (Graph API):")
    print("  1. Go to: https://developers.facebook.com/tools/explorer/")
    print("  2. Select your app and page")
    print("  3. Generate token with pages_read_engagement permissions")
    print()
    print("⚠️  Important Security Notes:")
    print("  - Never commit API tokens to version control")
    print("  - Use environment variables (.env file)")
    print("  - Regularly rotate your tokens")
    print("  - Follow platform rate limiting guidelines")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup-guide":
        show_api_setup_guide()
        sys.exit(0)

    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Data collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)