"""
Dataset Collection Pipeline for CreativeIQ
Collects design images and their performance metrics from various sources
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import hashlib
import logging

from app.core.config import settings
from app.data_collection.instagram_collector import InstagramDataCollector
from app.data_collection.linkedin_collector import LinkedInDataCollector
from app.data_collection.facebook_collector import FacebookDataCollector

logger = logging.getLogger(__name__)


class DatasetCollector:
    """
    Collects design performance data from social media APIs and other sources
    """

    def __init__(self):
        self.data_dir = Path("data/training")
        self.images_dir = self.data_dir / "images"
        self.metadata_dir = self.data_dir / "metadata"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Initialize real API collectors
        self.instagram_collector = InstagramDataCollector()
        self.linkedin_collector = LinkedInDataCollector()
        self.facebook_collector = FacebookDataCollector()

        logger.info("Initialized real data collectors for production data collection")

    async def collect_instagram_data(self, user_ids: List[str] = None, limit: int = 50, business_accounts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Collect real Instagram post data using Instagram API collectors
        """
        collected_data = []

        try:
            # Validate Instagram access token
            if not await self.instagram_collector.validate_access_token():
                logger.warning("Instagram access token not valid, skipping Instagram data collection")
                return collected_data

            # Collect user media (personal accounts)
            if user_ids:
                for user_id in user_ids:
                    try:
                        user_data = await self.instagram_collector.collect_user_media(user_id, limit)
                        collected_data.extend(user_data)
                        logger.info(f"Collected {len(user_data)} posts from Instagram user {user_id}")
                    except Exception as e:
                        logger.error(f"Error collecting Instagram data for user {user_id}: {e}")

            # Collect business insights (business accounts)
            if business_accounts:
                for business_id in business_accounts:
                    try:
                        business_data = await self.instagram_collector.collect_business_insights(business_id, 30)
                        collected_data.extend(business_data)
                        logger.info(f"Collected {len(business_data)} business posts from Instagram account {business_id}")
                    except Exception as e:
                        logger.error(f"Error collecting Instagram business data for {business_id}: {e}")

            # Collect hashtag performance data
            try:
                hashtag_data = await self.instagram_collector.collect_hashtag_performance(
                    ["design", "graphicdesign", "branding", "marketing", "creative"], 20
                )
                collected_data.extend(hashtag_data)
                logger.info(f"Collected {len(hashtag_data)} hashtag performance posts")
            except Exception as e:
                logger.error(f"Error collecting Instagram hashtag data: {e}")

        except Exception as e:
            logger.error(f"Error in Instagram data collection: {e}")

        return collected_data

    async def collect_linkedin_data(self, company_ids: List[str] = None, user_ids: List[str] = None, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Collect real LinkedIn post data using LinkedIn API collector
        """
        collected_data = []

        try:
            # Validate LinkedIn access token
            if not await self.linkedin_collector.validate_access_token():
                logger.warning("LinkedIn access token not valid, skipping LinkedIn data collection")
                return collected_data

            # Collect company posts
            if company_ids:
                for company_id in company_ids:
                    try:
                        company_data = await self.linkedin_collector.collect_company_posts(company_id, days_back)
                        collected_data.extend(company_data)
                        logger.info(f"Collected {len(company_data)} posts from LinkedIn company {company_id}")
                    except Exception as e:
                        logger.error(f"Error collecting LinkedIn data for company {company_id}: {e}")

            # Collect user posts
            if user_ids:
                for user_id in user_ids:
                    try:
                        user_data = await self.linkedin_collector.collect_user_posts(user_id, days_back)
                        collected_data.extend(user_data)
                        logger.info(f"Collected {len(user_data)} posts from LinkedIn user {user_id}")
                    except Exception as e:
                        logger.error(f"Error collecting LinkedIn data for user {user_id}: {e}")

            # Collect ad creative performance
            try:
                ad_data = await self.linkedin_collector.collect_ad_performance(days_back)
                collected_data.extend(ad_data)
                logger.info(f"Collected {len(ad_data)} LinkedIn ad creatives")
            except Exception as e:
                logger.error(f"Error collecting LinkedIn ad data: {e}")

        except Exception as e:
            logger.error(f"Error in LinkedIn data collection: {e}")

        return collected_data

    async def collect_facebook_data(self, page_ids: List[str] = None, ad_account_ids: List[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Collect real Facebook post data using Facebook API collector
        """
        collected_data = []

        try:
            # Validate Facebook access token
            if not await self.facebook_collector.validate_access_token():
                logger.warning("Facebook access token not valid, skipping Facebook data collection")
                return collected_data

            # Collect page posts
            if page_ids:
                for page_id in page_ids:
                    try:
                        page_data = await self.facebook_collector.collect_page_posts(page_id, limit)
                        collected_data.extend(page_data)
                        logger.info(f"Collected {len(page_data)} posts from Facebook page {page_id}")
                    except Exception as e:
                        logger.error(f"Error collecting Facebook data for page {page_id}: {e}")

            # Collect ad creative insights
            if ad_account_ids:
                for ad_account_id in ad_account_ids:
                    try:
                        ad_data = await self.facebook_collector.collect_ad_creative_insights(ad_account_id, 30)
                        collected_data.extend(ad_data)
                        logger.info(f"Collected {len(ad_data)} ad creatives from Facebook account {ad_account_id}")
                    except Exception as e:
                        logger.error(f"Error collecting Facebook ad data for account {ad_account_id}: {e}")

            # Collect hashtag content
            try:
                hashtag_data = await self.facebook_collector.collect_popular_hashtag_content(
                    ["design", "marketing", "branding"], 15
                )
                collected_data.extend(hashtag_data)
                logger.info(f"Collected {len(hashtag_data)} Facebook hashtag posts")
            except Exception as e:
                logger.error(f"Error collecting Facebook hashtag data: {e}")

        except Exception as e:
            logger.error(f"Error in Facebook data collection: {e}")

        return collected_data

    async def collect_design_platform_data(self, platforms: List[str] = ["dribbble", "behance"]) -> List[Dict[str, Any]]:
        """
        Collect high-quality design samples from design platforms
        Note: These are supplementary to real social media data
        """
        collected_data = []

        for platform in platforms:
            if platform == "dribbble":
                data = await self._collect_dribbble_data()
            elif platform == "behance":
                data = await self._collect_behance_data()
            else:
                continue

            collected_data.extend(data)

        return collected_data

    async def _collect_dribbble_data(self) -> List[Dict[str, Any]]:
        """
        Collect popular designs from Dribbble API
        """
        collected_data = []

        async with aiohttp.ClientSession() as session:
            try:
                # Dribbble API v2 (requires authentication)
                url = "https://api.dribbble.com/v2/shots"
                params = {
                    "access_token": self.api_credentials.get("dribbble", {}).get("access_token"),
                    "sort": "popular",
                    "timeframe": "month",
                    "per_page": 100
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        shots = await response.json()

                        for shot in shots:
                            image_path = await self._download_image(
                                session, shot["images"]["hidpi"], shot["id"]
                            )

                            if image_path:
                                metadata = {
                                    "id": f"dribbble_{shot['id']}",
                                    "platform": "dribbble",
                                    "image_path": str(image_path),
                                    "title": shot.get("title", ""),
                                    "description": shot.get("description", ""),
                                    "timestamp": shot["created_at"],
                                    "engagement_metrics": {
                                        "views": shot.get("views_count", 0),
                                        "likes": shot.get("likes_count", 0),
                                        "comments": shot.get("comments_count", 0),
                                        "saves": shot.get("buckets_count", 0)
                                    },
                                    "tags": shot.get("tags", []),
                                    "color_palette": shot.get("colors", []),
                                    "category": "design_showcase",
                                    "quality_score": self._calculate_dribbble_quality_score(shot)
                                }

                                collected_data.append(metadata)

            except Exception as e:
                logger.error(f"Error collecting Dribbble data: {e}")

        return collected_data

    async def collect_a_b_test_data(self, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect A/B test results for design variants
        """
        collected_data = []

        for config in test_configs:
            try:
                # Simulate A/B test data collection
                # In practice, this would integrate with your A/B testing platform

                test_results = await self._get_ab_test_results(config)

                for variant in test_results["variants"]:
                    image_path = await self._download_image_from_url(variant["image_url"], variant["id"])

                    if image_path:
                        metadata = {
                            "id": f"ab_test_{variant['id']}",
                            "platform": config["platform"],
                            "image_path": str(image_path),
                            "test_id": config["test_id"],
                            "variant_name": variant["name"],
                            "timestamp": test_results["start_date"],
                            "engagement_metrics": {
                                "conversion_rate": variant["conversion_rate"],
                                "click_through_rate": variant["ctr"],
                                "engagement_rate": variant["engagement_rate"],
                                "impressions": variant["impressions"],
                                "clicks": variant["clicks"],
                                "conversions": variant["conversions"]
                            },
                            "test_duration": test_results["duration_days"],
                            "audience_segment": config.get("audience_segment", "general"),
                            "statistical_significance": variant["significance"],
                            "category": "ab_test_result"
                        }

                        collected_data.append(metadata)

            except Exception as e:
                logger.error(f"Error collecting A/B test data: {e}")

        return collected_data

    async def _download_image(self, session: aiohttp.ClientSession, url: str, post_id: str) -> Optional[Path]:
        """
        Download image from URL and save locally
        """
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()

                    # Create filename with hash to avoid duplicates
                    image_hash = hashlib.md5(image_data).hexdigest()
                    filename = f"{post_id}_{image_hash}.jpg"
                    image_path = self.images_dir / filename

                    # Save image
                    with open(image_path, "wb") as f:
                        f.write(image_data)

                    # Validate image
                    try:
                        with Image.open(image_path) as img:
                            if img.size[0] < 100 or img.size[1] < 100:  # Skip tiny images
                                image_path.unlink()
                                return None
                    except Exception:
                        image_path.unlink()
                        return None

                    return image_path

        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")

        return None

    def save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """
        Save collected data to JSON and CSV formats
        """
        # Save as JSON for full metadata
        json_path = self.metadata_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Save as CSV for easy analysis
        df = pd.json_normalize(data)
        csv_path = self.metadata_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Dataset saved: {len(data)} records to {json_path} and {csv_path}")

    def create_training_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Combine all collected data into training dataset
        """
        all_data = []

        # Load all JSON files
        for json_file in self.metadata_dir.glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)
                all_data.extend(data)

        if not all_data:
            raise ValueError("No training data found. Run data collection first.")

        # Convert to DataFrame
        df = pd.json_normalize(all_data)

        # Calculate derived metrics
        df = self._calculate_derived_metrics(df)

        # Dataset statistics
        stats = {
            "total_samples": len(df),
            "platforms": df["platform"].value_counts().to_dict(),
            "date_range": {
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max()
            },
            "avg_engagement_rate": df.get("derived_engagement_rate", pd.Series([0])).mean(),
            "quality_distribution": df.get("quality_score", pd.Series([0])).describe().to_dict()
        }

        return df, stats

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived engagement and performance metrics
        """
        # Engagement rate calculation (platform-specific)
        df["derived_engagement_rate"] = 0.0

        # Instagram engagement rate
        instagram_mask = df["platform"] == "instagram"
        if instagram_mask.any():
            df.loc[instagram_mask, "derived_engagement_rate"] = (
                (df.loc[instagram_mask, "engagement_metrics.likes"].fillna(0) +
                 df.loc[instagram_mask, "engagement_metrics.comments"].fillna(0)) /
                df.loc[instagram_mask, "audience_size"].fillna(1)
            ) * 100

        # LinkedIn engagement rate
        linkedin_mask = df["platform"] == "linkedin"
        if linkedin_mask.any():
            df.loc[linkedin_mask, "derived_engagement_rate"] = (
                df.loc[linkedin_mask, "engagement_metrics.engagement_rate"].fillna(0)
            )

        # Normalize engagement rates (0-100 scale)
        df["normalized_engagement"] = (
            df["derived_engagement_rate"] / df["derived_engagement_rate"].max() * 100
        ).fillna(0)

        # Performance score (composite metric)
        df["performance_score"] = (
            df["normalized_engagement"] * 0.4 +
            (df.get("engagement_metrics.likes", 0) / df.get("engagement_metrics.impressions", 1) * 100) * 0.3 +
            (df.get("engagement_metrics.shares", 0) / df.get("engagement_metrics.impressions", 1) * 100) * 0.3
        ).fillna(0)

        return df

    # Utility methods for API integration

    def get_api_status(self) -> Dict[str, bool]:
        """
        Check status of all API integrations
        """
        return {
            "instagram": bool(settings.INSTAGRAM_ACCESS_TOKEN),
            "linkedin": bool(settings.LINKEDIN_ACCESS_TOKEN),
            "facebook": bool(settings.FACEBOOK_ACCESS_TOKEN)
        }

    async def validate_all_tokens(self) -> Dict[str, bool]:
        """
        Validate all API tokens
        """
        validation_results = {}

        try:
            validation_results["instagram"] = await self.instagram_collector.validate_access_token()
        except Exception as e:
            logger.error(f"Instagram token validation failed: {e}")
            validation_results["instagram"] = False

        try:
            validation_results["linkedin"] = await self.linkedin_collector.validate_access_token()
        except Exception as e:
            logger.error(f"LinkedIn token validation failed: {e}")
            validation_results["linkedin"] = False

        try:
            validation_results["facebook"] = await self.facebook_collector.validate_access_token()
        except Exception as e:
            logger.error(f"Facebook token validation failed: {e}")
            validation_results["facebook"] = False

        return validation_results

    def _extract_hashtags(self, caption: str) -> List[str]:
        """Extract hashtags from caption"""
        import re
        return re.findall(r'#\w+', caption.lower())

    def _extract_posting_time(self, timestamp: str) -> Dict[str, Any]:
        """Extract posting time features"""
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return {
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "is_weekend": dt.weekday() >= 5,
            "time_of_day": "morning" if 6 <= dt.hour < 12 else "afternoon" if 12 <= dt.hour < 18 else "evening" if 18 <= dt.hour < 22 else "night"
        }

    def _calculate_dribbble_quality_score(self, shot: Dict[str, Any]) -> float:
        """Calculate quality score for Dribbble shots"""
        likes = shot.get("likes_count", 0)
        views = shot.get("views_count", 1)
        return min(100, (likes / views) * 1000)  # Simplified quality metric

    async def _get_ab_test_results(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock A/B test results"""
        return {
            "start_date": datetime.now().isoformat(),
            "duration_days": 7,
            "variants": [
                {
                    "id": f"variant_a_{config['test_id']}",
                    "name": "Control",
                    "image_url": "https://example.com/variant_a.jpg",
                    "conversion_rate": 0.035,
                    "ctr": 0.12,
                    "engagement_rate": 0.08,
                    "impressions": 10000,
                    "clicks": 1200,
                    "conversions": 350,
                    "significance": 0.95
                },
                {
                    "id": f"variant_b_{config['test_id']}",
                    "name": "Test",
                    "image_url": "https://example.com/variant_b.jpg",
                    "conversion_rate": 0.042,
                    "ctr": 0.15,
                    "engagement_rate": 0.11,
                    "impressions": 10000,
                    "clicks": 1500,
                    "conversions": 420,
                    "significance": 0.98
                }
            ]
        }

    async def _download_image_from_url(self, url: str, image_id: str) -> Optional[Path]:
        """Download image from URL (mock implementation)"""
        # In practice, this would download the actual image
        return self.images_dir / f"{image_id}.jpg"


# Example usage and data collection scripts

async def collect_production_dataset(
    instagram_users: List[str] = None,
    instagram_business_accounts: List[str] = None,
    linkedin_companies: List[str] = None,
    linkedin_users: List[str] = None,
    facebook_pages: List[str] = None,
    facebook_ad_accounts: List[str] = None,
    include_design_platforms: bool = False
):
    """
    Production script to collect real training dataset from social media APIs
    """
    collector = DatasetCollector()

    print("🚀 Starting PRODUCTION data collection with real APIs...")
    all_collected_data = []

    # Collect Instagram data
    print("📸 Collecting real Instagram data...")
    try:
        instagram_data = await collector.collect_instagram_data(
            user_ids=instagram_users,
            business_accounts=instagram_business_accounts,
            limit=100
        )
        if instagram_data:
            collector.save_dataset(instagram_data, "instagram_real_data")
            all_collected_data.extend(instagram_data)
            print(f"✅ Collected {len(instagram_data)} real Instagram posts")
        else:
            print("⚠️ No Instagram data collected - check API credentials")
    except Exception as e:
        logger.error(f"Instagram collection failed: {e}")

    # Collect LinkedIn data
    print("💼 Collecting real LinkedIn data...")
    try:
        linkedin_data = await collector.collect_linkedin_data(
            company_ids=linkedin_companies,
            user_ids=linkedin_users,
            days_back=30
        )
        if linkedin_data:
            collector.save_dataset(linkedin_data, "linkedin_real_data")
            all_collected_data.extend(linkedin_data)
            print(f"✅ Collected {len(linkedin_data)} real LinkedIn posts")
        else:
            print("⚠️ No LinkedIn data collected - check API credentials")
    except Exception as e:
        logger.error(f"LinkedIn collection failed: {e}")

    # Collect Facebook data
    print("📘 Collecting real Facebook data...")
    try:
        facebook_data = await collector.collect_facebook_data(
            page_ids=facebook_pages,
            ad_account_ids=facebook_ad_accounts,
            limit=75
        )
        if facebook_data:
            collector.save_dataset(facebook_data, "facebook_real_data")
            all_collected_data.extend(facebook_data)
            print(f"✅ Collected {len(facebook_data)} real Facebook posts")
        else:
            print("⚠️ No Facebook data collected - check API credentials")
    except Exception as e:
        logger.error(f"Facebook collection failed: {e}")

    # Optionally collect design platform data as supplementary
    if include_design_platforms:
        print("🎨 Collecting supplementary design platform data...")
        try:
            design_data = await collector.collect_design_platform_data(
                platforms=["dribbble"]
            )
            if design_data:
                collector.save_dataset(design_data, "design_platforms_supplementary")
                all_collected_data.extend(design_data)
                print(f"✅ Collected {len(design_data)} design platform samples")
        except Exception as e:
            logger.error(f"Design platform collection failed: {e}")

    if not all_collected_data:
        print("❌ No data collected! Please check your API credentials in .env file")
        print("Required environment variables:")
        print("- INSTAGRAM_ACCESS_TOKEN")
        print("- LINKEDIN_ACCESS_TOKEN")
        print("- FACEBOOK_ACCESS_TOKEN")
        return None, None

    # Create combined training dataset
    print("📊 Creating combined PRODUCTION training dataset...")
    try:
        df, stats = collector.create_training_dataset()

        print(f"🎉 PRODUCTION dataset collection complete!")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Platforms: {stats['platforms']}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"Average engagement rate: {stats['avg_engagement_rate']:.2f}%")

        # Save combined dataset for training
        combined_file = collector.metadata_dir / "production_training_dataset.json"
        with open(combined_file, "w") as f:
            json.dump(all_collected_data, f, indent=2, default=str)

        print(f"💾 Production dataset saved to: {combined_file}")
        return df, stats

    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        return None, None


async def collect_sample_dataset():
    """
    Quick collection with default parameters for testing
    """
    return await collect_production_dataset(
        instagram_business_accounts=["me"],  # Use authenticated user's account
        include_design_platforms=True
    )


if __name__ == "__main__":
    # Run production data collection
    print("🚀 Starting CreativeIQ Production Data Collection")
    print("This will collect real data from social media APIs")
    print("Make sure your API credentials are configured in .env file\n")

    # Example with real account configuration
    asyncio.run(collect_production_dataset(
        # Configure these with your actual account IDs
        instagram_business_accounts=["me"],  # Use "me" for authenticated user
        linkedin_companies=[],  # Add your LinkedIn company IDs
        facebook_pages=[],      # Add your Facebook page IDs
        include_design_platforms=True
    ))