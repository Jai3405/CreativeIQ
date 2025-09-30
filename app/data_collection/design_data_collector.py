"""
Design-Focused Data Collection System
Streamlined collector for design inspiration and creative content
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from app.data_collection.pinterest_collector import PinterestDesignCollector
from app.data_collection.instagram_collector import InstagramDataCollector

logger = logging.getLogger(__name__)


class DesignDataCollector:
    """
    Main collector for design-focused training data
    Aggregates data from Pinterest, Instagram design accounts, and design platforms
    """

    def __init__(self):
        self.pinterest_collector = PinterestDesignCollector()
        self.instagram_collector = InstagramDataCollector()

        self.data_dir = Path("data/training")
        self.images_dir = self.data_dir / "images"
        self.metadata_dir = self.data_dir / "metadata"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        logger.info("Initialized design-focused data collectors")

    async def collect_design_inspiration_data(self, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Collect comprehensive design inspiration data
        """
        if config is None:
            config = self._get_default_design_config()

        all_collected_data = []
        collection_summary = {"sources": {}, "total_samples": 0}

        logger.info("🎨 Starting design inspiration data collection...")

        # 1. Pinterest Design Inspiration
        if config.get("pinterest", {}).get("enabled", True):
            logger.info("📌 Collecting Pinterest design inspiration...")
            try:
                pinterest_data = await self._collect_pinterest_data(config["pinterest"])
                if pinterest_data:
                    all_collected_data.extend(pinterest_data)
                    collection_summary["sources"]["pinterest"] = len(pinterest_data)
                    logger.info(f"✅ Collected {len(pinterest_data)} Pinterest design pins")
                else:
                    logger.warning("⚠️ No Pinterest data collected")
            except Exception as e:
                logger.error(f"❌ Pinterest collection failed: {e}")

        # 2. Instagram Design Accounts & Hashtags
        if config.get("instagram", {}).get("enabled", True):
            logger.info("📸 Collecting Instagram design content...")
            try:
                instagram_data = await self._collect_instagram_design_data(config["instagram"])
                if instagram_data:
                    all_collected_data.extend(instagram_data)
                    collection_summary["sources"]["instagram"] = len(instagram_data)
                    logger.info(f"✅ Collected {len(instagram_data)} Instagram design posts")
                else:
                    logger.warning("⚠️ No Instagram design data collected")
            except Exception as e:
                logger.error(f"❌ Instagram collection failed: {e}")

        # 3. Design Platforms (Dribbble/Behance) - Simulated for now
        if config.get("design_platforms", {}).get("enabled", True):
            logger.info("🎯 Collecting design platform content...")
            try:
                platform_data = await self._collect_design_platform_data(config["design_platforms"])
                if platform_data:
                    all_collected_data.extend(platform_data)
                    collection_summary["sources"]["design_platforms"] = len(platform_data)
                    logger.info(f"✅ Collected {len(platform_data)} design platform samples")
            except Exception as e:
                logger.error(f"❌ Design platform collection failed: {e}")

        collection_summary["total_samples"] = len(all_collected_data)

        # Save collected data
        if all_collected_data:
            await self._save_design_dataset(all_collected_data)
            logger.info(f"🎉 Total design samples collected: {collection_summary['total_samples']}")
        else:
            logger.warning("⚠️ No design data collected from any source")

        return all_collected_data

    async def _collect_pinterest_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect Pinterest design inspiration data
        """
        pinterest_data = []

        try:
            # Design inspiration by categories
            if config.get("collect_inspiration", True):
                inspiration_data = await self.pinterest_collector.collect_design_inspiration(
                    design_categories=config.get("design_categories", None)
                )
                pinterest_data.extend(inspiration_data)

            # Design boards
            if config.get("collect_boards", True):
                board_data = await self.pinterest_collector.collect_design_boards(
                    board_keywords=config.get("board_keywords", None)
                )
                pinterest_data.extend(board_data)

            # Trending design pins
            if config.get("collect_trending", True):
                trending_data = await self.pinterest_collector.collect_trending_design_pins(
                    limit=config.get("trending_limit", 50)
                )
                pinterest_data.extend(trending_data)

        except Exception as e:
            logger.error(f"Error in Pinterest data collection: {e}")

        return pinterest_data

    async def _collect_instagram_design_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect Instagram design-focused content
        """
        instagram_data = []

        try:
            # Design hashtag performance
            if config.get("collect_hashtags", True):
                hashtag_data = await self.instagram_collector.collect_design_hashtag_performance(
                    design_hashtags=config.get("design_hashtags", None),
                    limit_per_tag=config.get("hashtag_limit", 20)
                )
                instagram_data.extend(hashtag_data)

            # Design business accounts (if available)
            if config.get("design_accounts", []):
                for account in config["design_accounts"]:
                    try:
                        account_data = await self.instagram_collector.collect_business_insights(
                            account, days_back=30
                        )
                        instagram_data.extend(account_data)
                    except Exception as e:
                        logger.warning(f"Could not collect from account {account}: {e}")

        except Exception as e:
            logger.error(f"Error in Instagram design data collection: {e}")

        return instagram_data

    async def _collect_design_platform_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect data from design platforms (Dribbble, Behance)
        Note: This is simulated data for now - would need actual APIs
        """
        platform_data = []

        # Simulated design platform data
        design_samples = [
            {
                "id": "dribbble_sample_1",
                "platform": "dribbble_simulation",
                "title": "Modern Logo Design Collection",
                "category": "logo_design",
                "engagement_metrics": {"views": 2500, "likes": 180, "saves": 45},
                "design_type": "branding",
                "color_scheme": "minimalist",
                "collection_date": datetime.now().isoformat()
            },
            {
                "id": "behance_sample_1",
                "platform": "behance_simulation",
                "title": "UI/UX Design System",
                "category": "ui_design",
                "engagement_metrics": {"views": 1800, "appreciations": 120, "saves": 30},
                "design_type": "interface",
                "style": "clean_modern",
                "collection_date": datetime.now().isoformat()
            }
        ]

        if config.get("include_simulation", True):
            platform_data.extend(design_samples)
            logger.info("Added simulated design platform data (replace with real APIs)")

        return platform_data

    async def _save_design_dataset(self, data: List[Dict[str, Any]]):
        """
        Save design dataset with proper categorization
        """
        # Save full dataset
        output_file = self.metadata_dir / "design_inspiration_dataset.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Save by platform
        platforms = {}
        for item in data:
            platform = item.get("platform", "unknown")
            if platform not in platforms:
                platforms[platform] = []
            platforms[platform].append(item)

        for platform, platform_data in platforms.items():
            platform_file = self.metadata_dir / f"design_{platform}_data.json"
            with open(platform_file, "w") as f:
                json.dump(platform_data, f, indent=2, default=str)

        logger.info(f"💾 Saved design dataset: {len(data)} samples across {len(platforms)} platforms")

    def _get_default_design_config(self) -> Dict[str, Any]:
        """
        Default configuration for design data collection
        """
        return {
            "pinterest": {
                "enabled": True,
                "collect_inspiration": True,
                "collect_boards": True,
                "collect_trending": True,
                "design_categories": [
                    "graphic design", "logo design", "brand identity",
                    "ui ux design", "typography design", "color palette",
                    "layout design", "poster design", "minimalist design"
                ],
                "board_keywords": [
                    "graphic design inspiration", "logo collection",
                    "brand identity design", "ui design inspiration"
                ],
                "trending_limit": 50
            },
            "instagram": {
                "enabled": True,
                "collect_hashtags": True,
                "design_hashtags": [
                    "graphicdesign", "logodesign", "branding", "typography",
                    "uidesign", "uxdesign", "designinspiration", "minimalistdesign"
                ],
                "hashtag_limit": 15,
                "design_accounts": []  # Add specific design account IDs here
            },
            "design_platforms": {
                "enabled": True,
                "include_simulation": True,
                "platforms": ["dribbble", "behance"]
            }
        }

    async def get_collection_status(self) -> Dict[str, Any]:
        """
        Get status of available data collection APIs
        """
        status = {}

        # Pinterest status
        try:
            status["pinterest"] = await self.pinterest_collector.validate_access_token()
        except Exception:
            status["pinterest"] = False

        # Instagram status
        try:
            status["instagram"] = await self.instagram_collector.validate_access_token()
        except Exception:
            status["instagram"] = False

        # Design platforms (would check real APIs)
        status["dribbble"] = False  # Would implement real check
        status["behance"] = False   # Would implement real check

        return status


# Main collection function for easy use
async def collect_design_training_data(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to collect design training data
    Returns dataset and statistics
    """
    collector = DesignDataCollector()

    # Collect data
    dataset = await collector.collect_design_inspiration_data(custom_config)

    # Generate statistics
    stats = {
        "total_samples": len(dataset),
        "platforms": {},
        "categories": {},
        "collection_date": datetime.now().isoformat()
    }

    # Platform breakdown
    for item in dataset:
        platform = item.get("platform", "unknown")
        stats["platforms"][platform] = stats["platforms"].get(platform, 0) + 1

    # Category breakdown
    for item in dataset:
        category = item.get("category", item.get("design_type", "unknown"))
        stats["categories"][category] = stats["categories"].get(category, 0) + 1

    logger.info(f"🎨 Design data collection complete: {stats['total_samples']} samples")

    return {
        "dataset": dataset,
        "statistics": stats,
        "status": "success" if dataset else "no_data"
    }


if __name__ == "__main__":
    # Example usage
    asyncio.run(collect_design_training_data())