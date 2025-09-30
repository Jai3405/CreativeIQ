"""
Real Instagram Data Collector
Collects actual Instagram post data using Instagram Basic Display API and Graph API
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import hashlib

from app.core.config import settings

logger = logging.getLogger(__name__)


class InstagramDataCollector:
    """
    Collects real Instagram data using official APIs
    """

    def __init__(self):
        self.access_token = settings.INSTAGRAM_ACCESS_TOKEN
        self.base_url = "https://graph.instagram.com"
        self.basic_display_url = "https://graph.instagram.com"

        if not self.access_token:
            logger.warning("Instagram access token not configured")

    async def collect_design_account_media(self, account_username: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Collect media posts from design-focused Instagram accounts
        Focuses on design, art, and creative content
        """
        if not self.access_token:
            raise ValueError("Instagram access token not configured")

        collected_data = []

        async with aiohttp.ClientSession() as session:
            try:
                # Get user's media
                url = f"{self.basic_display_url}/me/media"
                params = {
                    "fields": "id,caption,media_type,media_url,permalink,timestamp,username",
                    "limit": limit,
                    "access_token": self.access_token
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        media_items = data.get("data", [])

                        for item in media_items:
                            if item.get("media_type") in ["IMAGE", "CAROUSEL_ALBUM"]:
                                # Get detailed insights for each post
                                insights = await self._get_media_insights(session, item["id"])

                                # Download and save image
                                image_path = await self._download_media_file(session, item["media_url"], item["id"])

                                post_data = {
                                    "id": item["id"],
                                    "platform": "instagram",
                                    "user_id": "me",  # Authenticated user
                                    "username": item.get("username"),
                                    "image_path": str(image_path) if image_path else None,
                                    "caption": item.get("caption", ""),
                                    "media_type": item.get("media_type"),
                                    "permalink": item.get("permalink"),
                                    "timestamp": item.get("timestamp"),
                                    "engagement_metrics": insights,
                                    "collection_date": datetime.now().isoformat()
                                }

                                collected_data.append(post_data)
                                logger.info(f"Collected Instagram post: {item['id']}")

                    else:
                        logger.error(f"Instagram API error: {response.status} - {await response.text()}")

            except Exception as e:
                logger.error(f"Error collecting Instagram data: {e}")

        return collected_data

    async def collect_business_insights(self, business_account_id: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Collect business insights using Instagram Graph API
        Requires Instagram Business Account
        """
        if not self.access_token:
            raise ValueError("Instagram access token not configured")

        collected_data = []
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        async with aiohttp.ClientSession() as session:
            try:
                # Get business account media with insights
                url = f"{self.base_url}/{business_account_id}/media"
                params = {
                    "fields": "id,caption,media_type,media_url,permalink,timestamp,insights.metric(engagement,impressions,reach,saved)",
                    "since": since_date,
                    "access_token": self.access_token
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        media_items = data.get("data", [])

                        for item in media_items:
                            if item.get("media_type") in ["IMAGE", "CAROUSEL_ALBUM"]:
                                # Process insights
                                insights_data = item.get("insights", {}).get("data", [])
                                engagement_metrics = self._process_insights(insights_data)

                                # Download image
                                image_path = await self._download_media_file(session, item["media_url"], item["id"])

                                # Get hashtags from caption
                                hashtags = self._extract_hashtags(item.get("caption", ""))

                                post_data = {
                                    "id": item["id"],
                                    "platform": "instagram_business",
                                    "business_account_id": business_account_id,
                                    "image_path": str(image_path) if image_path else None,
                                    "caption": item.get("caption", ""),
                                    "media_type": item.get("media_type"),
                                    "permalink": item.get("permalink"),
                                    "timestamp": item.get("timestamp"),
                                    "engagement_metrics": engagement_metrics,
                                    "hashtags": hashtags,
                                    "posting_time": self._analyze_posting_time(item.get("timestamp")),
                                    "collection_date": datetime.now().isoformat()
                                }

                                collected_data.append(post_data)
                                logger.info(f"Collected Instagram business post: {item['id']}")

                        # Handle pagination
                        next_page = data.get("paging", {}).get("next")
                        if next_page and len(collected_data) < 200:  # Limit total collection
                            await asyncio.sleep(1)  # Rate limiting
                            additional_data = await self._collect_paginated_data(session, next_page)
                            collected_data.extend(additional_data)

                    else:
                        logger.error(f"Instagram Business API error: {response.status} - {await response.text()}")

            except Exception as e:
                logger.error(f"Error collecting Instagram business data: {e}")

        return collected_data

    async def collect_design_hashtag_performance(self, design_hashtags: List[str] = None, limit_per_tag: int = 30) -> List[Dict[str, Any]]:
        """
        Collect top performing posts for design-specific hashtags
        Uses Instagram Graph API hashtag search
        """
        if design_hashtags is None:
            design_hashtags = [
                "graphicdesign", "logodesign", "branding", "typography",
                "uidesign", "uxdesign", "designinspiration", "minimalistdesign",
                "creativedesign", "brandidentity", "colorpalette", "layoutdesign",
                "posterdesign", "digitalart", "illustration", "designtrends"
            ]
        if not self.access_token:
            raise ValueError("Instagram access token not configured")

        collected_data = []

        async with aiohttp.ClientSession() as session:
            for hashtag in design_hashtags:
                try:
                    # Search for hashtag
                    search_url = f"{self.base_url}/ig_hashtag_search"
                    search_params = {
                        "user_id": "17841400027244616",  # Instagram user ID (can be extracted from token)
                        "q": hashtag,
                        "access_token": self.access_token
                    }

                    async with session.get(search_url, params=search_params) as response:
                        if response.status == 200:
                            search_data = await response.json()
                            hashtag_data = search_data.get("data", [])

                            if hashtag_data:
                                hashtag_id = hashtag_data[0]["id"]

                                # Get top media for hashtag
                                media_url = f"{self.base_url}/{hashtag_id}/top_media"
                                media_params = {
                                    "user_id": "17841400027244616",
                                    "fields": "id,media_type,media_url,permalink,timestamp,like_count,comments_count",
                                    "limit": limit_per_tag,
                                    "access_token": self.access_token
                                }

                                async with session.get(media_url, params=media_params) as media_response:
                                    if media_response.status == 200:
                                        media_data = await media_response.json()

                                        for item in media_data.get("data", []):
                                            if item.get("media_type") == "IMAGE":
                                                image_path = await self._download_media_file(
                                                    session, item["media_url"], f"hashtag_{hashtag}_{item['id']}"
                                                )

                                                post_data = {
                                                    "id": f"hashtag_{hashtag}_{item['id']}",
                                                    "platform": "instagram_hashtag",
                                                    "hashtag": hashtag,
                                                    "original_post_id": item["id"],
                                                    "image_path": str(image_path) if image_path else None,
                                                    "media_type": item.get("media_type"),
                                                    "permalink": item.get("permalink"),
                                                    "timestamp": item.get("timestamp"),
                                                    "engagement_metrics": {
                                                        "likes": item.get("like_count", 0),
                                                        "comments": item.get("comments_count", 0),
                                                        "engagement_rate": self._calculate_basic_engagement_rate(
                                                            item.get("like_count", 0),
                                                            item.get("comments_count", 0)
                                                        )
                                                    },
                                                    "collection_date": datetime.now().isoformat()
                                                }

                                                collected_data.append(post_data)

                    await asyncio.sleep(2)  # Rate limiting between hashtags

                except Exception as e:
                    logger.error(f"Error collecting hashtag data for #{hashtag}: {e}")

        return collected_data

    async def _get_media_insights(self, session: aiohttp.ClientSession, media_id: str) -> Dict[str, Any]:
        """
        Get insights for a specific media item
        """
        try:
            url = f"{self.basic_display_url}/{media_id}"
            params = {
                "fields": "like_count,comments_count",
                "access_token": self.access_token
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "likes": data.get("like_count", 0),
                        "comments": data.get("comments_count", 0),
                        "engagement_rate": self._calculate_basic_engagement_rate(
                            data.get("like_count", 0),
                            data.get("comments_count", 0)
                        )
                    }
        except Exception as e:
            logger.warning(f"Could not get insights for media {media_id}: {e}")

        return {"likes": 0, "comments": 0, "engagement_rate": 0.0}

    def _process_insights(self, insights_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process Instagram insights data into standardized format
        """
        metrics = {}

        for insight in insights_data:
            name = insight.get("name")
            values = insight.get("values", [])

            if values:
                if name == "engagement":
                    metrics["engagement"] = values[0].get("value", 0)
                elif name == "impressions":
                    metrics["impressions"] = values[0].get("value", 0)
                elif name == "reach":
                    metrics["reach"] = values[0].get("value", 0)
                elif name == "saved":
                    metrics["saves"] = values[0].get("value", 0)

        # Calculate engagement rate
        if metrics.get("impressions", 0) > 0:
            metrics["engagement_rate"] = (metrics.get("engagement", 0) / metrics["impressions"]) * 100
        else:
            metrics["engagement_rate"] = 0.0

        return metrics

    async def _download_media_file(self, session: aiohttp.ClientSession, media_url: str, media_id: str) -> Optional[Path]:
        """
        Download Instagram media file
        """
        try:
            async with session.get(media_url) as response:
                if response.status == 200:
                    content = await response.read()

                    # Create filename
                    file_hash = hashlib.md5(content).hexdigest()[:8]
                    filename = f"instagram_{media_id}_{file_hash}.jpg"

                    # Save to images directory
                    images_dir = Path("data/training/images")
                    images_dir.mkdir(parents=True, exist_ok=True)

                    file_path = images_dir / filename
                    with open(file_path, "wb") as f:
                        f.write(content)

                    logger.info(f"Downloaded Instagram image: {filename}")
                    return file_path

        except Exception as e:
            logger.error(f"Error downloading media {media_id}: {e}")

        return None

    async def _collect_paginated_data(self, session: aiohttp.ClientSession, next_url: str) -> List[Dict[str, Any]]:
        """
        Handle paginated Instagram API responses
        """
        collected_data = []

        try:
            async with session.get(next_url) as response:
                if response.status == 200:
                    data = await response.json()
                    media_items = data.get("data", [])

                    for item in media_items:
                        if item.get("media_type") in ["IMAGE", "CAROUSEL_ALBUM"]:
                            # Process similar to main collection
                            image_path = await self._download_media_file(session, item["media_url"], item["id"])

                            if image_path:
                                post_data = {
                                    "id": item["id"],
                                    "platform": "instagram_business",
                                    "image_path": str(image_path),
                                    "caption": item.get("caption", ""),
                                    "timestamp": item.get("timestamp"),
                                    "engagement_metrics": self._process_insights(
                                        item.get("insights", {}).get("data", [])
                                    ),
                                    "collection_date": datetime.now().isoformat()
                                }
                                collected_data.append(post_data)

        except Exception as e:
            logger.error(f"Error collecting paginated data: {e}")

        return collected_data

    def _extract_hashtags(self, caption: str) -> List[str]:
        """
        Extract hashtags from Instagram caption
        """
        import re
        if not caption:
            return []

        hashtags = re.findall(r'#(\w+)', caption.lower())
        return hashtags

    def _analyze_posting_time(self, timestamp: str) -> Dict[str, Any]:
        """
        Analyze posting time for engagement insights
        """
        if not timestamp:
            return {}

        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            return {
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "is_weekend": dt.weekday() >= 5,
                "time_of_day": self._get_time_of_day(dt.hour),
                "month": dt.month,
                "is_business_hours": 9 <= dt.hour <= 17
            }
        except Exception:
            return {}

    def _get_time_of_day(self, hour: int) -> str:
        """
        Categorize time of day
        """
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _calculate_basic_engagement_rate(self, likes: int, comments: int, followers: int = 1000) -> float:
        """
        Calculate basic engagement rate
        """
        if followers <= 0:
            return 0.0

        total_engagement = likes + (comments * 2)  # Weight comments higher
        return (total_engagement / followers) * 100

    async def validate_access_token(self) -> bool:
        """
        Validate Instagram access token
        """
        if not self.access_token:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.basic_display_url}/me"
                params = {
                    "fields": "id,username",
                    "access_token": self.access_token
                }

                async with session.get(url, params=params) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error validating Instagram token: {e}")
            return False


# Example usage
async def collect_instagram_data_example():
    """
    Example of how to collect real Instagram data
    """
    collector = InstagramDataCollector()

    # Validate token first
    if not await collector.validate_access_token():
        logger.error("Invalid Instagram access token")
        return

    # Collect user media
    user_data = await collector.collect_user_media("me", limit=25)

    # Collect hashtag performance
    hashtag_data = await collector.collect_hashtag_performance(
        ["design", "graphicdesign", "branding"],
        limit_per_tag=10
    )

    # Save collected data
    all_data = user_data + hashtag_data

    if all_data:
        output_file = Path("data/training/metadata/instagram_real_data.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2, default=str)

        logger.info(f"Collected {len(all_data)} real Instagram posts")
        return all_data

    return []


if __name__ == "__main__":
    asyncio.run(collect_instagram_data_example())