"""
Real Facebook Data Collector
Collects actual Facebook post data using Facebook Graph API
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


class FacebookDataCollector:
    """
    Collects real Facebook data using official Graph API
    """

    def __init__(self):
        self.access_token = settings.FACEBOOK_ACCESS_TOKEN
        self.base_url = "https://graph.facebook.com/v18.0"

        if not self.access_token:
            logger.warning("Facebook access token not configured")

    async def collect_page_posts(self, page_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Collect posts from a Facebook page
        Uses Facebook Graph API
        """
        if not self.access_token:
            raise ValueError("Facebook access token not configured")

        collected_data = []

        async with aiohttp.ClientSession() as session:
            try:
                # Get page posts with media
                url = f"{self.base_url}/{page_id}/posts"
                params = {
                    "fields": "id,message,full_picture,picture,permalink_url,created_time,attachments,reactions.summary(true),comments.summary(true),shares",
                    "limit": limit,
                    "access_token": self.access_token
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get("data", [])

                        for post in posts:
                            # Only process posts with images
                            if post.get("full_picture") or post.get("picture"):
                                # Get detailed insights
                                insights = await self._get_post_insights(session, post["id"])

                                # Get image URL
                                image_url = post.get("full_picture") or post.get("picture")

                                # Download and save image
                                image_path = await self._download_media_file(session, image_url, post["id"])

                                # Extract engagement metrics
                                reactions = post.get("reactions", {}).get("summary", {}).get("total_count", 0)
                                comments = post.get("comments", {}).get("summary", {}).get("total_count", 0)
                                shares = post.get("shares", {}).get("count", 0)

                                post_data = {
                                    "id": post["id"],
                                    "platform": "facebook",
                                    "page_id": page_id,
                                    "image_path": str(image_path) if image_path else None,
                                    "message": post.get("message", ""),
                                    "permalink": post.get("permalink_url"),
                                    "created_time": post.get("created_time"),
                                    "engagement_metrics": {
                                        "reactions": reactions,
                                        "comments": comments,
                                        "shares": shares,
                                        "total_engagement": reactions + comments + shares,
                                        "engagement_rate": self._calculate_engagement_rate(reactions, comments, shares),
                                        **insights
                                    },
                                    "attachments": self._process_attachments(post.get("attachments", {})),
                                    "posting_time": self._analyze_posting_time(post.get("created_time")),
                                    "collection_date": datetime.now().isoformat()
                                }

                                collected_data.append(post_data)
                                logger.info(f"Collected Facebook post: {post['id']}")

                        # Handle pagination
                        next_page = data.get("paging", {}).get("next")
                        if next_page and len(collected_data) < 200:  # Limit total collection
                            await asyncio.sleep(1)  # Rate limiting
                            additional_data = await self._collect_paginated_data(session, next_page)
                            collected_data.extend(additional_data)

                    else:
                        logger.error(f"Facebook API error: {response.status} - {await response.text()}")

            except Exception as e:
                logger.error(f"Error collecting Facebook data: {e}")

        return collected_data

    async def collect_ad_creative_insights(self, ad_account_id: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Collect Facebook ad creative performance data
        Uses Facebook Marketing API
        """
        if not self.access_token:
            raise ValueError("Facebook access token not configured")

        collected_data = []
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        async with aiohttp.ClientSession() as session:
            try:
                # Get ad creatives with performance data
                url = f"{self.base_url}/act_{ad_account_id}/ads"
                params = {
                    "fields": "id,name,creative{id,title,body,image_url,object_story_spec},insights{impressions,clicks,ctr,cpm,cpp,reach,frequency,actions}",
                    "time_range": f'{{"since":"{since_date}","until":"{datetime.now().strftime("%Y-%m-%d")}"}}',
                    "limit": 50,
                    "access_token": self.access_token
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        ads = data.get("data", [])

                        for ad in ads:
                            creative = ad.get("creative", {})
                            insights = ad.get("insights", {}).get("data", [])

                            if creative.get("image_url") and insights:
                                # Download creative image
                                image_path = await self._download_media_file(
                                    session, creative["image_url"], f"ad_{ad['id']}"
                                )

                                # Process insights data
                                insight_data = insights[0] if insights else {}
                                performance_metrics = self._process_ad_insights(insight_data)

                                ad_data = {
                                    "id": f"ad_{ad['id']}",
                                    "platform": "facebook_ads",
                                    "ad_account_id": ad_account_id,
                                    "ad_id": ad["id"],
                                    "ad_name": ad.get("name"),
                                    "image_path": str(image_path) if image_path else None,
                                    "creative": {
                                        "title": creative.get("title"),
                                        "body": creative.get("body"),
                                        "call_to_action": self._extract_cta(creative)
                                    },
                                    "performance_metrics": performance_metrics,
                                    "campaign_objective": self._infer_objective(performance_metrics),
                                    "collection_date": datetime.now().isoformat()
                                }

                                collected_data.append(ad_data)
                                logger.info(f"Collected Facebook ad creative: {ad['id']}")

                    else:
                        logger.error(f"Facebook Marketing API error: {response.status} - {await response.text()}")

            except Exception as e:
                logger.error(f"Error collecting Facebook ad data: {e}")

        return collected_data

    async def collect_popular_hashtag_content(self, hashtags: List[str], limit_per_tag: int = 20) -> List[Dict[str, Any]]:
        """
        Collect popular content for specific hashtags using Facebook's search
        Note: Facebook has limited hashtag search capabilities compared to Instagram
        """
        if not self.access_token:
            raise ValueError("Facebook access token not configured")

        collected_data = []

        async with aiohttp.ClientSession() as session:
            for hashtag in hashtags:
                try:
                    # Search for public posts with hashtag
                    # Note: This requires specific permissions and may be limited
                    search_url = f"{self.base_url}/search"
                    search_params = {
                        "q": f"#{hashtag}",
                        "type": "post",
                        "fields": "id,message,full_picture,permalink_url,created_time,reactions.summary(true),comments.summary(true)",
                        "limit": limit_per_tag,
                        "access_token": self.access_token
                    }

                    async with session.get(search_url, params=search_params) as response:
                        if response.status == 200:
                            search_data = await response.json()
                            posts = search_data.get("data", [])

                            for post in posts:
                                if post.get("full_picture"):
                                    # Download image
                                    image_path = await self._download_media_file(
                                        session, post["full_picture"], f"hashtag_{hashtag}_{post['id']}"
                                    )

                                    # Extract engagement metrics
                                    reactions = post.get("reactions", {}).get("summary", {}).get("total_count", 0)
                                    comments = post.get("comments", {}).get("summary", {}).get("total_count", 0)

                                    post_data = {
                                        "id": f"hashtag_{hashtag}_{post['id']}",
                                        "platform": "facebook_hashtag",
                                        "hashtag": hashtag,
                                        "original_post_id": post["id"],
                                        "image_path": str(image_path) if image_path else None,
                                        "message": post.get("message", ""),
                                        "permalink": post.get("permalink_url"),
                                        "created_time": post.get("created_time"),
                                        "engagement_metrics": {
                                            "reactions": reactions,
                                            "comments": comments,
                                            "engagement_rate": self._calculate_engagement_rate(reactions, comments, 0)
                                        },
                                        "hashtags": self._extract_hashtags(post.get("message", "")),
                                        "collection_date": datetime.now().isoformat()
                                    }

                                    collected_data.append(post_data)

                    await asyncio.sleep(2)  # Rate limiting between hashtags

                except Exception as e:
                    logger.error(f"Error collecting hashtag data for #{hashtag}: {e}")

        return collected_data

    async def _get_post_insights(self, session: aiohttp.ClientSession, post_id: str) -> Dict[str, Any]:
        """
        Get insights for a specific Facebook post
        """
        try:
            url = f"{self.base_url}/{post_id}/insights"
            params = {
                "metric": "post_impressions,post_reach,post_engaged_users,post_video_views",
                "access_token": self.access_token
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    insights = {}

                    for insight in data.get("data", []):
                        metric_name = insight.get("name")
                        values = insight.get("values", [])

                        if values:
                            if metric_name == "post_impressions":
                                insights["impressions"] = values[0].get("value", 0)
                            elif metric_name == "post_reach":
                                insights["reach"] = values[0].get("value", 0)
                            elif metric_name == "post_engaged_users":
                                insights["engaged_users"] = values[0].get("value", 0)
                            elif metric_name == "post_video_views":
                                insights["video_views"] = values[0].get("value", 0)

                    return insights

        except Exception as e:
            logger.warning(f"Could not get insights for post {post_id}: {e}")

        return {}

    def _process_ad_insights(self, insight_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Facebook ad insights data into standardized format
        """
        metrics = {
            "impressions": insight_data.get("impressions", 0),
            "clicks": insight_data.get("clicks", 0),
            "reach": insight_data.get("reach", 0),
            "frequency": insight_data.get("frequency", 0),
            "ctr": insight_data.get("ctr", 0),
            "cpm": insight_data.get("cpm", 0),
            "cpp": insight_data.get("cpp", 0)
        }

        # Process actions (conversions, etc.)
        actions = insight_data.get("actions", [])
        for action in actions:
            action_type = action.get("action_type")
            value = action.get("value", 0)

            if action_type in ["like", "comment", "share", "link_click", "post_engagement"]:
                metrics[f"{action_type}_count"] = value

        return metrics

    def _process_attachments(self, attachments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process Facebook post attachments
        """
        processed_attachments = []

        for attachment_data in attachments.get("data", []):
            if attachment_data.get("type") in ["photo", "album"]:
                media = attachment_data.get("media", {})

                attachment = {
                    "type": attachment_data.get("type"),
                    "title": attachment_data.get("title"),
                    "description": attachment_data.get("description"),
                    "media_url": media.get("image", {}).get("src"),
                    "width": media.get("image", {}).get("width"),
                    "height": media.get("image", {}).get("height")
                }
                processed_attachments.append(attachment)

        return processed_attachments

    async def _download_media_file(self, session: aiohttp.ClientSession, media_url: str, media_id: str) -> Optional[Path]:
        """
        Download Facebook media file
        """
        try:
            async with session.get(media_url) as response:
                if response.status == 200:
                    content = await response.read()

                    # Create filename
                    file_hash = hashlib.md5(content).hexdigest()[:8]
                    filename = f"facebook_{media_id}_{file_hash}.jpg"

                    # Save to images directory
                    images_dir = Path("data/training/images")
                    images_dir.mkdir(parents=True, exist_ok=True)

                    file_path = images_dir / filename
                    with open(file_path, "wb") as f:
                        f.write(content)

                    logger.info(f"Downloaded Facebook image: {filename}")
                    return file_path

        except Exception as e:
            logger.error(f"Error downloading media {media_id}: {e}")

        return None

    async def _collect_paginated_data(self, session: aiohttp.ClientSession, next_url: str) -> List[Dict[str, Any]]:
        """
        Handle paginated Facebook API responses
        """
        collected_data = []

        try:
            async with session.get(next_url) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get("data", [])

                    for post in posts:
                        if post.get("full_picture") or post.get("picture"):
                            image_url = post.get("full_picture") or post.get("picture")
                            image_path = await self._download_media_file(session, image_url, post["id"])

                            if image_path:
                                post_data = {
                                    "id": post["id"],
                                    "platform": "facebook",
                                    "image_path": str(image_path),
                                    "message": post.get("message", ""),
                                    "created_time": post.get("created_time"),
                                    "engagement_metrics": self._extract_basic_engagement(post),
                                    "collection_date": datetime.now().isoformat()
                                }
                                collected_data.append(post_data)

        except Exception as e:
            logger.error(f"Error collecting paginated data: {e}")

        return collected_data

    def _extract_hashtags(self, message: str) -> List[str]:
        """
        Extract hashtags from Facebook post message
        """
        import re
        if not message:
            return []

        hashtags = re.findall(r'#(\w+)', message.lower())
        return hashtags

    def _extract_cta(self, creative: Dict[str, Any]) -> Optional[str]:
        """
        Extract call-to-action from Facebook ad creative
        """
        object_story = creative.get("object_story_spec", {})
        link_data = object_story.get("link_data", {})
        return link_data.get("call_to_action", {}).get("type")

    def _infer_objective(self, metrics: Dict[str, Any]) -> str:
        """
        Infer campaign objective from performance metrics
        """
        if metrics.get("link_click_count", 0) > 0:
            return "traffic"
        elif metrics.get("post_engagement_count", 0) > 0:
            return "engagement"
        elif metrics.get("reach", 0) > metrics.get("impressions", 1) * 0.8:
            return "reach"
        else:
            return "awareness"

    def _analyze_posting_time(self, created_time: str) -> Dict[str, Any]:
        """
        Analyze posting time for engagement insights
        """
        if not created_time:
            return {}

        try:
            # Facebook timestamp format: 2024-01-15T14:30:00+0000
            dt = datetime.fromisoformat(created_time.replace('+0000', '+00:00'))

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

    def _calculate_engagement_rate(self, reactions: int, comments: int, shares: int, followers: int = 1000) -> float:
        """
        Calculate engagement rate for Facebook content
        """
        if followers <= 0:
            return 0.0

        total_engagement = reactions + (comments * 2) + (shares * 3)  # Weight shares highest
        return (total_engagement / followers) * 100

    def _extract_basic_engagement(self, post: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract basic engagement metrics from post data
        """
        reactions = post.get("reactions", {}).get("summary", {}).get("total_count", 0)
        comments = post.get("comments", {}).get("summary", {}).get("total_count", 0)
        shares = post.get("shares", {}).get("count", 0)

        return {
            "reactions": reactions,
            "comments": comments,
            "shares": shares,
            "total_engagement": reactions + comments + shares
        }

    async def validate_access_token(self) -> bool:
        """
        Validate Facebook access token
        """
        if not self.access_token:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/me"
                params = {
                    "fields": "id,name",
                    "access_token": self.access_token
                }

                async with session.get(url, params=params) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error validating Facebook token: {e}")
            return False


# Example usage
async def collect_facebook_data_example():
    """
    Example of how to collect real Facebook data
    """
    collector = FacebookDataCollector()

    # Validate token first
    if not await collector.validate_access_token():
        logger.error("Invalid Facebook access token")
        return

    # Collect page posts
    page_data = await collector.collect_page_posts("your_page_id", limit=25)

    # Collect hashtag content
    hashtag_data = await collector.collect_popular_hashtag_content(
        ["design", "marketing", "branding"],
        limit_per_tag=10
    )

    # Save collected data
    all_data = page_data + hashtag_data

    if all_data:
        output_file = Path("data/training/metadata/facebook_real_data.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2, default=str)

        logger.info(f"Collected {len(all_data)} real Facebook posts")
        return all_data

    return []


if __name__ == "__main__":
    asyncio.run(collect_facebook_data_example())