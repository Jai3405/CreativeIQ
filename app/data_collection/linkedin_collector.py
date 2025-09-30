"""
Real LinkedIn Data Collector
Collects actual LinkedIn post data using LinkedIn Marketing API and Company Pages API
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


class LinkedInDataCollector:
    """
    Collects real LinkedIn data using official LinkedIn APIs
    """

    def __init__(self):
        self.access_token = settings.LINKEDIN_ACCESS_TOKEN
        self.base_url = "https://api.linkedin.com/v2"

        if not self.access_token:
            logger.warning("LinkedIn access token not configured")

    async def collect_company_posts(self, company_id: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Collect company posts using LinkedIn Company Pages API
        """
        if not self.access_token:
            raise ValueError("LinkedIn access token not configured")

        collected_data = []
        start_date = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            try:
                # Get company shares/posts
                url = f"{self.base_url}/shares"
                params = {
                    "q": "owners",
                    "owners": f"urn:li:organization:{company_id}",
                    "sharesDateRange.start": start_date,
                    "sortBy": "CREATED",
                    "count": 50
                }

                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }

                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        shares = data.get("elements", [])

                        for share in shares:
                            # Process each share/post
                            post_data = await self._process_linkedin_share(session, share, company_id)
                            if post_data:
                                collected_data.append(post_data)
                                logger.info(f"Collected LinkedIn post: {post_data['id']}")

                    else:
                        error_text = await response.text()
                        logger.error(f"LinkedIn API error: {response.status} - {error_text}")

            except Exception as e:
                logger.error(f"Error collecting LinkedIn company data: {e}")

        return collected_data

    async def collect_user_posts(self, user_id: str = "me", days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Collect user's own posts using LinkedIn API
        """
        if not self.access_token:
            raise ValueError("LinkedIn access token not configured")

        collected_data = []
        start_date = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            try:
                # Get user's posts
                url = f"{self.base_url}/people/{user_id}/posts"
                params = {
                    "q": "author",
                    "author": f"urn:li:person:{user_id}",
                    "sortBy": "CREATED",
                    "count": 50
                }

                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }

                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get("elements", [])

                        for post in posts:
                            post_data = await self._process_linkedin_post(session, post, user_id)
                            if post_data:
                                collected_data.append(post_data)
                                logger.info(f"Collected LinkedIn user post: {post_data['id']}")

                    else:
                        error_text = await response.text()
                        logger.error(f"LinkedIn posts API error: {response.status} - {error_text}")

            except Exception as e:
                logger.error(f"Error collecting LinkedIn user posts: {e}")

        return collected_data

    async def collect_analytics_data(self, company_id: str, content_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Collect detailed analytics for LinkedIn content
        """
        if not self.access_token:
            raise ValueError("LinkedIn access token not configured")

        analytics_data = {}

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "X-Restli-Protocol-Version": "2.0.0"
            }

            for content_id in content_ids:
                try:
                    # Get analytics for specific content
                    url = f"{self.base_url}/organizationalEntityShareStatistics"
                    params = {
                        "q": "organizationalEntity",
                        "organizationalEntity": f"urn:li:organization:{company_id}",
                        "shares": [f"urn:li:share:{content_id}"]
                    }

                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            analytics_data[content_id] = self._process_analytics_data(data)

                        await asyncio.sleep(1)  # Rate limiting

                except Exception as e:
                    logger.error(f"Error collecting analytics for {content_id}: {e}")

        return analytics_data

    async def collect_trending_content(self, industry: str = "technology", limit: int = 20) -> List[Dict[str, Any]]:
        """
        Collect trending content in specific industry
        Note: This requires special API access
        """
        if not self.access_token:
            raise ValueError("LinkedIn access token not configured")

        collected_data = []

        async with aiohttp.ClientSession() as session:
            try:
                # Note: This endpoint may require special permissions
                url = f"{self.base_url}/networkUpdates"
                params = {
                    "count": limit,
                    "start": 0
                }

                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }

                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = data.get("values", [])

                        for update in updates:
                            if self._is_relevant_update(update, industry):
                                processed_update = await self._process_network_update(session, update)
                                if processed_update:
                                    collected_data.append(processed_update)

                    else:
                        logger.warning(f"Trending content not accessible: {response.status}")

            except Exception as e:
                logger.error(f"Error collecting trending content: {e}")

        return collected_data

    async def _process_linkedin_share(self, session: aiohttp.ClientSession, share: Dict[str, Any], company_id: str) -> Optional[Dict[str, Any]]:
        """
        Process a LinkedIn share/post into our format
        """
        try:
            share_id = share.get("id")
            if not share_id:
                return None

            # Extract content
            content = share.get("text", {}).get("text", "")
            created_time = share.get("created", {}).get("time")

            # Get media content if available
            media_content = share.get("content", {})
            image_url = None
            image_path = None

            # Check for shared media
            if "media" in media_content:
                media = media_content["media"]
                if media.get("mediaType") == "IMAGE":
                    image_url = media.get("originalUrl")

            # Download image if available
            if image_url:
                image_path = await self._download_linkedin_image(session, image_url, share_id)

            # Get engagement metrics
            engagement_metrics = await self._get_share_analytics(session, share_id, company_id)

            # Extract company info
            owner_info = await self._get_company_info(session, company_id)

            post_data = {
                "id": share_id,
                "platform": "linkedin",
                "company_id": company_id,
                "company_info": owner_info,
                "image_path": str(image_path) if image_path else None,
                "content": content,
                "media_type": "image" if image_url else "text",
                "created_time": created_time,
                "timestamp": datetime.fromtimestamp(created_time / 1000).isoformat() if created_time else None,
                "engagement_metrics": engagement_metrics,
                "posting_time": self._analyze_posting_time(created_time),
                "content_analysis": self._analyze_content(content),
                "collection_date": datetime.now().isoformat()
            }

            return post_data

        except Exception as e:
            logger.error(f"Error processing LinkedIn share: {e}")
            return None

    async def _process_linkedin_post(self, session: aiohttp.ClientSession, post: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
        """
        Process a LinkedIn user post
        """
        try:
            post_id = post.get("id")
            if not post_id:
                return None

            # Extract post content
            content = post.get("commentary", "")
            created_time = post.get("createdAt")

            # Check for media
            image_url = None
            image_path = None

            if "content" in post:
                media_content = post["content"]
                if "media" in media_content:
                    media = media_content["media"]
                    if isinstance(media, list) and media:
                        # Get first media item
                        first_media = media[0]
                        if first_media.get("mediaType") == "IMAGE":
                            image_url = first_media.get("downloadUrl")

            # Download image
            if image_url:
                image_path = await self._download_linkedin_image(session, image_url, post_id)

            # Get engagement data
            engagement_metrics = await self._get_post_analytics(session, post_id)

            post_data = {
                "id": post_id,
                "platform": "linkedin_user",
                "user_id": user_id,
                "image_path": str(image_path) if image_path else None,
                "content": content,
                "media_type": "image" if image_url else "text",
                "created_time": created_time,
                "timestamp": datetime.fromtimestamp(created_time / 1000).isoformat() if created_time else None,
                "engagement_metrics": engagement_metrics,
                "posting_time": self._analyze_posting_time(created_time),
                "content_analysis": self._analyze_content(content),
                "collection_date": datetime.now().isoformat()
            }

            return post_data

        except Exception as e:
            logger.error(f"Error processing LinkedIn post: {e}")
            return None

    async def _get_share_analytics(self, session: aiohttp.ClientSession, share_id: str, company_id: str) -> Dict[str, Any]:
        """
        Get analytics data for a LinkedIn share
        """
        try:
            url = f"{self.base_url}/organizationalEntityShareStatistics"
            params = {
                "q": "organizationalEntity",
                "organizationalEntity": f"urn:li:organization:{company_id}",
                "shares": [f"urn:li:share:{share_id}"]
            }

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "X-Restli-Protocol-Version": "2.0.0"
            }

            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_analytics_data(data)

        except Exception as e:
            logger.warning(f"Could not get analytics for share {share_id}: {e}")

        return {"likes": 0, "comments": 0, "shares": 0, "clicks": 0, "impressions": 0, "engagement_rate": 0.0}

    async def _get_post_analytics(self, session: aiohttp.ClientSession, post_id: str) -> Dict[str, Any]:
        """
        Get analytics for user post
        """
        # Note: User post analytics may require different permissions
        return {"likes": 0, "comments": 0, "shares": 0, "clicks": 0, "impressions": 0, "engagement_rate": 0.0}

    def _process_analytics_data(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process LinkedIn analytics into standardized format
        """
        metrics = {"likes": 0, "comments": 0, "shares": 0, "clicks": 0, "impressions": 0, "engagement_rate": 0.0}

        try:
            elements = analytics_data.get("elements", [])
            if elements:
                stats = elements[0]

                # Extract metrics
                total_shares = stats.get("totalShareStatistics", {})
                metrics["likes"] = total_shares.get("likeCount", 0)
                metrics["comments"] = total_shares.get("commentCount", 0)
                metrics["shares"] = total_shares.get("shareCount", 0)
                metrics["clicks"] = total_shares.get("clickCount", 0)
                metrics["impressions"] = total_shares.get("impressionCount", 0)

                # Calculate engagement rate
                if metrics["impressions"] > 0:
                    total_engagement = metrics["likes"] + metrics["comments"] + metrics["shares"]
                    metrics["engagement_rate"] = (total_engagement / metrics["impressions"]) * 100

        except Exception as e:
            logger.warning(f"Error processing analytics data: {e}")

        return metrics

    async def _download_linkedin_image(self, session: aiohttp.ClientSession, image_url: str, post_id: str) -> Optional[Path]:
        """
        Download LinkedIn image
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }

            async with session.get(image_url, headers=headers) as response:
                if response.status == 200:
                    content = await response.read()

                    # Create filename
                    file_hash = hashlib.md5(content).hexdigest()[:8]
                    filename = f"linkedin_{post_id}_{file_hash}.jpg"

                    # Save to images directory
                    images_dir = Path("data/training/images")
                    images_dir.mkdir(parents=True, exist_ok=True)

                    file_path = images_dir / filename
                    with open(file_path, "wb") as f:
                        f.write(content)

                    logger.info(f"Downloaded LinkedIn image: {filename}")
                    return file_path

        except Exception as e:
            logger.error(f"Error downloading LinkedIn image: {e}")

        return None

    async def _get_company_info(self, session: aiohttp.ClientSession, company_id: str) -> Dict[str, Any]:
        """
        Get company information
        """
        try:
            url = f"{self.base_url}/organizations/{company_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "X-Restli-Protocol-Version": "2.0.0"
            }

            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "name": data.get("localizedName", ""),
                        "industry": data.get("industries", [{}])[0].get("localizedName", "") if data.get("industries") else "",
                        "size": data.get("staffCount", 0),
                        "follower_count": data.get("followersCount", 0)
                    }

        except Exception as e:
            logger.warning(f"Could not get company info: {e}")

        return {"name": "", "industry": "", "size": 0, "follower_count": 0}

    def _analyze_posting_time(self, created_time: Optional[int]) -> Dict[str, Any]:
        """
        Analyze posting time for business insights
        """
        if not created_time:
            return {}

        try:
            dt = datetime.fromtimestamp(created_time / 1000)

            return {
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "is_weekend": dt.weekday() >= 5,
                "is_business_hours": 9 <= dt.hour <= 17,
                "time_of_day": self._get_time_of_day(dt.hour),
                "month": dt.month,
                "quarter": (dt.month - 1) // 3 + 1
            }
        except Exception:
            return {}

    def _get_time_of_day(self, hour: int) -> str:
        """
        Categorize time of day for business context
        """
        if 6 <= hour < 9:
            return "early_morning"
        elif 9 <= hour < 12:
            return "morning_business"
        elif 12 <= hour < 14:
            return "lunch_time"
        elif 14 <= hour < 17:
            return "afternoon_business"
        elif 17 <= hour < 20:
            return "evening"
        else:
            return "off_hours"

    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Analyze LinkedIn content for insights
        """
        if not content:
            return {}

        # Basic content analysis
        analysis = {
            "word_count": len(content.split()),
            "character_count": len(content),
            "has_hashtags": "#" in content,
            "hashtag_count": content.count("#"),
            "has_mentions": "@" in content,
            "has_urls": "http" in content.lower(),
            "has_call_to_action": any(cta in content.lower() for cta in ["click", "learn more", "contact", "visit", "download"]),
            "sentiment": self._basic_sentiment_analysis(content),
            "content_type": self._classify_content_type(content)
        }

        return analysis

    def _basic_sentiment_analysis(self, content: str) -> str:
        """
        Basic sentiment analysis
        """
        positive_words = ["great", "excellent", "amazing", "successful", "excited", "proud", "happy"]
        negative_words = ["challenging", "difficult", "problem", "issue", "concern", "disappointing"]

        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _classify_content_type(self, content: str) -> str:
        """
        Classify LinkedIn content type
        """
        content_lower = content.lower()

        if any(word in content_lower for word in ["hiring", "job", "position", "career"]):
            return "job_posting"
        elif any(word in content_lower for word in ["congratulations", "welcome", "celebrate"]):
            return "company_news"
        elif any(word in content_lower for word in ["learn", "tips", "how to", "guide"]):
            return "educational"
        elif any(word in content_lower for word in ["event", "webinar", "conference", "meeting"]):
            return "event_promotion"
        else:
            return "general"

    def _is_relevant_update(self, update: Dict[str, Any], industry: str) -> bool:
        """
        Check if network update is relevant to specified industry
        """
        # Basic relevance check - this would be more sophisticated in production
        content = str(update).lower()
        industry_keywords = {
            "technology": ["tech", "software", "ai", "digital", "innovation"],
            "design": ["design", "creative", "branding", "ux", "ui"],
            "marketing": ["marketing", "brand", "campaign", "advertising"]
        }

        if industry in industry_keywords:
            return any(keyword in content for keyword in industry_keywords[industry])

        return True  # Include all if no specific industry

    async def _process_network_update(self, session: aiohttp.ClientSession, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process network update into our format
        """
        # This would process network updates similar to posts
        # Implementation depends on the specific update format
        return None

    async def validate_access_token(self) -> bool:
        """
        Validate LinkedIn access token
        """
        if not self.access_token:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/people/~"
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }

                async with session.get(url, headers=headers) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error validating LinkedIn token: {e}")
            return False


# Example usage
async def collect_linkedin_data_example():
    """
    Example of collecting real LinkedIn data
    """
    collector = LinkedInDataCollector()

    # Validate token
    if not await collector.validate_access_token():
        logger.error("Invalid LinkedIn access token")
        return

    # Collect company posts
    company_data = await collector.collect_company_posts("your_company_id", days_back=30)

    # Collect user posts
    user_data = await collector.collect_user_posts("me", days_back=30)

    # Combine data
    all_data = company_data + user_data

    if all_data:
        output_file = Path("data/training/metadata/linkedin_real_data.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2, default=str)

        logger.info(f"Collected {len(all_data)} real LinkedIn posts")
        return all_data

    return []


if __name__ == "__main__":
    asyncio.run(collect_linkedin_data_example())