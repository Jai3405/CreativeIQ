"""
Reference-Based Design Data Collector
Collects metadata and URLs instead of downloading images
Fetches images on-demand during analysis
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import hashlib
import time

from app.core.config import settings
from app.data_collection.instagram_collector import InstagramDataCollector
from app.data_collection.pinterest_collector import PinterestDesignCollector

logger = logging.getLogger(__name__)


class ReferenceBasedDesignCollector:
    """
    Collects design data as references (URLs + metadata) instead of downloads
    Images are fetched on-demand during analysis
    """

    def __init__(self):
        self.data_dir = Path("data/training")
        self.references_dir = self.data_dir / "references"
        self.cache_dir = self.data_dir / "image_cache"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache settings
        self.cache_duration = timedelta(hours=24)  # Cache images for 24 hours
        self.max_cache_size = 100  # Maximum cached images

        # Initialize real API collectors
        self.instagram_collector = InstagramDataCollector()
        self.pinterest_collector = PinterestDesignCollector()

    async def collect_instagram_design_references(self, design_hashtags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Collect Instagram design posts as references (URLs + metadata only) - REAL DATA
        Uses VERIFIED design accounts and hashtags for high-quality curation
        """
        # VERIFIED DESIGN ACCOUNTS - HIGH QUALITY CURATION
        TIER_1_ACCOUNTS = [
            'logoinspirations',        # 1M - Best for logos
            'graphicdesigncentral',    # 450K - High quality curation
            'supplyanddesign',         # 178K - General design quality
            'thedesigntip',            # 700K - Design education
            'designarf'                # 181K - Illustration focus
        ]

        TIER_2_ACCOUNTS = [
            'creativroom',             # 79K - Creative showcase
            'design.feed',             # 77K - Curated feed
            'gfx_daily',               # 51K - 3D/Motion graphics
            'typography.studio'        # 172K - Typography specialist
        ]

        # TOP DESIGN HASHTAGS - VERIFIED ACTIVE
        if design_hashtags is None:
            design_hashtags = [
                "logodesign",              # 80M posts
                "brandidentity",           # 30M posts
                "graphicdesign",           # 150M posts
                "typography",              # 90M posts
                "uidesign",                # 25M posts
                "designinspiration"        # 50M posts
            ]

        references = []

        # COLLECTION PARAMETERS
        COLLECTION_RULES = {
            'min_engagement': 1000,      # Minimum likes for quality
            'max_per_source': 20,        # Limit per account (API constraints)
            'target_categories': [
                'logo', 'branding', 'social_media_post',
                'ui_design', 'typography', 'illustration'
            ],
            'exclude_keywords': [
                'selfie', 'food', 'fashion', 'fitness',
                'travel', 'personal', 'lifestyle'
            ]
        }

        try:
            # STRATEGY: Multi-tier collection from verified design sources
            logger.info("🎨 Collecting from VERIFIED design accounts and hashtags...")

            # Phase 1: Try to collect from top design accounts (if accessible)
            tier1_media = []
            logger.info(f"📍 Phase 1: Attempting Tier 1 accounts ({len(TIER_1_ACCOUNTS)} accounts)")

            for account in TIER_1_ACCOUNTS[:2]:  # Limit to 2 accounts for testing
                try:
                    # Note: This requires the account to be public or followed
                    account_media = await self.instagram_collector.collect_design_account_media(account, limit=20)
                    if account_media:
                        tier1_media.extend(account_media)
                        logger.info(f"✅ Collected {len(account_media)} posts from @{account}")
                except Exception as e:
                    logger.info(f"⚠️ Could not access @{account}: {e}")

            # Phase 2: Fallback to personal account with SMART FILTERING
            personal_media = []
            if len(tier1_media) < 10:  # If we didn't get enough from design accounts
                logger.info("📍 Phase 2: Using personal account with design filtering...")
                all_personal = await self.instagram_collector.collect_design_account_media("me", limit=50)

                # Apply SMART FILTERING for design relevance
                for media in all_personal:
                    caption = media.get("caption", "").lower()
                    if self._is_professional_design_post(caption, design_hashtags, COLLECTION_RULES):
                        personal_media.append(media)

            # Combine all sources
            real_media = tier1_media + personal_media

            logger.info(f"📊 Collection Summary:")
            logger.info(f"  Tier 1 accounts: {len(tier1_media)} posts")
            logger.info(f"  Filtered personal: {len(personal_media)} posts")
            logger.info(f"  Total quality posts: {len(real_media)}")

            # Convert to reference format with enhanced metadata
            for i, media in enumerate(real_media):
                caption = media.get("caption", "")
                design_elements = self._extract_design_elements_from_caption(caption)

                # Categorize the design type
                design_category = self._categorize_design_type(caption, design_elements)

                reference = {
                    "id": f"instagram_verified_{media.get('id')}",
                    "platform": "instagram",
                    "image_url": media.get("media_url"),
                    "caption": caption,
                    "author": media.get("username", "unknown"),
                    "timestamp": media.get("timestamp"),
                    "permalink": media.get("permalink"),
                    "media_type": media.get("media_type"),
                    "design_elements": design_elements,
                    "design_category": design_category,
                    "quality_score": self._calculate_quality_score(media, caption),
                    "source_tier": "tier1" if i < len(tier1_media) else "personal_filtered",
                    "reference_collected": datetime.now().isoformat(),
                    "fetch_status": "pending",
                    "data_source": "verified_design_api"
                }
                references.append(reference)

            logger.info(f"✅ Collected {len(references)} REAL Instagram design references")

        except Exception as e:
            logger.warning(f"Failed to collect real Instagram data: {e}")
            logger.info("Falling back to demo data for testing...")

            # Fallback to synthetic data if API fails
            for hashtag in design_hashtags[:2]:  # Limit for demo
                hashtag_refs = await self._collect_hashtag_references(hashtag)
                references.extend(hashtag_refs)

        return references

    async def collect_pinterest_design_references(self) -> List[Dict[str, Any]]:
        """
        Collect Pinterest design inspiration as references
        """
        design_categories = [
            "graphic design", "logo design", "ui ux design",
            "typography design", "color palette", "branding inspiration"
        ]

        references = []

        for category in design_categories[:2]:  # Limit for demo
            category_refs = await self._collect_pinterest_category_references(category)
            references.extend(category_refs)

        logger.info(f"Collected {len(references)} Pinterest design references")
        return references

    async def collect_dribbble_design_references(self) -> List[Dict[str, Any]]:
        """
        Collect Dribbble design portfolios as references
        """
        # Simulate Dribbble popular shots
        references = [
            {
                "id": f"dribbble_ref_{i}",
                "platform": "dribbble",
                "title": f"Modern Logo Design {i}",
                "description": "Clean, minimalist logo design with modern typography",
                "image_url": f"https://cdn.dribbble.com/users/sample/shots/{i}/shot.png",
                "author": f"designer_{i}",
                "tags": ["logo", "branding", "minimalist", "typography"],
                "stats": {
                    "views": 2500 + i * 100,
                    "likes": 180 + i * 10,
                    "saves": 45 + i * 5
                },
                "color_palette": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
                "design_type": "logo_design",
                "style": "minimalist",
                "created_date": (datetime.now() - timedelta(days=i)).isoformat(),
                "reference_collected": datetime.now().isoformat(),
                "fetch_status": "pending"
            }
            for i in range(1, 11)  # 10 sample references
        ]

        logger.info(f"Collected {len(references)} Dribbble design references")
        return references

    async def _collect_hashtag_references(self, hashtag: str) -> List[Dict[str, Any]]:
        """
        Collect references from Instagram hashtag (simulated)
        """
        # Simulate Instagram hashtag posts
        references = [
            {
                "id": f"instagram_{hashtag}_{i}",
                "platform": "instagram",
                "hashtag": hashtag,
                "image_url": f"https://instagram.com/p/sample_{hashtag}_{i}/media",
                "caption": f"Amazing {hashtag} inspiration! #designlife #creative",
                "author": f"designer_{hashtag}_{i}",
                "engagement": {
                    "likes": 500 + i * 50,
                    "comments": 20 + i * 2,
                    "shares": 10 + i,
                    "engagement_rate": (520 + i * 52) / 10000  # Simulated follower base
                },
                "posting_time": {
                    "hour": 14 + (i % 8),
                    "day_of_week": i % 7,
                    "is_weekend": (i % 7) >= 5
                },
                "design_elements": {
                    "has_text": True,
                    "design_style": "modern" if i % 2 == 0 else "vintage",
                    "color_scheme": "vibrant" if i % 3 == 0 else "muted"
                },
                "reference_collected": datetime.now().isoformat(),
                "fetch_status": "pending"
            }
            for i in range(1, 8)  # 7 posts per hashtag
        ]

        return references

    async def _collect_pinterest_category_references(self, category: str) -> List[Dict[str, Any]]:
        """
        Collect references from Pinterest design category (simulated)
        """
        references = [
            {
                "id": f"pinterest_{category.replace(' ', '_')}_{i}",
                "platform": "pinterest",
                "category": category,
                "image_url": f"https://i.pinimg.com/originals/sample_{category}_{i}.jpg",
                "title": f"Stunning {category} inspiration",
                "description": f"Beautiful example of {category} with modern aesthetics",
                "board_name": f"{category} Collection",
                "pin_stats": {
                    "saves": 150 + i * 20,
                    "comments": 8 + i,
                    "reactions": 25 + i * 3
                },
                "visual_features": {
                    "dominant_colors": ["#FF6B6B", "#4ECDC4"] if i % 2 == 0 else ["#333", "#FFF"],
                    "style": "minimalist" if i % 3 == 0 else "detailed",
                    "composition": "centered" if i % 2 == 0 else "asymmetric"
                },
                "design_relevance": 0.8 + (i % 3) * 0.05,
                "trending_score": 100 + i * 15,
                "reference_collected": datetime.now().isoformat(),
                "fetch_status": "pending"
            }
            for i in range(1, 12)  # 11 pins per category
        ]

        return references

    async def fetch_image_on_demand(self, reference: Dict[str, Any]) -> Optional[bytes]:
        """
        Fetch image data on-demand when needed for analysis
        """
        image_url = reference.get("image_url")
        if not image_url:
            return None

        # Check cache first
        cached_image = await self._get_cached_image(reference["id"])
        if cached_image:
            logger.info(f"Using cached image for {reference['id']}")
            return cached_image

        # Fetch from URL
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()

                        # Cache the image
                        await self._cache_image(reference["id"], image_data)

                        # Update reference status
                        reference["fetch_status"] = "success"
                        reference["last_fetched"] = datetime.now().isoformat()

                        logger.info(f"Fetched image for {reference['id']}")
                        return image_data
                    else:
                        logger.warning(f"Failed to fetch image for {reference['id']}: {response.status}")
                        reference["fetch_status"] = "failed"
                        return None

        except Exception as e:
            logger.error(f"Error fetching image for {reference['id']}: {e}")
            reference["fetch_status"] = "error"
            return None

    async def _get_cached_image(self, reference_id: str) -> Optional[bytes]:
        """
        Get image from cache if available and not expired
        """
        cache_file = self.cache_dir / f"{reference_id}.jpg"

        if cache_file.exists():
            # Check if cache is still valid
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time < self.cache_duration:
                with open(cache_file, "rb") as f:
                    return f.read()
            else:
                # Remove expired cache
                cache_file.unlink()

        return None

    async def _cache_image(self, reference_id: str, image_data: bytes):
        """
        Cache image data temporarily
        """
        try:
            # Clean cache if too large
            await self._clean_cache_if_needed()

            cache_file = self.cache_dir / f"{reference_id}.jpg"
            with open(cache_file, "wb") as f:
                f.write(image_data)

        except Exception as e:
            logger.warning(f"Failed to cache image {reference_id}: {e}")

    async def _clean_cache_if_needed(self):
        """
        Clean old cached images if cache is too large
        """
        cache_files = list(self.cache_dir.glob("*.jpg"))

        if len(cache_files) > self.max_cache_size:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda f: f.stat().st_mtime)

            for old_file in cache_files[:len(cache_files) - self.max_cache_size]:
                try:
                    old_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove old cache file {old_file}: {e}")

    def save_references_dataset(self, references: List[Dict[str, Any]], filename: str):
        """
        Save references dataset (metadata + URLs only)
        """
        # Save as JSON
        references_file = self.references_dir / f"{filename}.json"
        with open(references_file, "w") as f:
            json.dump(references, f, indent=2, default=str)

        # Create summary
        summary = {
            "total_references": len(references),
            "platforms": {},
            "categories": {},
            "collection_date": datetime.now().isoformat(),
            "data_type": "references_only",
            "total_file_size": references_file.stat().st_size
        }

        # Platform breakdown
        for ref in references:
            platform = ref.get("platform", "unknown")
            summary["platforms"][platform] = summary["platforms"].get(platform, 0) + 1

        # Category breakdown
        for ref in references:
            category = ref.get("category", ref.get("hashtag", "unknown"))
            summary["categories"][category] = summary["categories"].get(category, 0) + 1

        # Save summary
        summary_file = self.references_dir / f"{filename}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved {len(references)} design references to {references_file}")
        logger.info(f"Reference dataset size: {summary['total_file_size'] / 1024:.1f} KB (vs ~{len(references) * 500} KB for downloaded images)")

    def load_references_dataset(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load references dataset
        """
        references_file = self.references_dir / f"{filename}.json"

        if references_file.exists():
            with open(references_file, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"References file not found: {references_file}")
            return []

    async def analyze_reference_quality(self, references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze quality of collected references without downloading images
        """
        analysis = {
            "total_references": len(references),
            "platform_distribution": {},
            "design_categories": {},
            "engagement_stats": {
                "avg_likes": 0,
                "avg_shares": 0,
                "high_engagement_count": 0
            },
            "url_validity": {
                "valid_urls": 0,
                "invalid_urls": 0
            },
            "collection_freshness": {
                "recent": 0,  # Last 7 days
                "moderate": 0,  # 7-30 days
                "old": 0  # 30+ days
            }
        }

        total_likes = 0
        total_shares = 0

        for ref in references:
            # Platform distribution
            platform = ref.get("platform", "unknown")
            analysis["platform_distribution"][platform] = analysis["platform_distribution"].get(platform, 0) + 1

            # Engagement stats
            engagement = ref.get("engagement", ref.get("stats", ref.get("pin_stats", {})))
            likes = engagement.get("likes", engagement.get("saves", 0))
            shares = engagement.get("shares", engagement.get("comments", 0))

            total_likes += likes
            total_shares += shares

            if likes > 100:  # High engagement threshold
                analysis["engagement_stats"]["high_engagement_count"] += 1

            # URL validity (basic check)
            image_url = ref.get("image_url", "")
            if image_url and any(domain in image_url for domain in ["instagram.com", "pinimg.com", "dribbble.com"]):
                analysis["url_validity"]["valid_urls"] += 1
            else:
                analysis["url_validity"]["invalid_urls"] += 1

        # Calculate averages
        if len(references) > 0:
            analysis["engagement_stats"]["avg_likes"] = total_likes / len(references)
            analysis["engagement_stats"]["avg_shares"] = total_shares / len(references)

        # Quality score
        quality_score = (
            (analysis["url_validity"]["valid_urls"] / len(references)) * 0.4 +
            (analysis["engagement_stats"]["high_engagement_count"] / len(references)) * 0.3 +
            (len(analysis["platform_distribution"]) / 3) * 0.3  # Platform diversity
        ) * 100

        analysis["quality_score"] = round(quality_score, 2)

        return analysis

    def _extract_design_elements_from_caption(self, caption: str) -> Dict[str, Any]:
        """
        Extract design elements from Instagram caption text
        """
        if not caption:
            return {"has_text": False, "design_style": "unknown", "color_scheme": "unknown"}

        caption_lower = caption.lower()

        # Detect design style
        style = "unknown"
        if any(word in caption_lower for word in ["minimalist", "minimal", "clean", "simple"]):
            style = "minimalist"
        elif any(word in caption_lower for word in ["vintage", "retro", "classic"]):
            style = "vintage"
        elif any(word in caption_lower for word in ["modern", "contemporary", "fresh"]):
            style = "modern"

        # Detect color scheme
        color_scheme = "unknown"
        if any(word in caption_lower for word in ["colorful", "vibrant", "bright", "rainbow"]):
            color_scheme = "vibrant"
        elif any(word in caption_lower for word in ["monochrome", "black", "white", "gray"]):
            color_scheme = "monochrome"
        elif any(word in caption_lower for word in ["pastel", "soft", "muted"]):
            color_scheme = "muted"

        return {
            "has_text": len(caption) > 0,
            "design_style": style,
            "color_scheme": color_scheme,
            "caption_length": len(caption),
            "has_hashtags": "#" in caption
        }

    def _is_professional_design_post(self, caption: str, design_hashtags: List[str], rules: Dict[str, Any]) -> bool:
        """
        Enhanced filtering for PROFESSIONAL design content only
        """
        if not caption:
            return False

        # EXCLUDE personal/lifestyle content first
        exclude_keywords = rules.get('exclude_keywords', [])
        for exclude_word in exclude_keywords:
            if exclude_word in caption:
                return False

        # PROFESSIONAL DESIGN KEYWORDS (higher threshold)
        professional_keywords = [
            "design", "graphic", "logo", "brand", "typography", "layout",
            "visual", "creative", "portfolio", "project", "client", "concept",
            "identity", "branding", "illustration", "artwork", "composition",
            "minimalist", "modern", "inspiration", "designer", "studio"
        ]

        # VERIFIED DESIGN HASHTAGS (from your list)
        verified_hashtags = [
            "#logodesign", "#brandidentity", "#graphicdesign", "#typography",
            "#uidesign", "#designinspiration", "#logo", "#branding",
            "#design", "#creative", "#illustration", "#artwork", "#portfolio"
        ]

        # Check for verified design hashtags (strongest signal)
        hashtag_score = 0
        for hashtag in verified_hashtags:
            if hashtag in caption:
                hashtag_score += 2  # Strong design signal

        # Check for professional keywords
        keyword_score = sum(1 for keyword in professional_keywords if keyword in caption)

        # Enhanced criteria for professional content
        if hashtag_score >= 2:  # At least 1 verified design hashtag
            return True
        elif keyword_score >= 3:  # Multiple professional keywords
            return True
        elif any(tag in caption for tag in design_hashtags):  # Target hashtags
            return True

        return False

    def _categorize_design_type(self, caption: str, design_elements: Dict[str, Any]) -> str:
        """
        Categorize the type of design based on content analysis
        """
        caption_lower = caption.lower()

        # Logo/Branding detection
        if any(word in caption_lower for word in ["logo", "logodesign", "brand", "identity"]):
            return "logo_branding"

        # Typography detection
        if any(word in caption_lower for word in ["typography", "typeface", "font", "lettering"]):
            return "typography"

        # UI/UX detection
        if any(word in caption_lower for word in ["ui", "ux", "interface", "app", "website"]):
            return "ui_ux"

        # Illustration detection
        if any(word in caption_lower for word in ["illustration", "artwork", "drawing", "sketch"]):
            return "illustration"

        # Social media design
        if any(word in caption_lower for word in ["post", "social", "instagram", "template"]):
            return "social_media"

        # Photography/Visual
        if any(word in caption_lower for word in ["photo", "photography", "visual", "image"]):
            return "photography"

        return "general_design"

    def _calculate_quality_score(self, media: Dict[str, Any], caption: str) -> float:
        """
        Calculate quality score for design post (0.0 - 1.0)
        """
        score = 0.5  # Base score

        # Engagement indicators (if available)
        likes = media.get("like_count", 0)
        if likes > 10000:
            score += 0.3
        elif likes > 1000:
            score += 0.2
        elif likes > 100:
            score += 0.1

        # Caption quality
        if len(caption) > 50:  # Detailed description
            score += 0.1

        # Professional hashtags
        professional_tags = ["#design", "#logo", "#branding", "#typography", "#creative"]
        tag_count = sum(1 for tag in professional_tags if tag in caption.lower())
        score += min(tag_count * 0.05, 0.15)

        # Ensure score is within bounds
        return min(max(score, 0.0), 1.0)


# Main collection function
async def collect_design_references() -> Dict[str, Any]:
    """
    Main function to collect design references from all platforms
    """
    collector = ReferenceBasedDesignCollector()

    logger.info("🔗 Starting reference-based design data collection...")

    all_references = []

    # Collect from all platforms
    instagram_refs = await collector.collect_instagram_design_references()
    pinterest_refs = await collector.collect_pinterest_design_references()
    dribbble_refs = await collector.collect_dribbble_design_references()

    all_references.extend(instagram_refs)
    all_references.extend(pinterest_refs)
    all_references.extend(dribbble_refs)

    # Save references dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector.save_references_dataset(all_references, f"design_references_{timestamp}")

    # Analyze quality
    quality_analysis = await collector.analyze_reference_quality(all_references)

    result = {
        "references": all_references,
        "statistics": quality_analysis,
        "collection_type": "reference_based",
        "benefits": [
            "Faster collection (no downloads)",
            "Smaller storage footprint",
            "On-demand image fetching",
            "Real-time data access"
        ]
    }

    logger.info(f"🎨 Collected {len(all_references)} design references")
    logger.info(f"📊 Quality score: {quality_analysis['quality_score']}/100")

    return result


if __name__ == "__main__":
    asyncio.run(collect_design_references())