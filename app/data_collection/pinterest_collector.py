"""
Pinterest Design Data Collector
Collects design inspiration and visual content from Pinterest API
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import hashlib

from app.core.config import settings

logger = logging.getLogger(__name__)


class PinterestDesignCollector:
    """
    Collects design inspiration data from Pinterest API
    Focuses on design-related boards and pins
    """

    def __init__(self):
        self.access_token = settings.PINTEREST_ACCESS_TOKEN
        self.base_url = "https://api.pinterest.com/v5"

        if not self.access_token:
            logger.warning("Pinterest access token not configured")

    async def collect_design_inspiration(self, design_categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Collect design inspiration pins from Pinterest
        """
        if design_categories is None:
            design_categories = [
                "graphic design",
                "logo design",
                "brand identity",
                "ui ux design",
                "typography design",
                "color palette",
                "layout design",
                "branding inspiration",
                "creative design",
                "design trends"
            ]

        collected_data = []

        async with aiohttp.ClientSession() as session:
            for category in design_categories:
                try:
                    # Search for pins in design category
                    pins = await self._search_design_pins(session, category, limit=50)

                    for pin in pins:
                        # Download pin image
                        image_path = await self._download_pin_image(session, pin["media"]["images"]["original"]["url"], pin["id"])

                        if image_path:
                            pin_data = {
                                "id": f"pinterest_{pin['id']}",
                                "platform": "pinterest",
                                "category": category,
                                "image_path": str(image_path),
                                "title": pin.get("title", ""),
                                "description": pin.get("description", ""),
                                "alt_text": pin.get("alt_text", ""),
                                "link": pin.get("link", ""),
                                "board_name": pin.get("board_name", ""),
                                "engagement_metrics": {
                                    "save_count": pin.get("save_count", 0),
                                    "comment_count": pin.get("comment_count", 0),
                                    "reaction_count": pin.get("reaction_count", 0)
                                },
                                "visual_features": self._extract_visual_features(pin),
                                "design_tags": self._extract_design_tags(pin.get("description", "") + " " + pin.get("title", "")),
                                "color_dominance": await self._analyze_pin_colors(session, pin["media"]["images"]["original"]["url"]),
                                "collection_date": datetime.now().isoformat()
                            }

                            collected_data.append(pin_data)
                            logger.info(f"Collected Pinterest pin: {pin['id']} in category {category}")

                except Exception as e:
                    logger.error(f"Error collecting Pinterest data for category {category}: {e}")

                # Rate limiting
                await asyncio.sleep(1)

        return collected_data

    async def collect_design_boards(self, board_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Collect pins from design-focused boards
        """
        if board_keywords is None:
            board_keywords = [
                "graphic design inspiration",
                "logo collection",
                "brand identity design",
                "ui design",
                "typography",
                "color schemes",
                "creative layouts"
            ]

        collected_data = []

        async with aiohttp.ClientSession() as session:
            for keyword in board_keywords:
                try:
                    # Search for design boards
                    boards = await self._search_design_boards(session, keyword)

                    for board in boards[:3]:  # Top 3 boards per keyword
                        # Get pins from this board
                        board_pins = await self._get_board_pins(session, board["id"], limit=25)

                        for pin in board_pins:
                            if pin.get("media", {}).get("images", {}).get("original"):
                                image_path = await self._download_pin_image(
                                    session,
                                    pin["media"]["images"]["original"]["url"],
                                    f"board_{board['id']}_{pin['id']}"
                                )

                                if image_path:
                                    pin_data = {
                                        "id": f"pinterest_board_{board['id']}_{pin['id']}",
                                        "platform": "pinterest_board",
                                        "board_keyword": keyword,
                                        "board_name": board.get("name", ""),
                                        "board_description": board.get("description", ""),
                                        "image_path": str(image_path),
                                        "title": pin.get("title", ""),
                                        "description": pin.get("description", ""),
                                        "engagement_metrics": {
                                            "save_count": pin.get("save_count", 0),
                                            "comment_count": pin.get("comment_count", 0)
                                        },
                                        "design_context": {
                                            "board_follower_count": board.get("follower_count", 0),
                                            "board_pin_count": board.get("pin_count", 0)
                                        },
                                        "collection_date": datetime.now().isoformat()
                                    }

                                    collected_data.append(pin_data)

                except Exception as e:
                    logger.error(f"Error collecting Pinterest board data for {keyword}: {e}")

                await asyncio.sleep(2)  # Rate limiting

        return collected_data

    async def collect_trending_design_pins(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect trending pins in design categories
        """
        collected_data = []

        async with aiohttp.ClientSession() as session:
            try:
                # Get trending pins in design category
                url = f"{self.base_url}/pins"
                params = {
                    "pin_filter": "all",
                    "page_size": limit,
                    "access_token": self.access_token
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pins = data.get("items", [])

                        for pin in pins:
                            # Filter for design-related content
                            if self._is_design_related(pin):
                                image_path = await self._download_pin_image(
                                    session,
                                    pin["media"]["images"]["original"]["url"],
                                    f"trending_{pin['id']}"
                                )

                                if image_path:
                                    pin_data = {
                                        "id": f"pinterest_trending_{pin['id']}",
                                        "platform": "pinterest_trending",
                                        "image_path": str(image_path),
                                        "title": pin.get("title", ""),
                                        "description": pin.get("description", ""),
                                        "engagement_metrics": {
                                            "save_count": pin.get("save_count", 0),
                                            "comment_count": pin.get("comment_count", 0)
                                        },
                                        "trending_score": self._calculate_trending_score(pin),
                                        "design_relevance": self._assess_design_relevance(pin),
                                        "collection_date": datetime.now().isoformat()
                                    }

                                    collected_data.append(pin_data)

                    else:
                        logger.error(f"Pinterest API error: {response.status}")

            except Exception as e:
                logger.error(f"Error collecting trending Pinterest data: {e}")

        return collected_data

    async def _search_design_pins(self, session: aiohttp.ClientSession, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for pins related to design query
        """
        try:
            url = f"{self.base_url}/search/pins"
            params = {
                "query": query,
                "limit": limit,
                "access_token": self.access_token
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])
                else:
                    logger.error(f"Pinterest search error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error searching Pinterest pins: {e}")
            return []

    async def _search_design_boards(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        """
        Search for design-related boards
        """
        try:
            url = f"{self.base_url}/search/boards"
            params = {
                "query": query,
                "access_token": self.access_token
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])
                else:
                    return []

        except Exception as e:
            logger.error(f"Error searching Pinterest boards: {e}")
            return []

    async def _get_board_pins(self, session: aiohttp.ClientSession, board_id: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Get pins from a specific board
        """
        try:
            url = f"{self.base_url}/boards/{board_id}/pins"
            params = {
                "page_size": limit,
                "access_token": self.access_token
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])
                else:
                    return []

        except Exception as e:
            logger.error(f"Error getting board pins: {e}")
            return []

    async def _download_pin_image(self, session: aiohttp.ClientSession, image_url: str, pin_id: str) -> Optional[Path]:
        """
        Download Pinterest pin image
        """
        try:
            async with session.get(image_url) as response:
                if response.status == 200:
                    content = await response.read()

                    # Create filename
                    file_hash = hashlib.md5(content).hexdigest()[:8]
                    filename = f"pinterest_{pin_id}_{file_hash}.jpg"

                    # Save to images directory
                    images_dir = Path("data/training/images")
                    images_dir.mkdir(parents=True, exist_ok=True)

                    file_path = images_dir / filename
                    with open(file_path, "wb") as f:
                        f.write(content)

                    logger.info(f"Downloaded Pinterest image: {filename}")
                    return file_path

        except Exception as e:
            logger.error(f"Error downloading Pinterest image {pin_id}: {e}")

        return None

    def _extract_visual_features(self, pin: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract visual features from pin metadata
        """
        features = {}

        # Image dimensions
        if "media" in pin and "images" in pin["media"]:
            original = pin["media"]["images"].get("original", {})
            features["width"] = original.get("width", 0)
            features["height"] = original.get("height", 0)
            features["aspect_ratio"] = features["width"] / features["height"] if features["height"] > 0 else 0

        # Extract color info if available
        features["has_text"] = bool(pin.get("title") or pin.get("description"))

        return features

    def _extract_design_tags(self, text: str) -> List[str]:
        """
        Extract design-related tags from pin text
        """
        design_keywords = [
            "logo", "branding", "typography", "color", "palette", "layout",
            "design", "graphic", "visual", "creative", "inspiration", "ui", "ux",
            "minimalist", "modern", "vintage", "clean", "professional", "elegant"
        ]

        text_lower = text.lower()
        found_tags = [keyword for keyword in design_keywords if keyword in text_lower]
        return found_tags

    async def _analyze_pin_colors(self, session: aiohttp.ClientSession, image_url: str) -> Dict[str, Any]:
        """
        Basic color analysis from pin metadata
        Note: Actual color analysis would require image processing
        """
        # Placeholder for color analysis
        # In a full implementation, this would download and analyze the image
        return {
            "dominant_colors": [],
            "color_count": 0,
            "brightness": "unknown",
            "saturation": "unknown"
        }

    def _is_design_related(self, pin: Dict[str, Any]) -> bool:
        """
        Check if pin is related to design
        """
        text = (pin.get("title", "") + " " + pin.get("description", "")).lower()
        design_terms = ["design", "logo", "brand", "graphic", "ui", "ux", "typography", "color", "creative"]
        return any(term in text for term in design_terms)

    def _calculate_trending_score(self, pin: Dict[str, Any]) -> float:
        """
        Calculate trending score based on engagement
        """
        saves = pin.get("save_count", 0)
        comments = pin.get("comment_count", 0)

        # Simple trending score calculation
        return saves * 1.0 + comments * 2.0

    def _assess_design_relevance(self, pin: Dict[str, Any]) -> float:
        """
        Assess how relevant the pin is to design
        """
        text = (pin.get("title", "") + " " + pin.get("description", "")).lower()
        design_terms = ["design", "logo", "brand", "graphic", "ui", "ux", "typography", "color", "creative", "inspiration"]

        relevance_score = sum(1 for term in design_terms if term in text)
        return min(relevance_score / len(design_terms), 1.0)

    async def validate_access_token(self) -> bool:
        """
        Validate Pinterest access token
        """
        if not self.access_token:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/user_account"
                params = {"access_token": self.access_token}

                async with session.get(url, params=params) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error validating Pinterest token: {e}")
            return False


# Example usage
async def collect_pinterest_design_data_example():
    """
    Example of collecting design-focused Pinterest data
    """
    collector = PinterestDesignCollector()

    # Validate token first
    if not await collector.validate_access_token():
        logger.error("Invalid Pinterest access token")
        return

    # Collect design inspiration
    inspiration_data = await collector.collect_design_inspiration()

    # Collect from design boards
    board_data = await collector.collect_design_boards()

    # Collect trending design pins
    trending_data = await collector.collect_trending_design_pins(limit=50)

    # Combine all data
    all_data = inspiration_data + board_data + trending_data

    if all_data:
        output_file = Path("data/training/metadata/pinterest_design_data.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2, default=str)

        logger.info(f"Collected {len(all_data)} Pinterest design pins")
        return all_data

    return []


if __name__ == "__main__":
    asyncio.run(collect_pinterest_design_data_example())