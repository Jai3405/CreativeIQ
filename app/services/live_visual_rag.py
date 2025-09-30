"""
Live Visual RAG System
Real-time design retrieval and analysis without static datasets
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
from datetime import datetime
import json
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class LiveVisualRAG:
    """
    Live Visual RAG system that searches design platforms in real-time
    No static datasets - retrieves similar designs on-demand
    """

    def __init__(self):
        self.dribbble_token = settings.DRIBBBLE_ACCESS_TOKEN
        self.behance_key = settings.BEHANCE_API_KEY

        # API endpoints
        self.dribbble_base = "https://api.dribbble.com/v2"
        self.behance_base = "https://api.behance.net/v2"

        logger.info("🔴 Live Visual RAG initialized - Real-time design search enabled")

    async def analyze_and_retrieve(self,
                                 user_image: Image.Image,
                                 user_goals: List[str] = None) -> Dict[str, Any]:
        """
        Main Live RAG function:
        1. Analyze user's design with VLM
        2. Generate smart search queries
        3. Search Dribbble + Behance in parallel
        4. Rank by visual similarity
        5. Generate data-driven recommendations
        """
        logger.info("🎨 Starting Live Visual RAG analysis...")

        # Step 1: VLM Analysis of user's design
        design_analysis = await self._analyze_user_design_with_vlm(user_image, user_goals)

        # Step 2: Generate targeted search queries
        search_queries = self._generate_smart_queries(design_analysis)

        # Step 3: Live API searches in parallel
        retrieved_designs = await self._search_design_platforms_parallel(search_queries)

        # Step 4: Visual similarity ranking (if we have enough results)
        if len(retrieved_designs) > 10:
            similar_designs = await self._rank_by_visual_similarity(user_image, retrieved_designs)
        else:
            similar_designs = retrieved_designs

        # Step 5: Comparative analysis and recommendations
        recommendations = await self._generate_data_driven_recommendations(
            design_analysis, similar_designs
        )

        return {
            "user_design_analysis": design_analysis,
            "search_queries_used": search_queries,
            "retrieved_designs": similar_designs[:20],  # Top 20
            "total_found": len(retrieved_designs),
            "recommendations": recommendations,
            "metadata": {
                "search_time": datetime.now().isoformat(),
                "platforms_searched": ["dribbble", "behance"],
                "analysis_method": "live_visual_rag"
            }
        }

    async def _analyze_user_design_with_vlm(self,
                                          image: Image.Image,
                                          user_goals: List[str] = None) -> Dict[str, Any]:
        """
        Use VLM (LLaVA/Qwen2-VL) to analyze user's design
        Extract style, colors, type, composition, industry
        """
        # TODO: Integrate actual VLM model (LLaVA-1.6 or Qwen2-VL)
        # For now, return structured analysis based on image properties

        width, height = image.size
        aspect_ratio = width / height

        # Basic analysis (will be replaced with VLM)
        analysis = {
            "design_type": self._infer_design_type(width, height, aspect_ratio),
            "style_keywords": self._extract_style_keywords(image),
            "color_palette": self._extract_color_palette(image),
            "composition": self._analyze_composition(image),
            "complexity": self._estimate_complexity(width, height),
            "industry_hints": self._infer_industry(user_goals) if user_goals else [],
            "dimensions": {"width": width, "height": height, "aspect_ratio": aspect_ratio}
        }

        logger.info(f"📊 VLM Analysis: {analysis['design_type']} | {analysis['style_keywords']} | {len(analysis['color_palette'])} colors")
        return analysis

    def _generate_smart_queries(self, design_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate targeted search queries based on VLM analysis
        """
        design_type = design_analysis.get("design_type", "design")
        style_keywords = design_analysis.get("style_keywords", [])
        industry_hints = design_analysis.get("industry_hints", [])

        queries = []

        # Primary query: type + style
        if style_keywords:
            primary_style = style_keywords[0]
            queries.append(f"{primary_style} {design_type}")

        # Industry-specific queries
        for industry in industry_hints[:2]:  # Top 2 industries
            queries.append(f"{industry} {design_type}")

        # Style combinations
        if len(style_keywords) > 1:
            style_combo = " ".join(style_keywords[:2])
            queries.append(f"{style_combo} design")

        # Fallback generic query
        if not queries:
            queries.append(f"{design_type} design")

        logger.info(f"🔍 Generated search queries: {queries}")
        return queries[:4]  # Max 4 queries to avoid API limits

    async def _search_design_platforms_parallel(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Search Dribbble and Behance APIs in parallel for each query
        """
        all_designs = []

        # Create tasks for parallel execution
        search_tasks = []

        for query in queries:
            # Dribbble search
            if self.dribbble_token:
                search_tasks.append(self._search_dribbble(query))

            # Behance search
            if self.behance_key:
                search_tasks.append(self._search_behance(query))

        # Execute all searches in parallel
        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):  # Successful result
                    all_designs.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Search failed: {result}")

        # Remove duplicates based on URL or ID
        unique_designs = self._deduplicate_designs(all_designs)

        logger.info(f"🔍 Live search results: {len(unique_designs)} unique designs from {len(queries)} queries")
        return unique_designs

    async def _search_dribbble(self, query: str, limit: int = 24) -> List[Dict[str, Any]]:
        """
        Search Dribbble API for designs matching query
        """
        if not self.dribbble_token:
            logger.warning("Dribbble token not configured")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.dribbble_base}/shots"
                params = {
                    "q": query,
                    "per_page": limit,
                    "access_token": self.dribbble_token
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        designs = []

                        for shot in data.get("data", []):
                            design = {
                                "id": f"dribbble_{shot['id']}",
                                "platform": "dribbble",
                                "title": shot.get("title", ""),
                                "image_url": shot.get("images", {}).get("normal", ""),
                                "thumbnail_url": shot.get("images", {}).get("teaser", ""),
                                "author": shot.get("user", {}).get("name", ""),
                                "author_url": shot.get("user", {}).get("html_url", ""),
                                "url": shot.get("html_url", ""),
                                "stats": {
                                    "views": shot.get("views_count", 0),
                                    "likes": shot.get("likes_count", 0),
                                    "comments": shot.get("comments_count", 0)
                                },
                                "tags": shot.get("tags", []),
                                "created_at": shot.get("created_at", ""),
                                "search_query": query
                            }
                            designs.append(design)

                        logger.info(f"✅ Dribbble: {len(designs)} results for '{query}'")
                        return designs

                    else:
                        logger.warning(f"Dribbble API error: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Dribbble search error: {e}")
            return []

    async def _search_behance(self, query: str, limit: int = 24) -> List[Dict[str, Any]]:
        """
        Search Behance API for projects matching query
        """
        if not self.behance_key:
            logger.warning("Behance API key not configured")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.behance_base}/search/projects"
                params = {
                    "q": query,
                    "per_page": limit,
                    "api_key": self.behance_key,
                    "field": "graphic design"
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        designs = []

                        for project in data.get("search_results", []):
                            # Get covers (thumbnail images)
                            covers = project.get("covers", {})
                            thumbnail_url = ""
                            if covers:
                                thumbnail_url = covers.get("230", covers.get("202", ""))

                            design = {
                                "id": f"behance_{project['id']}",
                                "platform": "behance",
                                "title": project.get("name", ""),
                                "image_url": thumbnail_url,
                                "thumbnail_url": thumbnail_url,
                                "author": ", ".join([owner.get("display_name", "") for owner in project.get("owners", [])]),
                                "author_url": project.get("owners", [{}])[0].get("url", "") if project.get("owners") else "",
                                "url": project.get("url", ""),
                                "stats": {
                                    "views": project.get("stats", {}).get("views", 0),
                                    "likes": project.get("stats", {}).get("appreciations", 0),
                                    "comments": project.get("stats", {}).get("comments", 0)
                                },
                                "tags": project.get("tags", []),
                                "created_at": project.get("created_on", ""),
                                "search_query": query
                            }
                            designs.append(design)

                        logger.info(f"✅ Behance: {len(designs)} results for '{query}'")
                        return designs

                    else:
                        logger.warning(f"Behance API error: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Behance search error: {e}")
            return []

    def _deduplicate_designs(self, designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate designs based on URL or title similarity
        """
        seen_urls = set()
        seen_titles = set()
        unique_designs = []

        for design in designs:
            url = design.get("url", "")
            title = design.get("title", "").lower().strip()

            # Skip if we've seen this URL or very similar title
            if url in seen_urls or title in seen_titles:
                continue

            seen_urls.add(url)
            seen_titles.add(title)
            unique_designs.append(design)

        return unique_designs

    async def _rank_by_visual_similarity(self,
                                       user_image: Image.Image,
                                       designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use CLIP embeddings to rank designs by visual similarity to user's image
        TODO: Implement CLIP model for visual similarity scoring
        """
        # Placeholder: For now, return designs sorted by engagement
        # This will be replaced with actual CLIP similarity scoring

        def engagement_score(design):
            stats = design.get("stats", {})
            likes = stats.get("likes", 0)
            views = stats.get("views", 1)  # Avoid division by zero
            return likes / max(views, 1) * 1000  # Engagement rate * 1000

        sorted_designs = sorted(designs, key=engagement_score, reverse=True)

        logger.info(f"📊 Ranked {len(sorted_designs)} designs by engagement (CLIP ranking coming soon)")
        return sorted_designs

    async def _generate_data_driven_recommendations(self,
                                                  user_analysis: Dict[str, Any],
                                                  similar_designs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate recommendations based on comparison with retrieved designs
        """
        if not similar_designs:
            return {"error": "No similar designs found for comparison"}

        # Analyze patterns in top-performing designs
        top_designs = similar_designs[:10]  # Top 10 by similarity/engagement

        # Extract patterns
        total_likes = sum(d.get("stats", {}).get("likes", 0) for d in top_designs)
        total_views = sum(d.get("stats", {}).get("views", 1) for d in top_designs)
        avg_engagement = (total_likes / max(total_views, 1)) * 100

        # Common platforms
        platform_counts = {}
        for design in top_designs:
            platform = design.get("platform", "unknown")
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        best_platform = max(platform_counts.items(), key=lambda x: x[1])[0] if platform_counts else "dribbble"

        recommendations = {
            "performance_insights": [
                f"Similar {user_analysis.get('design_type', 'designs')} average {avg_engagement:.1f}% engagement rate",
                f"Top performers are most common on {best_platform}",
                f"Analyzed {len(similar_designs)} professional examples"
            ],
            "style_recommendations": self._analyze_style_trends(top_designs),
            "best_platform": best_platform,
            "engagement_benchmark": {
                "average_likes": total_likes / len(top_designs),
                "average_views": total_views / len(top_designs),
                "engagement_rate": avg_engagement
            },
            "similar_designs_analyzed": len(similar_designs)
        }

        return recommendations

    def _analyze_style_trends(self, designs: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze style trends from top-performing designs
        """
        trends = []

        # Analyze titles and tags for common patterns
        all_text = []
        for design in designs:
            title = design.get("title", "").lower()
            tags = [tag.lower() for tag in design.get("tags", [])]
            all_text.extend([title] + tags)

        # Find common style keywords
        style_keywords = ["minimal", "modern", "vintage", "bold", "clean", "creative", "professional"]
        found_styles = []

        for keyword in style_keywords:
            count = sum(1 for text in all_text if keyword in text)
            if count >= 2:  # At least 2 mentions
                found_styles.append(keyword)

        if found_styles:
            trends.append(f"Popular styles: {', '.join(found_styles[:3])}")

        return trends

    # Helper methods for basic analysis (until VLM integration)
    def _infer_design_type(self, width: int, height: int, aspect_ratio: float) -> str:
        """Infer design type from dimensions"""
        if abs(aspect_ratio - 1.0) < 0.1:
            return "logo"
        elif aspect_ratio > 1.5:
            return "banner"
        elif aspect_ratio < 0.7:
            return "poster"
        else:
            return "social_post"

    def _extract_style_keywords(self, image: Image.Image) -> List[str]:
        """Extract style keywords (placeholder for VLM)"""
        return ["modern", "clean"]

    def _extract_color_palette(self, image: Image.Image) -> List[str]:
        """Extract color palette (placeholder for VLM)"""
        return ["#2C3E50", "#3498DB"]

    def _analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze composition (placeholder for VLM)"""
        return {"symmetry": "balanced", "complexity": "moderate"}

    def _estimate_complexity(self, width: int, height: int) -> str:
        """Estimate design complexity"""
        pixel_count = width * height
        if pixel_count > 1000000:
            return "complex"
        elif pixel_count > 400000:
            return "moderate"
        else:
            return "simple"

    def _infer_industry(self, user_goals: List[str]) -> List[str]:
        """Infer industry from user goals"""
        industry_keywords = {
            "tech": ["technology", "startup", "app", "software", "digital"],
            "fashion": ["fashion", "clothing", "style", "brand"],
            "food": ["restaurant", "food", "cafe", "dining"],
            "health": ["medical", "health", "wellness", "fitness"],
            "finance": ["financial", "banking", "investment", "money"]
        }

        detected_industries = []
        for goal in user_goals:
            goal_lower = goal.lower()
            for industry, keywords in industry_keywords.items():
                if any(keyword in goal_lower for keyword in keywords):
                    detected_industries.append(industry)

        return list(set(detected_industries))  # Remove duplicates


# Demo function
async def demo_live_visual_rag():
    """
    Demo the Live Visual RAG system
    """
    print("🔴 Live Visual RAG Demo")
    print("=" * 40)

    # Create test image
    from PIL import Image
    test_image = Image.new('RGB', (400, 400), (100, 150, 200))

    rag = LiveVisualRAG()

    result = await rag.analyze_and_retrieve(
        test_image,
        user_goals=["professional logo", "tech startup"]
    )

    print(f"📊 Analysis Results:")
    print(f"  Design type: {result['user_design_analysis']['design_type']}")
    print(f"  Search queries: {result['search_queries_used']}")
    print(f"  Designs found: {result['total_found']}")
    print(f"  Platforms: {result['metadata']['platforms_searched']}")

    if result['retrieved_designs']:
        print(f"  Sample result: {result['retrieved_designs'][0]['title']}")

    print(f"🎯 Recommendations:")
    for insight in result['recommendations'].get('performance_insights', []):
        print(f"  • {insight}")


if __name__ == "__main__":
    asyncio.run(demo_live_visual_rag())