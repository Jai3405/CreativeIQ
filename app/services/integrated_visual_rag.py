"""
Integrated Reference-Based Visual RAG System
Combines reference collection with intelligent visual retrieval
No downloads - pure metadata search with on-demand fetching
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import re

from app.services.on_demand_image_fetcher import OnDemandImageFetcher

logger = logging.getLogger(__name__)


class IntegratedVisualRAG:
    """
    Smart Visual RAG that works with reference-based collection
    No downloads required - intelligent metadata search + on-demand fetching
    """

    def __init__(self):
        self.references_dir = Path("data/training/references")
        self.design_database = []
        self.metadata_index = {}

        logger.info("🔗🧠 Initialized Integrated Visual RAG")

    async def load_design_references(self, references_file: str = None):
        """
        Load design references from collection system
        """
        if references_file:
            ref_file = self.references_dir / f"{references_file}.json"
        else:
            # Find most recent references (exclude summary files)
            ref_files = [f for f in self.references_dir.glob("design_references_*.json")
                        if not f.name.endswith("_summary.json")]
            if not ref_files:
                logger.warning("No design references found. Run: make collect-design-references")
                return
            ref_file = max(ref_files, key=lambda f: f.stat().st_mtime)

        if ref_file.exists():
            with open(ref_file, 'r') as f:
                self.design_database = json.load(f)

            # Build smart metadata index
            await self._build_metadata_index()

            logger.info(f"📂 Loaded {len(self.design_database)} design references for RAG")
        else:
            logger.warning(f"References file not found: {ref_file}")

    async def analyze_user_design_with_rag(self,
                                          user_image: Image.Image,
                                          user_goals: List[str] = None) -> Dict[str, Any]:
        """
        Main RAG function: Analyze user design with retrieved similar designs
        """
        logger.info("🎨 Analyzing user design with Visual RAG")

        # Step 1: Extract user image characteristics (no external downloads)
        user_features = self._extract_user_image_features(user_image)

        # Step 2: Intelligent retrieval from metadata (no downloads)
        retrieved_designs = await self._smart_retrieve_similar_designs(user_features, user_goals)

        # Step 3: On-demand fetch only the most relevant designs
        comparative_analysis = await self._on_demand_comparative_analysis(
            user_image, user_features, retrieved_designs[:3]  # Only top 3
        )

        # Step 4: Generate RAG-enhanced insights
        enhanced_insights = await self._generate_rag_insights(
            user_features, retrieved_designs, comparative_analysis
        )

        return {
            "user_design_analysis": user_features,
            "retrieved_references": retrieved_designs,
            "comparative_analysis": comparative_analysis,
            "rag_enhanced_insights": enhanced_insights,
            "metadata": {
                "total_database_size": len(self.design_database),
                "retrieved_count": len(retrieved_designs),
                "images_fetched": len([r for r in comparative_analysis if r.get('image_fetched')]),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }

    async def get_design_inspiration(self,
                                   inspiration_query: str,
                                   style_filters: List[str] = None) -> Dict[str, Any]:
        """
        Get design inspiration using RAG without downloads
        """
        logger.info(f"💡 Getting design inspiration for: {inspiration_query}")

        # Smart search in metadata
        inspiration_designs = await self._search_inspiration_metadata(inspiration_query, style_filters)

        # Analyze patterns in retrieved designs (metadata only)
        patterns = self._analyze_design_patterns_from_metadata(inspiration_designs)

        # Generate inspiration insights
        insights = self._generate_inspiration_insights(inspiration_query, inspiration_designs, patterns)

        return {
            "query": inspiration_query,
            "style_filters": style_filters,
            "inspiration_designs": inspiration_designs[:10],  # Top 10
            "design_patterns": patterns,
            "inspiration_insights": insights,
            "actionable_tips": self._generate_actionable_tips(patterns)
        }

    async def _build_metadata_index(self):
        """
        Build searchable index from metadata (no image processing)
        """
        self.metadata_index = {
            "by_keywords": {},
            "by_colors": {},
            "by_platform": {},
            "by_engagement": {},
            "by_style": {}
        }

        for i, design in enumerate(self.design_database):
            # Index by keywords
            keywords = self._extract_keywords_from_design(design)
            for keyword in keywords:
                if keyword not in self.metadata_index["by_keywords"]:
                    self.metadata_index["by_keywords"][keyword] = []
                self.metadata_index["by_keywords"][keyword].append(i)

            # Index by colors (if available in metadata)
            colors = design.get('color_palette', design.get('visual_features', {}).get('dominant_colors', []))
            for color in colors:
                if color not in self.metadata_index["by_colors"]:
                    self.metadata_index["by_colors"][color] = []
                self.metadata_index["by_colors"][color].append(i)

            # Index by platform
            platform = design.get('platform', 'unknown')
            if platform not in self.metadata_index["by_platform"]:
                self.metadata_index["by_platform"][platform] = []
            self.metadata_index["by_platform"][platform].append(i)

            # Index by engagement level
            engagement = self._calculate_engagement_score(design)
            engagement_tier = "high" if engagement > 1000 else "medium" if engagement > 100 else "low"
            if engagement_tier not in self.metadata_index["by_engagement"]:
                self.metadata_index["by_engagement"][engagement_tier] = []
            self.metadata_index["by_engagement"][engagement_tier].append(i)

        logger.info("📇 Built metadata search index")

    def _extract_user_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract features from user's image (local processing only)
        """
        # Basic image properties
        width, height = image.size
        aspect_ratio = width / height

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract basic color info without numpy
        # Sample center pixel for color analysis
        center_x, center_y = width // 2, height // 2
        try:
            center_pixel = image.getpixel((center_x, center_y))
            avg_red, avg_green, avg_blue = center_pixel[:3]
        except:
            # Fallback if getpixel fails
            avg_red, avg_green, avg_blue = 128, 128, 128

        avg_brightness = (avg_red + avg_green + avg_blue) / 3

        # Infer design type from dimensions
        design_type = self._infer_design_type(width, height, aspect_ratio)

        # Color characteristics
        color_temperature = "warm" if avg_red > avg_blue else "cool"
        brightness_level = "bright" if avg_brightness > 180 else "dark" if avg_brightness < 80 else "medium"

        return {
            "dimensions": {"width": width, "height": height},
            "aspect_ratio": round(aspect_ratio, 2),
            "design_type": design_type,
            "color_info": {
                "avg_brightness": round(avg_brightness, 1),
                "color_temperature": color_temperature,
                "brightness_level": brightness_level,
                "dominant_rgb": [round(avg_red), round(avg_green), round(avg_blue)]
            },
            "format_category": self._categorize_format(aspect_ratio),
            "complexity_estimate": self._estimate_complexity_simple(width, height, avg_brightness)
        }

    async def _smart_retrieve_similar_designs(self,
                                            user_features: Dict[str, Any],
                                            user_goals: List[str] = None) -> List[Dict[str, Any]]:
        """
        Smart retrieval using metadata index (no downloads)
        """
        candidate_indices = set()

        # Search by design type
        design_type = user_features.get('design_type', '')
        for keyword in self.metadata_index.get("by_keywords", {}):
            if design_type.lower() in keyword.lower():
                candidate_indices.update(self.metadata_index["by_keywords"][keyword])

        # Search by user goals
        if user_goals:
            for goal in user_goals:
                for keyword in self.metadata_index.get("by_keywords", {}):
                    if any(word in keyword.lower() for word in goal.lower().split()):
                        candidate_indices.update(self.metadata_index["by_keywords"][keyword])

        # Search by format category
        format_cat = user_features.get('format_category', '')
        if format_cat in self.metadata_index.get("by_keywords", {}):
            candidate_indices.update(self.metadata_index["by_keywords"][format_cat])

        # Get high engagement designs
        if "high" in self.metadata_index.get("by_engagement", {}):
            high_engagement = set(self.metadata_index["by_engagement"]["high"][:50])  # Top 50
            candidate_indices.update(high_engagement)

        # Score and rank candidates
        scored_designs = []
        for idx in candidate_indices:
            if idx < len(self.design_database):
                design = self.design_database[idx].copy()
                score = self._calculate_similarity_score(user_features, design)
                design['relevance_score'] = score
                scored_designs.append(design)

        # Sort by relevance score
        scored_designs.sort(key=lambda x: x['relevance_score'], reverse=True)

        logger.info(f"🔍 Retrieved {len(scored_designs)} similar designs from metadata")
        return scored_designs[:10]  # Top 10

    async def _on_demand_comparative_analysis(self,
                                            user_image: Image.Image,
                                            user_features: Dict[str, Any],
                                            top_designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch and analyze only the top retrieved designs
        """
        comparative_results = []

        async with OnDemandImageFetcher() as fetcher:
            for design in top_designs:
                result = {
                    "design_id": design['id'],
                    "platform": design['platform'],
                    "relevance_score": design.get('relevance_score', 0),
                    "metadata_comparison": self._compare_metadata(user_features, design)
                }

                # Try to fetch image for visual comparison
                try:
                    fetched_image = await fetcher.fetch_image_for_analysis(design)
                    if fetched_image:
                        result['image_fetched'] = True
                        result['visual_comparison'] = self._compare_images(user_image, fetched_image)
                        result['design_insights'] = self._extract_design_insights(design, fetched_image)
                    else:
                        result['image_fetched'] = False
                        result['reason'] = 'Failed to fetch image'
                except Exception as e:
                    result['image_fetched'] = False
                    result['reason'] = f'Error: {str(e)}'

                comparative_results.append(result)

        fetched_count = len([r for r in comparative_results if r.get('image_fetched')])
        logger.info(f"🖼️ Fetched {fetched_count}/{len(top_designs)} reference images for comparison")

        return comparative_results

    async def _generate_rag_insights(self,
                                   user_features: Dict[str, Any],
                                   retrieved_designs: List[Dict[str, Any]],
                                   comparative_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate RAG-enhanced insights
        """
        insights = {
            "design_assessment": self._assess_user_design(user_features),
            "similar_designs_found": len(retrieved_designs),
            "performance_insights": [],
            "style_recommendations": [],
            "platform_insights": {},
            "engagement_predictions": {}
        }

        # Analyze performance patterns
        high_performers = [d for d in retrieved_designs if self._calculate_engagement_score(d) > 500]
        if high_performers:
            common_patterns = self._find_common_patterns(high_performers)
            insights["performance_insights"] = [
                f"High-performing {user_features['design_type']} designs often use {pattern}"
                for pattern in common_patterns[:3]
            ]

        # Platform-specific insights
        platform_distribution = {}
        for design in retrieved_designs[:5]:
            platform = design.get('platform', 'unknown')
            platform_distribution[platform] = platform_distribution.get(platform, 0) + 1

        if platform_distribution:
            top_platform = max(platform_distribution, key=platform_distribution.get)
            insights["platform_insights"] = {
                "best_platform": top_platform,
                "distribution": platform_distribution,
                "recommendation": f"Your design style is popular on {top_platform}"
            }

        # Style recommendations from successful comparisons
        successful_comparisons = [c for c in comparative_analysis if c.get('image_fetched')]
        if successful_comparisons:
            for comp in successful_comparisons[:2]:
                if comp.get('visual_comparison', {}).get('similarity_score', 0) > 0.7:
                    insights["style_recommendations"].append(
                        f"Consider elements from {comp['platform']} design with {comp['relevance_score']:.2f} relevance"
                    )

        return insights

    def _extract_keywords_from_design(self, design: Dict[str, Any]) -> List[str]:
        """
        Extract searchable keywords from design metadata
        """
        keywords = []

        # From title
        title = design.get('title', '')
        keywords.extend(re.findall(r'\b\w+\b', title.lower()))

        # From description
        description = design.get('description', '')
        keywords.extend(re.findall(r'\b\w+\b', description.lower()))

        # From tags
        tags = design.get('tags', design.get('design_tags', []))
        keywords.extend([tag.lower() for tag in tags])

        # From category/hashtag
        category = design.get('category', design.get('hashtag', ''))
        if category:
            keywords.append(category.lower())

        # Design-specific keywords
        if design.get('platform') == 'pinterest':
            keywords.extend(['inspiration', 'trend', 'creative'])
        elif design.get('platform') == 'dribbble':
            keywords.extend(['portfolio', 'professional', 'showcase'])
        elif design.get('platform') == 'instagram':
            keywords.extend(['social', 'engagement', 'viral'])

        return list(set(keywords))  # Remove duplicates

    def _calculate_engagement_score(self, design: Dict[str, Any]) -> float:
        """
        Calculate engagement score from metadata
        """
        engagement = design.get('engagement_metrics', design.get('stats', design.get('pin_stats', {})))

        likes = engagement.get('likes', engagement.get('saves', 0))
        comments = engagement.get('comments', 0)
        shares = engagement.get('shares', engagement.get('reactions', 0))

        # Weighted engagement score
        return likes * 1.0 + comments * 2.0 + shares * 3.0

    def _infer_design_type(self, width: int, height: int, aspect_ratio: float) -> str:
        """
        Infer design type from dimensions
        """
        if abs(aspect_ratio - 1.0) < 0.1:
            return "logo" if min(width, height) < 800 else "square_post"
        elif aspect_ratio > 2.0:
            return "banner"
        elif aspect_ratio < 0.6:
            return "mobile_story"
        elif aspect_ratio > 1.3:
            return "landscape_post"
        else:
            return "graphic_design"

    def _categorize_format(self, aspect_ratio: float) -> str:
        """
        Categorize format for search
        """
        if aspect_ratio > 1.8:
            return "wide"
        elif aspect_ratio < 0.7:
            return "tall"
        else:
            return "standard"

    def _estimate_complexity_simple(self, width: int, height: int, brightness: float) -> str:
        """
        Estimate design complexity without image processing
        """
        # Simple heuristic based on dimensions and brightness
        pixel_count = width * height

        if pixel_count > 1000000:  # High resolution suggests detail
            return "complex"
        elif brightness > 200 or brightness < 50:  # Extreme brightness suggests complexity
            return "medium"
        else:
            return "simple"

    def _assess_user_design(self, user_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess user design characteristics
        """
        return {
            "format": user_features.get("format_category", "unknown"),
            "complexity": user_features.get("complexity_estimate", "simple"),
            "color_analysis": user_features.get("color_info", {}),
            "suitability": self._determine_design_suitability(user_features)
        }

    def _determine_design_suitability(self, user_features: Dict[str, Any]) -> List[str]:
        """
        Determine what the design is suitable for
        """
        suitability = []

        design_type = user_features.get("design_type", "")
        if "logo" in design_type:
            suitability.append("branding")
        if "square" in user_features.get("format_category", ""):
            suitability.append("social_media")
        if user_features.get("complexity_estimate") == "simple":
            suitability.append("professional_use")

        return suitability if suitability else ["general_purpose"]

    def _find_common_patterns(self, designs: List[Dict[str, Any]]) -> List[str]:
        """
        Find common patterns in high-performing designs
        """
        patterns = []

        # Analyze platform patterns
        platforms = {}
        for design in designs:
            platform = design.get("platform", "unknown")
            platforms[platform] = platforms.get(platform, 0) + 1

        if platforms:
            top_platform = max(platforms, key=platforms.get)
            patterns.append(f"{top_platform} platform features")

        # Analyze color patterns
        color_schemes = {}
        for design in designs:
            visual_features = design.get("visual_features", {})
            colors = visual_features.get("dominant_colors", [])
            if colors:
                scheme = "vibrant" if len(colors) > 2 else "minimal"
                color_schemes[scheme] = color_schemes.get(scheme, 0) + 1

        if color_schemes:
            top_scheme = max(color_schemes, key=color_schemes.get)
            patterns.append(f"{top_scheme} color schemes")

        return patterns[:3]  # Return top 3 patterns

    def _calculate_similarity_score(self, user_features: Dict[str, Any], design: Dict[str, Any]) -> float:
        """
        Calculate similarity score based on metadata
        """
        score = 0.0

        # Design type match
        design_keywords = self._extract_keywords_from_design(design)
        if user_features.get('design_type', '').lower() in design_keywords:
            score += 0.4

        # Format match
        if user_features.get('format_category') in design_keywords:
            score += 0.2

        # Engagement boost
        engagement = self._calculate_engagement_score(design)
        if engagement > 1000:
            score += 0.2
        elif engagement > 100:
            score += 0.1

        # Platform preference (Pinterest and Dribbble often have higher quality)
        platform = design.get('platform', '')
        if platform in ['pinterest', 'dribbble']:
            score += 0.1

        return score

    def _compare_metadata(self, user_features: Dict[str, Any], design: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare user design with reference design metadata
        """
        return {
            "design_type_match": user_features.get('design_type') in self._extract_keywords_from_design(design),
            "engagement_level": "high" if self._calculate_engagement_score(design) > 500 else "medium",
            "platform": design.get('platform'),
            "estimated_performance": "good" if self._calculate_engagement_score(design) > 200 else "average"
        }

    def _compare_images(self, user_image: Image.Image, reference_image: Image.Image) -> Dict[str, Any]:
        """
        Basic visual comparison (simplified)
        """
        import numpy as np

        # Resize for comparison
        user_resized = user_image.resize((100, 100))
        ref_resized = reference_image.resize((100, 100))

        # Convert to arrays
        user_array = np.array(user_resized)
        ref_array = np.array(ref_resized)

        # Simple similarity metric
        diff = np.abs(user_array.astype(float) - ref_array.astype(float))
        similarity = 1.0 - (np.mean(diff) / 255.0)

        return {
            "similarity_score": max(0.0, similarity),
            "color_similarity": "high" if similarity > 0.8 else "medium" if similarity > 0.5 else "low"
        }

    def _extract_design_insights(self, design: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
        """
        Extract insights from successfully fetched design
        """
        return {
            "success_factors": [
                f"High engagement ({self._calculate_engagement_score(design):.0f} points)",
                f"Popular on {design.get('platform', 'unknown')}",
                f"Effective {design.get('category', 'design')} example"
            ],
            "visual_elements": {
                "dimensions": image.size,
                "aspect_ratio": round(image.size[0] / image.size[1], 2)
            }
        }

    async def _search_inspiration_metadata(self, query: str, style_filters: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for inspiration using metadata only
        """
        query_words = query.lower().split()
        matching_designs = []

        for design in self.design_database:
            keywords = self._extract_keywords_from_design(design)

            # Check query match
            if any(word in keywords for word in query_words):
                match_score = sum(1 for word in query_words if word in keywords)
                design_copy = design.copy()
                design_copy['match_score'] = match_score
                matching_designs.append(design_copy)

        # Filter by style if provided
        if style_filters:
            filtered_designs = []
            for design in matching_designs:
                keywords = self._extract_keywords_from_design(design)
                if any(style.lower() in keywords for style in style_filters):
                    filtered_designs.append(design)
            matching_designs = filtered_designs

        # Sort by match score and engagement
        matching_designs.sort(
            key=lambda x: (x.get('match_score', 0), self._calculate_engagement_score(x)),
            reverse=True
        )

        return matching_designs

    def _analyze_design_patterns_from_metadata(self, designs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns from metadata only
        """
        if not designs:
            return {}

        platforms = [d.get('platform') for d in designs]
        categories = [d.get('category', d.get('hashtag', '')) for d in designs]

        return {
            "popular_platforms": {p: platforms.count(p) for p in set(platforms)},
            "common_categories": {c: categories.count(c) for c in set(categories) if c},
            "avg_engagement": sum(self._calculate_engagement_score(d) for d in designs) / len(designs),
            "high_performers": len([d for d in designs if self._calculate_engagement_score(d) > 500])
        }

    def _generate_inspiration_insights(self, query: str, designs: List[Dict[str, Any]], patterns: Dict[str, Any]) -> List[str]:
        """
        Generate inspiration insights
        """
        insights = []

        if patterns.get('popular_platforms'):
            top_platform = max(patterns['popular_platforms'], key=patterns['popular_platforms'].get)
            insights.append(f"'{query}' designs perform well on {top_platform}")

        if patterns.get('avg_engagement', 0) > 300:
            insights.append(f"High engagement potential: average {patterns['avg_engagement']:.0f} engagement points")

        if patterns.get('high_performers', 0) > 2:
            insights.append(f"{patterns['high_performers']} high-performing examples found for inspiration")

        return insights

    def _generate_actionable_tips(self, patterns: Dict[str, Any]) -> List[str]:
        """
        Generate actionable design tips
        """
        tips = []

        if patterns.get('popular_platforms', {}).get('pinterest', 0) > 2:
            tips.append("Consider Pinterest-style visual storytelling")

        if patterns.get('avg_engagement', 0) > 500:
            tips.append("Focus on high-contrast elements for better engagement")

        tips.append("Study the top 3 retrieved designs for style inspiration")
        tips.append("Maintain consistency with successful patterns in your niche")

        return tips

    def get_rag_status(self) -> Dict[str, Any]:
        """
        Get status of the RAG system
        """
        return {
            "database_loaded": len(self.design_database) > 0,
            "total_references": len(self.design_database),
            "indexed_keywords": len(self.metadata_index.get("by_keywords", {})),
            "indexed_platforms": len(self.metadata_index.get("by_platform", {})),
            "ready_for_analysis": len(self.design_database) > 0
        }


# Quick demo function
async def demo_integrated_rag():
    """
    Demo the integrated RAG system
    """
    rag = IntegratedVisualRAG()

    # Load design references
    await rag.load_design_references()

    # Check status
    status = rag.get_rag_status()
    print(f"🔗🧠 RAG Status: {status}")

    # Demo inspiration search
    inspiration = await rag.get_design_inspiration("modern logo design", ["minimalist", "clean"])
    print(f"💡 Found {len(inspiration['inspiration_designs'])} inspiration designs")


if __name__ == "__main__":
    asyncio.run(demo_integrated_rag())