from typing import List, Dict, Any
from PIL import Image
import numpy as np

from app.models.schemas import (
    DesignRecommendation, ColorPalette, TypographyAnalysis,
    LayoutAnalysis, VisualHierarchy
)


class RecommendationEngine:
    """
    AI-powered design recommendation generator
    """

    def __init__(self):
        self.recommendation_templates = {
            "color": {
                "low_harmony": {
                    "priority": "high",
                    "description": "Improve color harmony by using a more cohesive color scheme",
                    "technical": "Consider using complementary, triadic, or analogous color relationships"
                },
                "poor_contrast": {
                    "priority": "high",
                    "description": "Increase contrast between text and background for better readability",
                    "technical": "Aim for WCAG AA contrast ratio of 4.5:1 or higher"
                },
                "too_many_colors": {
                    "priority": "medium",
                    "description": "Reduce the number of colors to create a more focused design",
                    "technical": "Limit to 3-5 primary colors plus neutrals"
                }
            },
            "typography": {
                "poor_pairing": {
                    "priority": "high",
                    "description": "Improve font pairing by using complementary typefaces",
                    "technical": "Pair serif with sans-serif, or use different weights of the same font family"
                },
                "readability": {
                    "priority": "high",
                    "description": "Increase text readability by adjusting size, spacing, or contrast",
                    "technical": "Use minimum 14px font size, 1.5x line height, and sufficient color contrast"
                },
                "hierarchy": {
                    "priority": "medium",
                    "description": "Establish clearer typography hierarchy with varied font sizes",
                    "technical": "Use size ratios of 1.2-2.5x between hierarchy levels"
                }
            },
            "layout": {
                "poor_balance": {
                    "priority": "medium",
                    "description": "Improve visual balance by redistributing elements",
                    "technical": "Follow rule of thirds or golden ratio for element placement"
                },
                "weak_composition": {
                    "priority": "medium",
                    "description": "Strengthen composition by aligning elements to a grid system",
                    "technical": "Use 12-column or 16-column grid for consistent alignment"
                },
                "white_space": {
                    "priority": "low",
                    "description": "Optimize white space usage for better breathing room",
                    "technical": "Aim for 20-40% white space ratio in your design"
                }
            },
            "hierarchy": {
                "weak_focal_points": {
                    "priority": "medium",
                    "description": "Create stronger focal points to guide viewer attention",
                    "technical": "Use size, color, or contrast to emphasize key elements"
                },
                "poor_flow": {
                    "priority": "medium",
                    "description": "Improve visual flow to guide the viewer's eye naturally",
                    "technical": "Arrange elements to follow Z-pattern or F-pattern reading flow"
                }
            }
        }

    async def generate_recommendations(
        self, vlm_analysis: str, color_analysis: ColorPalette,
        typography_analysis: TypographyAnalysis, layout_analysis: LayoutAnalysis,
        hierarchy_analysis: VisualHierarchy
    ) -> List[DesignRecommendation]:
        """
        Generate comprehensive design recommendations
        """
        recommendations = []

        # Color recommendations
        color_recs = self._analyze_color_issues(color_analysis)
        recommendations.extend(color_recs)

        # Typography recommendations
        typography_recs = self._analyze_typography_issues(typography_analysis)
        recommendations.extend(typography_recs)

        # Layout recommendations
        layout_recs = self._analyze_layout_issues(layout_analysis)
        recommendations.extend(layout_recs)

        # Hierarchy recommendations
        hierarchy_recs = self._analyze_hierarchy_issues(hierarchy_analysis)
        recommendations.extend(hierarchy_recs)

        # VLM-based recommendations
        vlm_recs = self._parse_vlm_recommendations(vlm_analysis)
        recommendations.extend(vlm_recs)

        # Sort by priority and impact
        recommendations.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}[x.priority],
            x.impact_score
        ), reverse=True)

        return recommendations[:8]  # Return top 8 recommendations

    async def generate_variants(self, image: Image.Image, variant_count: int) -> List[Dict[str, Any]]:
        """
        Generate A/B test variants of the design
        """
        variants = []

        # Define variant strategies
        strategies = [
            {
                "name": "High Contrast",
                "description": "Increase contrast for better readability",
                "changes": ["Darken text", "Lighten background", "Increase color saturation"],
                "expected_improvement": "15-25% better readability"
            },
            {
                "name": "Simplified Color Palette",
                "description": "Reduce colors to 3-4 primary colors",
                "changes": ["Remove secondary colors", "Increase primary color dominance", "Add neutral tones"],
                "expected_improvement": "10-20% better focus"
            },
            {
                "name": "Enhanced Typography",
                "description": "Improve text hierarchy and readability",
                "changes": ["Increase font sizes", "Improve font pairing", "Add more white space"],
                "expected_improvement": "20-30% better engagement"
            },
            {
                "name": "Optimized Layout",
                "description": "Reorganize elements for better visual flow",
                "changes": ["Apply grid system", "Improve white space", "Strengthen focal points"],
                "expected_improvement": "15-25% better user flow"
            },
            {
                "name": "Mobile Optimized",
                "description": "Optimize for mobile viewing",
                "changes": ["Larger touch targets", "Simplified layout", "Increased font sizes"],
                "expected_improvement": "25-35% better mobile performance"
            }
        ]

        # Generate requested number of variants
        for i in range(min(variant_count, len(strategies))):
            strategy = strategies[i]

            variant = {
                "variant_id": f"variant_{i+1}",
                "strategy": strategy["name"],
                "description": strategy["description"],
                "changes": strategy["changes"],
                "expected_improvement": strategy["expected_improvement"],
                "confidence": self._calculate_variant_confidence(strategy),
                "implementation_difficulty": self._assess_implementation_difficulty(strategy),
                "estimated_time": self._estimate_implementation_time(strategy)
            }

            variants.append(variant)

        return variants

    def _analyze_color_issues(self, color_analysis: ColorPalette) -> List[DesignRecommendation]:
        """
        Analyze color-related issues and generate recommendations
        """
        recommendations = []

        # Check harmony score
        if color_analysis.harmony_score < 70:
            template = self.recommendation_templates["color"]["low_harmony"]
            recommendations.append(DesignRecommendation(
                category="color",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(85 - color_analysis.harmony_score)
            ))

        # Check accessibility
        if color_analysis.accessibility_score < 70:
            template = self.recommendation_templates["color"]["poor_contrast"]
            recommendations.append(DesignRecommendation(
                category="color",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(90 - color_analysis.accessibility_score)
            ))

        # Check color count
        if len(color_analysis.dominant_colors) > 6:
            template = self.recommendation_templates["color"]["too_many_colors"]
            recommendations.append(DesignRecommendation(
                category="color",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(len(color_analysis.dominant_colors) * 5)
            ))

        return recommendations

    def _analyze_typography_issues(self, typography_analysis: TypographyAnalysis) -> List[DesignRecommendation]:
        """
        Analyze typography issues and generate recommendations
        """
        recommendations = []

        # Check font pairing
        if typography_analysis.font_pairing_score < 70:
            template = self.recommendation_templates["typography"]["poor_pairing"]
            recommendations.append(DesignRecommendation(
                category="typography",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(80 - typography_analysis.font_pairing_score)
            ))

        # Check readability
        if typography_analysis.readability_score < 70:
            template = self.recommendation_templates["typography"]["readability"]
            recommendations.append(DesignRecommendation(
                category="typography",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(85 - typography_analysis.readability_score)
            ))

        # Check hierarchy
        if typography_analysis.text_hierarchy_score < 70:
            template = self.recommendation_templates["typography"]["hierarchy"]
            recommendations.append(DesignRecommendation(
                category="typography",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(75 - typography_analysis.text_hierarchy_score)
            ))

        return recommendations

    def _analyze_layout_issues(self, layout_analysis: LayoutAnalysis) -> List[DesignRecommendation]:
        """
        Analyze layout issues and generate recommendations
        """
        recommendations = []

        # Check balance
        if layout_analysis.balance_score < 70:
            template = self.recommendation_templates["layout"]["poor_balance"]
            recommendations.append(DesignRecommendation(
                category="layout",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(75 - layout_analysis.balance_score)
            ))

        # Check composition
        if layout_analysis.composition_score < 70:
            template = self.recommendation_templates["layout"]["weak_composition"]
            recommendations.append(DesignRecommendation(
                category="layout",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(80 - layout_analysis.composition_score)
            ))

        # Check white space
        if layout_analysis.white_space_usage < 60 or layout_analysis.white_space_usage > 80:
            template = self.recommendation_templates["layout"]["white_space"]
            optimal_white_space = 70
            deviation = abs(layout_analysis.white_space_usage - optimal_white_space)
            recommendations.append(DesignRecommendation(
                category="layout",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(deviation)
            ))

        return recommendations

    def _analyze_hierarchy_issues(self, hierarchy_analysis: VisualHierarchy) -> List[DesignRecommendation]:
        """
        Analyze visual hierarchy issues and generate recommendations
        """
        recommendations = []

        # Check hierarchy score
        if hierarchy_analysis.hierarchy_score < 70:
            template = self.recommendation_templates["hierarchy"]["weak_focal_points"]
            recommendations.append(DesignRecommendation(
                category="hierarchy",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(80 - hierarchy_analysis.hierarchy_score)
            ))

        # Check focal points
        if len(hierarchy_analysis.attention_areas) < 2:
            template = self.recommendation_templates["hierarchy"]["poor_flow"]
            recommendations.append(DesignRecommendation(
                category="hierarchy",
                priority=template["priority"],
                description=template["description"],
                technical_details=template["technical"],
                impact_score=self._calculate_impact_score(60)
            ))

        return recommendations

    def _parse_vlm_recommendations(self, vlm_analysis: str) -> List[DesignRecommendation]:
        """
        Parse VLM analysis for additional recommendations
        """
        recommendations = []

        # Simple keyword-based parsing (in production, use more sophisticated NLP)
        analysis_lower = vlm_analysis.lower()

        if "contrast" in analysis_lower and ("low" in analysis_lower or "poor" in analysis_lower):
            recommendations.append(DesignRecommendation(
                category="accessibility",
                priority="high",
                description="AI analysis suggests improving contrast for better accessibility",
                technical_details="Increase contrast ratio between text and background elements",
                impact_score=75
            ))

        if "cluttered" in analysis_lower or "busy" in analysis_lower:
            recommendations.append(DesignRecommendation(
                category="simplification",
                priority="medium",
                description="AI analysis suggests simplifying the design to reduce visual clutter",
                technical_details="Remove non-essential elements and increase white space",
                impact_score=60
            ))

        if "font" in analysis_lower and ("mix" in analysis_lower or "many" in analysis_lower):
            recommendations.append(DesignRecommendation(
                category="typography",
                priority="medium",
                description="AI analysis suggests reducing the number of fonts used",
                technical_details="Limit to 2-3 font families maximum for better consistency",
                impact_score=55
            ))

        return recommendations

    def _calculate_impact_score(self, deficit: float) -> float:
        """
        Calculate impact score based on how much improvement is needed
        """
        # Higher deficit = higher impact potential
        impact = min(100, max(20, deficit * 1.5))
        return impact

    def _calculate_variant_confidence(self, strategy: Dict[str, Any]) -> float:
        """
        Calculate confidence in variant success
        """
        # Base confidence on strategy type and expected improvement
        base_confidence = 70

        if "contrast" in strategy["name"].lower():
            base_confidence = 85  # High confidence in contrast improvements
        elif "typography" in strategy["name"].lower():
            base_confidence = 80  # High confidence in typography improvements
        elif "mobile" in strategy["name"].lower():
            base_confidence = 75  # Medium-high confidence in mobile optimizations

        return base_confidence

    def _assess_implementation_difficulty(self, strategy: Dict[str, Any]) -> str:
        """
        Assess implementation difficulty for variant
        """
        name = strategy["name"].lower()

        if "color" in name:
            return "Easy"
        elif "typography" in name:
            return "Medium"
        elif "layout" in name:
            return "Hard"
        elif "mobile" in name:
            return "Medium"
        else:
            return "Medium"

    def _estimate_implementation_time(self, strategy: Dict[str, Any]) -> str:
        """
        Estimate implementation time for variant
        """
        difficulty = self._assess_implementation_difficulty(strategy)

        time_estimates = {
            "Easy": "1-2 hours",
            "Medium": "3-5 hours",
            "Hard": "6-10 hours"
        }

        return time_estimates.get(difficulty, "3-5 hours")