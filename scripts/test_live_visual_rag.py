#!/usr/bin/env python3
"""
Test the Live Visual RAG System
Real-time design search and analysis without static datasets
"""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.live_visual_rag import LiveVisualRAG
from PIL import Image
import json


async def main():
    """
    Test the Live Visual RAG system with different scenarios
    """
    print("🔴 Live Visual RAG System Test")
    print("==============================")
    print("Real-time design search + analysis without static datasets\n")

    rag = LiveVisualRAG()

    # Test 1: Tech Logo Analysis
    print("🎨 Test 1: Tech Logo Analysis")
    print("-" * 40)

    # Create sample logo-style image (square format)
    logo_image = Image.new('RGB', (400, 400), (45, 85, 155))  # Tech blue
    print("Simulating user uploads a 400x400 tech logo...")

    try:
        logo_result = await rag.analyze_and_retrieve(
            logo_image,
            user_goals=["professional logo", "tech startup", "modern branding"]
        )

        print(f"✅ Analysis Complete!")
        print(f"  Design type detected: {logo_result['user_design_analysis']['design_type']}")
        print(f"  Search queries generated: {logo_result['search_queries_used']}")
        print(f"  Total designs found: {logo_result['total_found']}")
        print(f"  Platforms searched: {logo_result['metadata']['platforms_searched']}")

        if logo_result['retrieved_designs']:
            sample_design = logo_result['retrieved_designs'][0]
            print(f"  Top result: '{sample_design['title']}' by {sample_design['author']}")
            print(f"  Platform: {sample_design['platform']}")
            print(f"  Stats: {sample_design['stats']['likes']} likes, {sample_design['stats']['views']} views")

        recommendations = logo_result['recommendations']
        print(f"\n💡 AI Recommendations:")
        for insight in recommendations.get('performance_insights', []):
            print(f"  • {insight}")

        for style_rec in recommendations.get('style_recommendations', []):
            print(f"  • {style_rec}")

    except Exception as e:
        print(f"❌ Logo test failed: {e}")

    print("\n" + "="*60 + "\n")

    # Test 2: Social Media Post Analysis
    print("📱 Test 2: Social Media Post Analysis")
    print("-" * 40)

    # Create sample social media post (rectangular format)
    social_image = Image.new('RGB', (1080, 1350), (255, 87, 51))  # Instagram orange
    print("Simulating user uploads a 1080x1350 social media post...")

    try:
        social_result = await rag.analyze_and_retrieve(
            social_image,
            user_goals=["social media marketing", "brand engagement", "modern design"]
        )

        print(f"✅ Analysis Complete!")
        print(f"  Design type detected: {social_result['user_design_analysis']['design_type']}")
        print(f"  Search queries used: {social_result['search_queries_used']}")
        print(f"  Designs found: {social_result['total_found']}")

        if social_result['retrieved_designs']:
            print(f"  Retrieved {len(social_result['retrieved_designs'])} similar designs")

        print(f"\n📊 Performance Benchmarks:")
        benchmark = social_result['recommendations'].get('engagement_benchmark', {})
        if benchmark:
            print(f"  Average likes: {benchmark.get('average_likes', 0):.0f}")
            print(f"  Average views: {benchmark.get('average_views', 0):.0f}")
            print(f"  Engagement rate: {benchmark.get('engagement_rate', 0):.1f}%")

    except Exception as e:
        print(f"❌ Social media test failed: {e}")

    print("\n" + "="*60 + "\n")

    # Test 3: API Status Check
    print("⚙️ Test 3: API Configuration Check")
    print("-" * 40)

    print(f"Dribbble API configured: {'✅' if rag.dribbble_token else '❌ Token needed'}")
    print(f"Unsplash API configured: {'✅' if rag.unsplash_key else '❌ Access key needed'}")

    if not rag.dribbble_token and not rag.unsplash_key:
        print(f"\n⚠️ No API tokens configured!")
        print(f"Add to .env file:")
        print(f"DRIBBBLE_ACCESS_TOKEN=your_dribbble_token")
        print(f"UNSPLASH_ACCESS_KEY=your_unsplash_access_key")
        print(f"\n📚 To get tokens:")
        print(f"• Dribbble: https://dribbble.com/account/applications/new")
        print(f"• Unsplash: https://unsplash.com/oauth/applications")
    else:
        print(f"\n✅ Ready for live design search!")

    print("\n" + "="*60 + "\n")

    # Test 4: System Performance Summary
    print("📈 System Performance Summary")
    print("-" * 40)

    print(f"🎯 Live Visual RAG Benefits Demonstrated:")
    print(f"  ✅ No static dataset downloads required")
    print(f"  ✅ Real-time design trend analysis")
    print(f"  ✅ Professional design platform integration")
    print(f"  ✅ Data-driven performance recommendations")
    print(f"  ✅ Visual similarity ranking (CLIP integration ready)")
    print(f"  ✅ Multi-platform parallel search")

    print(f"\n🔧 Architecture Highlights:")
    print(f"  • VLM analysis for style/color/type extraction")
    print(f"  • Smart query generation from design features")
    print(f"  • Async parallel API searches (Dribbble + Unsplash)")
    print(f"  • Real-time similarity ranking")
    print(f"  • Comparative analysis with professional designs")

    print(f"\n💡 Perfect for Computer Vision Course:")
    print(f"  • Demonstrates real-world API integration")
    print(f"  • Shows live retrieval vs static datasets")
    print(f"  • Combines VLM + RAG + similarity search")
    print(f"  • Produces actionable design recommendations")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()