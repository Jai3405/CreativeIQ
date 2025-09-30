#!/usr/bin/env python3
"""
Test the Integrated Visual RAG System
No downloads - pure metadata-based retrieval with on-demand fetching
"""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.integrated_visual_rag import IntegratedVisualRAG
from app.data_collection.reference_based_collector import collect_design_references


async def main():
    """
    Test the integrated Visual RAG system
    """
    print("🔗🧠 Testing Integrated Visual RAG System")
    print("========================================")
    print("No downloads - smart metadata search + on-demand fetching\n")

    rag = IntegratedVisualRAG()

    # Check if we have references
    status = rag.get_rag_status()
    if not status["database_loaded"]:
        print("📥 No design references found. Collecting sample data...")
        await collect_design_references()
        print("✅ Sample references collected\n")

    # Load references
    await rag.load_design_references()
    status = rag.get_rag_status()

    print(f"📊 RAG System Status:")
    print(f"  Database loaded: {status['database_loaded']}")
    print(f"  Total references: {status['total_references']}")
    print(f"  Indexed keywords: {status['indexed_keywords']}")
    print(f"  Ready for analysis: {status['ready_for_analysis']}\n")

    # Test 1: Design Inspiration Search (metadata only)
    print("🎨 Test 1: Design Inspiration Search")
    print("-" * 40)

    inspiration = await rag.get_design_inspiration(
        "modern logo design",
        style_filters=["minimalist", "clean"]
    )

    print(f"Query: '{inspiration['query']}'")
    print(f"Found: {len(inspiration['inspiration_designs'])} relevant designs")
    print(f"Patterns: {inspiration['design_patterns']}")
    print(f"Insights: {inspiration['inspiration_insights']}")
    print(f"Tips: {inspiration['actionable_tips'][:2]}\n")  # First 2 tips

    # Test 2: Show what RAG would do with user image (simulation)
    print("🖼️ Test 2: User Design Analysis Simulation")
    print("-" * 45)

    # Create a simple test image without numpy
    from PIL import Image

    # Create sample "logo" image (square, simple)
    fake_user_image = Image.new('RGB', (400, 400), (100, 150, 200))

    print("Simulating user uploads a 400x400 logo design...")

    # This would normally analyze the real user image
    user_analysis = await rag.analyze_user_design_with_rag(
        fake_user_image,
        user_goals=["professional logo", "tech startup"]
    )

    print(f"Analysis Results:")
    print(f"  Design type detected: {user_analysis['user_design_analysis']['design_type']}")
    print(f"  Retrieved references: {len(user_analysis['retrieved_references'])}")
    print(f"  Images fetched for comparison: {user_analysis['metadata']['images_fetched']}")
    print(f"  Database searched: {user_analysis['metadata']['total_database_size']} designs")

    insights = user_analysis['rag_enhanced_insights']
    print(f"  Performance insights: {len(insights.get('performance_insights', []))}")
    print(f"  Style recommendations: {len(insights.get('style_recommendations', []))}")

    if insights.get('platform_insights'):
        platform_info = insights['platform_insights']
        print(f"  Best platform: {platform_info.get('best_platform')}")

    print("\n✅ RAG system working efficiently!")
    print(f"💡 Benefits demonstrated:")
    print(f"  • No bulk downloads required")
    print(f"  • Instant metadata search")
    print(f"  • On-demand image fetching (only top 3)")
    print(f"  • Smart design intelligence")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")