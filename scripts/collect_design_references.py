#!/usr/bin/env python3
"""
CreativeIQ Reference-Based Design Data Collection
Collects design URLs and metadata instead of downloading images
Much faster and more efficient for large-scale collection
"""

import asyncio
import argparse
import sys
from pathlib import Path
import logging

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.data_collection.reference_based_collector import collect_design_references
from app.services.on_demand_image_fetcher import OnDemandImageFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main reference-based design data collection
    """
    parser = argparse.ArgumentParser(description="Collect design references (URLs + metadata) for CreativeIQ")
    parser.add_argument("--platforms", nargs="+", default=["instagram", "pinterest", "dribbble"],
                       help="Platforms to collect from")
    parser.add_argument("--test-fetch", action="store_true",
                       help="Test fetching a few images on-demand after collection")
    parser.add_argument("--preload", type=int, default=0,
                       help="Number of images to preload into cache")
    parser.add_argument("--cache-info", action="store_true",
                       help="Show image cache information")

    args = parser.parse_args()

    print("🔗 CreativeIQ Reference-Based Data Collection")
    print("=============================================")
    print("Collecting design URLs and metadata (no downloads)")
    print("Images fetched on-demand during analysis\n")

    # Show cache info if requested
    if args.cache_info:
        await show_cache_info()
        return 0

    try:
        # Collect design references
        print("📊 Collecting design references...")
        result = await collect_design_references()

        if result["references"]:
            references = result["references"]
            stats = result["statistics"]

            print("\n🎉 Reference collection completed!")
            print(f"📈 Collection Statistics:")
            print(f"  Total references: {stats['total_references']}")
            print(f"  Platforms: {dict(stats['platform_distribution'])}")
            print(f"  Quality score: {stats['quality_score']}/100")
            print(f"  Valid URLs: {stats['url_validity']['valid_urls']}/{stats['total_references']}")

            # Show benefits of reference-based approach
            print(f"\n💡 Reference-Based Benefits:")
            for benefit in result["benefits"]:
                print(f"  • {benefit}")

            # Storage comparison
            estimated_download_size = len(references) * 500  # KB per image
            print(f"\n💾 Storage Efficiency:")
            print(f"  References dataset: ~{len(references) * 2} KB")
            print(f"  vs Full downloads: ~{estimated_download_size} KB")
            print(f"  Space saved: ~{estimated_download_size - len(references) * 2} KB ({((estimated_download_size - len(references) * 2) / estimated_download_size * 100):.1f}%)")

            # Test on-demand fetching if requested
            if args.test_fetch:
                await test_on_demand_fetching(references[:3])

            # Preload some images if requested
            if args.preload > 0:
                await preload_images(references[:args.preload])

            # Show next steps
            print(f"\n🎯 Next Steps:")
            print(f"  1. Review references: data/training/references/")
            print(f"  2. Test analysis: python -m app.services.design_analyzer")
            print(f"  3. Train models: python scripts/train_models.py --mode full")

            return 0
        else:
            print("\n❌ No design references collected!")
            return 1

    except Exception as e:
        print(f"\n💥 Error during collection: {e}")
        return 1


async def test_on_demand_fetching(sample_references: list):
    """
    Test on-demand image fetching with sample references
    """
    print(f"\n🧪 Testing on-demand image fetching...")

    async with OnDemandImageFetcher() as fetcher:
        success_count = 0

        for ref in sample_references:
            print(f"  Fetching: {ref['id']} from {ref['platform']}")

            image = await fetcher.fetch_image_for_analysis(ref)
            if image:
                print(f"    ✅ Success: {image.size} pixels")
                success_count += 1
            else:
                print(f"    ❌ Failed to fetch")

        print(f"\n📊 Fetch Test Results: {success_count}/{len(sample_references)} successful")

        # Show cache info
        cache_info = fetcher.get_cache_info()
        print(f"💾 Cache: {cache_info['cached_images']} images, {cache_info['total_cache_size_mb']} MB")


async def preload_images(references: list):
    """
    Preload images into cache for faster future access
    """
    print(f"\n🔄 Preloading {len(references)} images into cache...")

    async with OnDemandImageFetcher() as fetcher:
        await fetcher.preload_images(references)

        cache_info = fetcher.get_cache_info()
        print(f"✅ Preloading complete")
        print(f"💾 Cache: {cache_info['cached_images']} images, {cache_info['total_cache_size_mb']} MB")


async def show_cache_info():
    """
    Show current image cache information
    """
    print("💾 Image Cache Information")
    print("=" * 30)

    async with OnDemandImageFetcher() as fetcher:
        cache_info = fetcher.get_cache_info()

        print(f"Cached images: {cache_info['cached_images']}")
        print(f"Total cache size: {cache_info['total_cache_size_mb']} MB")
        print(f"Expired files: {cache_info['expired_files']}")
        print(f"Cache duration: {cache_info['cache_duration_hours']} hours")
        print(f"Cache directory: {cache_info['cache_directory']}")

        # Clean expired cache
        if cache_info['expired_files'] > 0:
            print(f"\n🧹 Cleaning {cache_info['expired_files']} expired files...")
            cleaned = await fetcher.clean_expired_cache()
            print(f"✅ Cleaned {cleaned} expired cache files")


def show_reference_based_guide():
    """
    Show guide for reference-based data collection
    """
    print("\n📖 Reference-Based Collection Guide")
    print("=" * 40)
    print()
    print("🔗 What is Reference-Based Collection?")
    print("  • Collects URLs and metadata instead of downloading images")
    print("  • Images fetched on-demand during analysis")
    print("  • Much faster collection and smaller storage footprint")
    print("  • Always accesses fresh data from original sources")
    print()
    print("💡 Benefits:")
    print("  ✅ 90%+ faster collection (no downloads)")
    print("  ✅ 95%+ smaller storage requirements")
    print("  ✅ Real-time access to fresh content")
    print("  ✅ No copyright/storage concerns")
    print("  ✅ Scalable to millions of references")
    print()
    print("🛠️ How It Works:")
    print("  1. Collect URLs + metadata from design platforms")
    print("  2. Store references in JSON files (lightweight)")
    print("  3. Fetch images on-demand during AI analysis")
    print("  4. Temporary caching for performance")
    print()
    print("🎯 Perfect for:")
    print("  • Large-scale design data collection")
    print("  • Computer vision training datasets")
    print("  • Real-time design analysis")
    print("  • Research and development")
    print()
    print("🚀 Commands:")
    print("  python scripts/collect_design_references.py")
    print("  python scripts/collect_design_references.py --test-fetch")
    print("  python scripts/collect_design_references.py --cache-info")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--guide":
        show_reference_based_guide()
        sys.exit(0)

    print("🔗 CreativeIQ Reference-Based Collection")
    print("=======================================")
    print("Efficient design data collection via URLs")
    print("Images fetched on-demand for analysis\n")

    try:
        exit_code = asyncio.run(main())
        if exit_code == 0:
            print("\n🎉 Reference collection completed!")
            print("Ready for AI training and analysis.")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        print("\nRun with --guide for more information")
        sys.exit(1)