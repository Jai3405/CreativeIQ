"""
On-Demand Image Fetcher for Reference-Based Data
Fetches images only when needed for analysis
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image
import io
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OnDemandImageFetcher:
    """
    Fetches images on-demand from URLs when needed for analysis
    Provides temporary caching for performance
    """

    def __init__(self, cache_dir: Path = None, cache_duration_hours: int = 24):
        self.cache_dir = cache_dir or Path("data/training/image_cache")
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Session for reuse
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'CreativeIQ/1.0 Design Analysis Tool'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def fetch_image_for_analysis(self, reference: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Fetch image from reference for immediate analysis
        Returns PIL Image object ready for computer vision processing
        """
        image_url = reference.get("image_url")
        reference_id = reference.get("id")

        if not image_url or not reference_id:
            logger.warning(f"Invalid reference: missing URL or ID")
            return None

        try:
            # Check cache first
            cached_image = await self._get_cached_pil_image(reference_id)
            if cached_image:
                logger.info(f"📁 Using cached image for {reference_id}")
                return cached_image

            # Fetch from URL
            image_data = await self._fetch_image_data(image_url)
            if not image_data:
                return None

            # Convert to PIL Image
            try:
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image = pil_image.convert('RGB')  # Ensure RGB format

                # Cache the raw data
                await self._cache_image_data(reference_id, image_data)

                logger.info(f"🖼️ Fetched and processed image for {reference_id}")
                return pil_image

            except Exception as e:
                logger.error(f"Failed to process image data for {reference_id}: {e}")
                return None

        except Exception as e:
            logger.error(f"Error fetching image for analysis {reference_id}: {e}")
            return None

    async def fetch_multiple_images(self, references: list, max_concurrent: int = 5) -> Dict[str, Optional[Image.Image]]:
        """
        Fetch multiple images concurrently with rate limiting
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_single(ref):
            async with semaphore:
                return ref['id'], await self.fetch_image_for_analysis(ref)

        tasks = [fetch_single(ref) for ref in references]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        images = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in concurrent fetch: {result}")
                continue

            ref_id, image = result
            images[ref_id] = image

        logger.info(f"🔄 Fetched {len([img for img in images.values() if img is not None])}/{len(references)} images successfully")
        return images

    async def _fetch_image_data(self, image_url: str) -> Optional[bytes]:
        """
        Fetch raw image data from URL
        """
        if not self.session:
            logger.error("Session not initialized. Use async context manager.")
            return None

        try:
            async with self.session.get(image_url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')

                    # Validate content type
                    if not content_type.startswith('image/'):
                        logger.warning(f"Invalid content type for image: {content_type}")
                        return None

                    image_data = await response.read()

                    # Validate image size (prevent huge downloads)
                    if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                        logger.warning(f"Image too large: {len(image_data)} bytes")
                        return None

                    return image_data

                else:
                    logger.warning(f"Failed to fetch image: HTTP {response.status}")
                    return None

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching image from {image_url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching image data: {e}")
            return None

    async def _get_cached_pil_image(self, reference_id: str) -> Optional[Image.Image]:
        """
        Get cached PIL image if available and not expired
        """
        cache_file = self.cache_dir / f"{reference_id}.jpg"

        if cache_file.exists():
            # Check if cache is still valid
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time < self.cache_duration:
                try:
                    pil_image = Image.open(cache_file)
                    return pil_image.convert('RGB')
                except Exception as e:
                    logger.warning(f"Failed to load cached image {reference_id}: {e}")
                    # Remove corrupted cache
                    cache_file.unlink()

        return None

    async def _cache_image_data(self, reference_id: str, image_data: bytes):
        """
        Cache image data to disk
        """
        try:
            cache_file = self.cache_dir / f"{reference_id}.jpg"

            # Convert to PIL and save as JPEG (consistent format)
            pil_image = Image.open(io.BytesIO(image_data))
            pil_image = pil_image.convert('RGB')

            # Resize if too large (for faster future loading)
            if pil_image.size[0] > 1920 or pil_image.size[1] > 1920:
                pil_image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)

            pil_image.save(cache_file, 'JPEG', quality=85, optimize=True)

        except Exception as e:
            logger.warning(f"Failed to cache image {reference_id}: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the image cache
        """
        cache_files = list(self.cache_dir.glob("*.jpg"))

        total_size = sum(f.stat().st_size for f in cache_files)

        # Check expired files
        expired_count = 0
        for cache_file in cache_files:
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time > self.cache_duration:
                expired_count += 1

        return {
            "cached_images": len(cache_files),
            "total_cache_size_mb": round(total_size / (1024 * 1024), 2),
            "expired_files": expired_count,
            "cache_duration_hours": self.cache_duration.total_seconds() / 3600,
            "cache_directory": str(self.cache_dir)
        }

    async def clean_expired_cache(self):
        """
        Clean expired cache files
        """
        cache_files = list(self.cache_dir.glob("*.jpg"))
        cleaned_count = 0

        for cache_file in cache_files:
            try:
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - cache_time > self.cache_duration:
                    cache_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to clean cache file {cache_file}: {e}")

        logger.info(f"🧹 Cleaned {cleaned_count} expired cache files")
        return cleaned_count

    async def preload_images(self, references: list, max_concurrent: int = 3):
        """
        Preload images into cache (background operation)
        """
        logger.info(f"🔄 Preloading {len(references)} images into cache...")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def preload_single(ref):
            async with semaphore:
                # Check if already cached
                cached = await self._get_cached_pil_image(ref['id'])
                if cached:
                    return True

                # Fetch and cache
                image_data = await self._fetch_image_data(ref['image_url'])
                if image_data:
                    await self._cache_image_data(ref['id'], image_data)
                    return True
                return False

        tasks = [preload_single(ref) for ref in references]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        logger.info(f"✅ Preloaded {success_count}/{len(references)} images")


# Example usage function
async def demo_on_demand_fetching():
    """
    Demo of on-demand image fetching
    """
    # Sample references (in practice, these come from the reference collector)
    sample_references = [
        {
            "id": "demo_pinterest_1",
            "platform": "pinterest",
            "image_url": "https://i.pinimg.com/564x/example1.jpg",
            "title": "Modern Logo Design"
        },
        {
            "id": "demo_instagram_1",
            "platform": "instagram",
            "image_url": "https://example.com/design.jpg",
            "title": "UI Design Inspiration"
        }
    ]

    async with OnDemandImageFetcher() as fetcher:
        # Fetch single image
        image = await fetcher.fetch_image_for_analysis(sample_references[0])
        if image:
            print(f"✅ Fetched image: {image.size}")

        # Fetch multiple images
        images = await fetcher.fetch_multiple_images(sample_references)
        print(f"📊 Fetched {len(images)} images")

        # Cache info
        cache_info = fetcher.get_cache_info()
        print(f"💾 Cache: {cache_info['cached_images']} images, {cache_info['total_cache_size_mb']} MB")


if __name__ == "__main__":
    asyncio.run(demo_on_demand_fetching())