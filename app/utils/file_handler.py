from fastapi import UploadFile
from PIL import Image
import io
import os
from typing import Optional
from app.core.config import settings


def validate_image_file(file: UploadFile) -> bool:
    """
    Validate uploaded image file
    """
    # Check file extension
    if not file.filename:
        return False

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        return False

    # Check file size (UploadFile doesn't have size attribute directly)
    # This would be checked during actual file processing

    return True


async def process_uploaded_image(file: UploadFile) -> Image.Image:
    """
    Process uploaded image file and return PIL Image
    """
    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > settings.MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes")

    # Create PIL Image
    try:
        image = Image.open(io.BytesIO(content))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def save_image(image: Image.Image, filename: str) -> str:
    """
    Save image to upload directory
    """
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(settings.UPLOAD_DIR, filename)

    image.save(filepath)
    return filepath


def preprocess_image_for_analysis(image: Image.Image, target_size: tuple = (512, 512)) -> Image.Image:
    """
    Preprocess image for analysis (resize, normalize, etc.)
    """
    # Resize while maintaining aspect ratio
    image.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Create new image with target size and paste the resized image
    processed_image = Image.new('RGB', target_size, (255, 255, 255))

    # Calculate position to center the image
    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2

    processed_image.paste(image, (x, y))

    return processed_image