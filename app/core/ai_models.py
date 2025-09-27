import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import asyncio
from typing import Optional, Dict, Any
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class AIModelManager:
    def __init__(self):
        self.processor: Optional[LlavaNextProcessor] = None
        self.model: Optional[LlavaNextForConditionalGeneration] = None
        self.device = settings.DEVICE
        self.model_name = settings.MODEL_NAME
        self.initialized = False

    async def initialize(self):
        """Initialize the VLM models asynchronously"""
        try:
            logger.info(f"Initializing AI models on device: {self.device}")

            # Load processor and model
            self.processor = LlavaNextProcessor.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )

            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=settings.MODEL_CACHE_DIR
            )

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")

            self.initialized = True
            logger.info("AI models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise

    async def analyze_design(self, image: Image.Image, prompt: str) -> str:
        """Analyze a design image using the VLM"""
        if not self.initialized:
            raise RuntimeError("AI models not initialized")

        try:
            # Prepare inputs
            inputs = self.processor(prompt, image, return_tensors="pt")

            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)

            # Extract only the generated part (after the prompt)
            prompt_length = len(prompt)
            if response.startswith(prompt):
                response = response[prompt_length:].strip()

            return response

        except Exception as e:
            logger.error(f"Error during design analysis: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("AI models cleaned up")


# Global instance
ai_manager = AIModelManager()