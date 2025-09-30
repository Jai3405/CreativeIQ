#!/usr/bin/env python3
"""
Simple test to verify CLIP integration works
"""

import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image

def test_clip_import():
    """Test if we can import CLIP components"""
    try:
        import torch
        import clip
        print("✅ CLIP and PyTorch imported successfully")

        # Test CLIP model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"✅ CLIP model loaded on {device}")

        # Test basic embedding generation
        test_image = Image.new('RGB', (224, 224), (128, 128, 128))
        image_tensor = preprocess(test_image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        print(f"✅ Generated CLIP embedding: shape {embedding.shape}")
        print(f"✅ Embedding norm: {embedding.norm().item():.3f}")

        return True

    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        return False

def test_live_rag_initialization():
    """Test if LiveVisualRAG can be initialized with CLIP"""
    try:
        from app.services.live_visual_rag import LiveVisualRAG

        print("🔴 Testing LiveVisualRAG initialization...")
        rag = LiveVisualRAG()
        print("✅ LiveVisualRAG initialized successfully with CLIP")
        return True

    except Exception as e:
        print(f"❌ LiveVisualRAG initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 CLIP Integration Test")
    print("=" * 40)

    success = True

    print("\n1. Testing CLIP Import and Basic Functionality:")
    success &= test_clip_import()

    print("\n2. Testing LiveVisualRAG with CLIP:")
    success &= test_live_rag_initialization()

    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! CLIP integration is working.")
        print("🚀 Ready for full visual similarity ranking!")
    else:
        print("💥 Some tests failed. Check the errors above.")