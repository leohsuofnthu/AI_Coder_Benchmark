"""
Pre-download Sentence Transformer Model for Render Deployment
This script downloads the model during build time to avoid runtime delays.
"""

import os
from sentence_transformers import SentenceTransformer

# Model name - using fast CPU-optimized model (same as hierarchy_evaluator.py)
# all-MiniLM-L6-v2: ~90MB, 2-5x faster than e5-base on CPU, good quality
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def download_model():
    """Download and cache the sentence transformer model."""
    print(f"[MODEL] Starting model download: {MODEL_NAME}")
    print(f"[MODEL] This may take 1-2 minutes on first run...")
    
    try:
        # Download model (will be cached in ~/.cache/huggingface/)
        model = SentenceTransformer(MODEL_NAME)
        print(f"[MODEL] ✓ Model downloaded successfully")
        print(f"[MODEL] Model device: {model.device}")
        print(f"[MODEL] Model max sequence length: {model.max_seq_length}")
        
        # Test encoding to verify model works
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text, show_progress_bar=False)
        print(f"[MODEL] ✓ Test encoding successful (embedding dimension: {len(embedding)})")
        
        return True
    except Exception as e:
        print(f"[MODEL] ✗ Error downloading model: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        print("[MODEL] Warning: Model download failed. It will be downloaded on first use.")
        exit(0)  # Don't fail build, model will download on first use
    else:
        print("[MODEL] Model ready for use!")

