from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Model names (you can change based on what you install)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"

# For clustering
N_CLUSTERS = 5   # adjust after exploring

# GPT (if you use openai library)
OPENAI_MODEL_NAME = "gpt-4.1-mini"  # or whatever you have access to