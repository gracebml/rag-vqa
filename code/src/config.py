"""
Configuration file for the RAG-enhanced VQA system
"""
import os
from pathlib import Path

# Detect if running on Kaggle
IS_KAGGLE = Path('/kaggle/working').exists()

# Base paths
if IS_KAGGLE:
    # Kaggle paths
    BASE_DIR = Path('/kaggle/working')
    CODE_DIR = BASE_DIR / "code"
    DATA_DIR = Path('/kaggle/input')
    MODELS_DIR = BASE_DIR / "models"
else:
    # Local paths
    BASE_DIR = Path(__file__).parent.parent.parent
    CODE_DIR = BASE_DIR / "code"
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"

# Model configurations

QWEN2VL_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
QWEN2VL_4BIT = True  # Use 4-bit quantization for T4 GPU

# Vietnamese embedding model for retrieval
VIETNAMESE_EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
# Alternative: "bkai-foundation-models/vietnamese-bi-encoder"

# OCR configuration
USE_PADDLE_OCR = True  # Set False to use Tesseract
PADDLE_OCR_LANG = "vietnamese"

# RAG configuration
RETRIEVAL_METHOD = "hybrid"  # "bm25", "embedding", or "hybrid"
TOP_K_RETRIEVE = 3
BM25_K1 = 1.5
BM25_B = 0.75

# Wikipedia configuration
WIKIPEDIA_LANG = "vi"  # Vietnamese Wikipedia
WIKIPEDIA_FALLBACK = True  # Use Wikipedia if local KB doesn't have results

# Knowledge base paths
if IS_KAGGLE:
    # Kaggle: adjust dataset name if different
    KB_JSON_PATH = Path('/kaggle/input/vietnamese-knowledge-base/knowledge_base.json')
    VQA_TEST_PATH = Path('/kaggle/input/vqa-test/vqa_test.json')
    # Vector DB can be from input dataset or working directory
    VECTOR_DB_PATH = Path('/kaggle/input/vqa-vector-db/vector_db') if Path('/kaggle/input/vqa-vector-db').exists() else MODELS_DIR / "vector_db"
else:
    # Local paths
    KB_JSON_PATH = DATA_DIR / "knowledge_base.json"
    VQA_TEST_PATH = DATA_DIR / "vqa_test.json"
    VECTOR_DB_PATH = MODELS_DIR / "vector_db"

# Generation parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4


# Device
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    # Fallback if torch not available yet
    DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

# Logging
LOG_LEVEL = "INFO"


# EVALUATION METRICS CONFIGURATION (

# BERTScore configuration
BERTSCORE_MODEL = "bert-base-multilingual-cased"
# Alternative models for Vietnamese:
# - "xlm-roberta-base" (better but slower)
# - "vinai/phobert-base" (Vietnamese specific)

# LLM-as-a-Judge configuration
GEMINI_MODEL = "gemini-1.5-flash"  # Free tier model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)  # Set via environment variable
LLM_JUDGE_DELAY = 1.0  # Delay between API calls (seconds) to avoid rate limit

# Evaluation output paths
if IS_KAGGLE:
    EVAL_OUTPUT_DIR = BASE_DIR / "results"
else:
    EVAL_OUTPUT_DIR = BASE_DIR / "results"

EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation settings
EVAL_USE_BERT_SCORE = True
EVAL_USE_LLM_JUDGE = True  # Set to False if no API key
EVAL_MAX_SAMPLES = None  # Set to int to limit samples, None for all

# Results filenames
PREDICTIONS_FILE = "predictions.json"
EVALUATION_RESULTS_FILE = "evaluation_results.json"
EVALUATION_METRICS_FILE = "metrics_summary.json"