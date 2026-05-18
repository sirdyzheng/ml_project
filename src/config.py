"""All hyperparameters and paths in one place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CHUNK_SIZE = 120
CHUNK_OVERLAP = 30
STM_CAPACITY = 40
LTM_MAX_SIZE = 120
CONSOLIDATION_FREQ = 5
PROMOTION_TOP_K = 3
SCORE_THRESHOLD = 0.35
DUPLICATE_SIM_THRESHOLD = 0.85
RETRIEVAL_TOP_K = 6
STM_TOP_K = 4
LTM_TOP_K = 4

SCORE_WEIGHTS = {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42
DEV_SIZE = 50
TEST_SIZE = 150
TOTAL_SAMPLE = DEV_SIZE + TEST_SIZE
EXPERIMENT_SEEDS = [42, 43, 44]

HISTORY_KEYWORDS = [
    "before", "earlier", "previously", "first", "ago",
    "last time", "back then", "initially", "originally",
]

ROUTING_SIM_MARGIN = 0.05
